import functools
import io
import logging
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

from antlr4 import FileStream
from antlr4.Token import CommonToken
from antlr4.tree.Tree import ParseTree, TerminalNode
from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer
from pygls.uris import from_fs_path, to_fs_path
from pygls.workspace import TextDocument
from typeshed_client import SearchContext

from core.analysis import PythonAnalyzer
from core.source import PythonSource, UriSourceStream
from grammar import PythonParser
from semantics.base import SemanticError
from semantics.entity import PyClass, PyFunction, PyModule, PyParameters
from semantics.structure import PyFunctionCall, PythonContext
from semantics.symbol import Symbol
from semantics.token import TokenKind, TokenModifier, is_blank_token, is_synthetic_token
from semantics.types import (
    PyArguments,
    PyClassType,
    PyFunctionType,
    PyGenericAlias,
    PyModuleType,
    PyType,
    PyTypeVarDef,
    match_arguments_to_parameters,
)

from .utils import (
    collect_argument_commas,
    find_ancestor_node,
    get_argument_index,
    is_inside_arguments,
    node_at_token_index,
    snake_case_to_camel_case,
    token_at_position,
    token_end_position,
    token_start_position,
)

TOKEN_KINDS = [kind.value for kind in TokenKind]
TOKEN_MODIFIERS = [
    snake_case_to_camel_case(modifier.name) for modifier in TokenModifier
]

PYTHON_KEYWORDS = [
    "False",
    "await",
    "else",
    "import",
    "pass",
    "None",
    "break",
    "except",
    "in",
    "raise",
    "True",
    "class",
    "finally",
    "is",
    "return",
    "and",
    "continue",
    "for",
    "lambda",
    "try",
    "as",
    "def",
    "from",
    "nonlocal",
    "while",
    "assert",
    "del",
    "global",
    "not",
    "with",
    "async",
    "elif",
    "if",
    "or",
    "yield",
]

NON_COMPLETION_TOKENS = {
    PythonParser.COMMENT,
    PythonParser.EXPLICIT_LINE_JOINING,
    PythonParser.STRING_LITERAL,
    PythonParser.BYTES_LITERAL,
    PythonParser.INTEGER,
    PythonParser.FLOAT_NUMBER,
    PythonParser.IMAG_NUMBER,
}
"""Tokens that should not trigger completions."""


logger = logging.getLogger(__name__)


class DocumentDiagnostics(NamedTuple):
    version: Optional[int]
    dianogstics: list[lsp.Diagnostic]


class PythonLanguageServer(LanguageServer):
    """
    A language server for Python.
    """

    def __init__(self):
        super().__init__("python-language-server", "v0.1.0")

        self.analyzer: PythonAnalyzer
        self.root_paths: list[Path] = []
        self.search_context: Optional[SearchContext] = None

        self.documents: dict[str, PyModule] = {}
        self.document_tokens: dict[str, list[CommonToken]] = {}
        self.diagnostics: dict[str, DocumentDiagnostics] = {}

    def initialize(self, params: lsp.InitializeParams):
        if params.root_uri:
            self.root_paths.append(Path(to_fs_path(params.root_uri)))
        if params.workspace_folders:
            self.root_paths.extend(
                Path(to_fs_path(folder.uri)) for folder in params.workspace_folders
            )

        logger.warning(f"Root paths: {self.root_paths}")
        logger.warning(f"Search context: {self.search_context}")

        self.analyzer = PythonAnalyzer(
            root_paths=self.root_paths,
            search_context=self.search_context,
        )
        self.analyzer.load_typeshed()

    def parse_document(self, doc: TextDocument):
        module = self.analyzer.create_module(Path(to_fs_path(doc.uri)))
        module.loader = functools.partial(
            PythonSource.parse_uri_source, doc.source, doc.uri
        )
        self.analyzer.load_module(module, reload=True)
        self.documents[doc.uri] = module

        self.document_tokens[doc.uri] = [
            token
            for token in module.source.stream.tokens
            if not is_synthetic_token(token)
        ]

        diagnostics: list[lsp.Diagnostic] = []
        for error in module.context.errors:
            if isinstance(error, SemanticError) and error.range:
                start_token, end_token = error.range
                error_range = lsp.Range(
                    start=token_start_position(start_token),
                    end=token_end_position(end_token),
                )
                message = error.message
            else:
                # Use a placeholder range if the error location is unknown.
                error_range = lsp.Range(
                    start=lsp.Position(0, 0),
                    end=lsp.Position(0, 0),
                )
                message = str(error)

            diagnostics.append(
                lsp.Diagnostic(
                    range=error_range,
                    message=message,
                    severity=lsp.DiagnosticSeverity.Error,
                )
            )

        self.diagnostics[doc.uri] = DocumentDiagnostics(doc.version, diagnostics)

    def close_document(self, uri: str):
        if uri in self.documents:
            self.analyzer.unload_module(self.documents[uri])
            del self.documents[uri]
            del self.document_tokens[uri]
            del self.diagnostics[uri]

            self.text_document_publish_diagnostics(
                lsp.PublishDiagnosticsParams(uri=uri, diagnostics=[])
            )

    def publish_diagnostics(self):
        for uri, (version, diagnostics) in self.diagnostics.items():
            self.text_document_publish_diagnostics(
                lsp.PublishDiagnosticsParams(
                    uri=uri,
                    version=version,
                    diagnostics=diagnostics,
                )
            )

    def get_semantic_tokens(self, uri: str) -> Optional[lsp.SemanticTokens]:
        if uri not in self.documents:
            return None

        module = self.documents[uri]
        data: list[int] = []

        prev_line, prev_column = 1, 0

        for token in self.document_tokens[uri]:
            token: CommonToken
            if is_blank_token(token):
                continue

            # Compute token kind and modifiers.
            kind = module.context.get_token_kind(token)
            modifiers = module.context.get_token_modifiers(token)

            if kind is TokenKind.NONE:
                continue

            # Convert to integer values.
            kind_value = TOKEN_KINDS.index(kind.value)
            modifiers_value = modifiers.value

            # Split token text by newlines.
            for i, text in enumerate(token.text.split("\n")):
                # Compute current line and column.
                if i == 0:
                    line, column = token.line, token.column
                else:
                    line, column = line + 1, 0

                # Compute relative line and offset.
                if line == prev_line:
                    offset = 0, column - prev_column
                else:
                    offset = line - prev_line, column

                # Append token data.
                data.extend((*offset, len(text), kind_value, modifiers_value))

                # Update previous line and column.
                prev_line, prev_column = line, column

        return lsp.SemanticTokens(data=data)

    def get_hover(self, uri: str, position: lsp.Position) -> Optional[lsp.Hover]:
        if uri not in self.documents:
            return None

        module = self.documents[uri]
        token = token_at_position(self.document_tokens[uri], position, strict=True)
        if token is None:
            return None

        type_ = module.context.get_token_entity_type(token)
        if type_ is None:
            return None

        if isinstance(type_, PyModuleType):
            result = f"(module) {type_.module.name}"

        elif isinstance(type_, PyClassType):
            result = f"(class) {type_.cls.name}"

        elif isinstance(type_, PyFunctionType):
            func_label, _ = format_signature(
                type_.get_parameters(), type_.get_return_type()
            )
            result = f"({type_.func.tag}) def {type_.func.name}{func_label}"

        elif isinstance(type_, PyTypeVarDef | PyGenericAlias):
            result = f"(type) {type_}"

        else:
            result = f"(variable) {token.text}: {type_}"

        return lsp.Hover(
            contents=lsp.MarkupContent(
                kind=lsp.MarkupKind.Markdown,
                value=f"```python\n{result}\n```",
            ),
            range=lsp.Range(
                start=token_start_position(token),
                end=token_end_position(token),
            ),
        )

    def get_definition(
        self, uri: str, position: lsp.Position
    ) -> Optional[lsp.Location]:
        if uri not in self.documents:
            return None

        module = self.documents[uri]
        token = token_at_position(self.document_tokens[uri], position, strict=True)
        if token is None:
            return None

        # Check if the symbol is a module.
        type_ = module.context.get_token_entity_type(token)
        if isinstance(type_, PyModuleType) and (path := type_.module.path):
            if target_uri := from_fs_path(str(path)):
                return lsp.Location(
                    uri=target_uri,
                    range=lsp.Range(start=lsp.Position(0, 0), end=lsp.Position(0, 0)),
                )
            else:
                logger.warning(f"Failed to resolve module path: {path}")

        # Check if the symbol refers to another symbol in some module.
        symbol = module.context.get_token_target(token)
        if symbol is None or symbol.token is None:
            return None

        input_stream = symbol.token.getInputStream()
        if isinstance(input_stream, FileStream):
            target_uri = from_fs_path(input_stream.fileName)
            if target_uri is None:
                logger.warning(f"Failed to resolve file path: {input_stream.fileName}")
                return None
        elif isinstance(input_stream, UriSourceStream):
            target_uri = input_stream.uri
        else:
            return None

        return lsp.Location(
            uri=target_uri,
            range=lsp.Range(
                start=token_start_position(symbol.token),
                end=token_end_position(symbol.token),
            ),
        )

    def get_completions(
        self, uri: str, position: lsp.Position
    ) -> Iterator[lsp.CompletionItem]:
        if uri not in self.documents:
            return

        module = self.documents[uri]

        token = token_at_position(
            self.document_tokens[uri], position, anchor="end", skip_ws=True
        )
        if token is None or token.type in NON_COMPLETION_TOKENS:
            return

        node = node_at_token_index(module.source.tree, token.tokenIndex)
        parent_node = node.parentCtx

        if (
            isinstance(
                parent_node,
                PythonParser.PrimaryContext
                | PythonParser.TargetWithStarAtomContext
                | PythonParser.SingleSubscriptAttributeTargetContext
                | PythonParser.DelTargetContext,
            )
            and (node is parent_node.DOT() or node is parent_node.NAME())
        ) or (
            isinstance(parent_node, PythonParser.InvalidPrimaryContext)
            and node is parent_node.DOT()
        ):
            # Attribute access.
            base_type = module.context.get_node_type(parent_node.primary())
            for symbol, _ in base_type.iter_attrs():
                yield self.get_symbol_completion(symbol)
            return

        if find_ancestor_node(node, PythonParser.ImportStmtContext):
            # TODO: Import statement.
            return

        yield from self.get_keyword_completions()

        # Yield symbols available in the current scope.
        scope = module.context.get_node_scope(node)
        for symbol in scope.iter_symbols(parents=True):
            yield self.get_symbol_completion(symbol)

        yield from self.get_argument_completions(module, token, node)

    @staticmethod
    def get_keyword_completions() -> Iterator[lsp.CompletionItem]:
        for keyword in PYTHON_KEYWORDS:
            yield lsp.CompletionItem(
                label=keyword,
                kind=lsp.CompletionItemKind.Keyword,
            )

    @staticmethod
    def get_symbol_completion(symbol: Symbol) -> lsp.CompletionItem:
        kind = lsp.CompletionItemKind.Variable
        if entity := symbol.resolve_entity():
            if isinstance(entity, PyModule):
                kind = lsp.CompletionItemKind.Module
            elif isinstance(entity, PyClass):
                kind = lsp.CompletionItemKind.Class
            elif isinstance(entity, PyFunction):
                if entity.is_method:
                    kind = lsp.CompletionItemKind.Method
                else:
                    kind = lsp.CompletionItemKind.Function

        return lsp.CompletionItem(
            label=symbol.name,
            kind=kind,
        )

    def get_argument_completions(
        self, module: PyModule, token: CommonToken, node: PythonParser.PrimaryContext
    ) -> Iterator[lsp.CompletionItem]:
        if func_call := find_function_call(token, node, module.context):
            call_node, (func_type, func_args) = func_call
        else:
            return

        # Check if a keyword argument is desired.
        separators: list[TerminalNode] = []
        if lparen := call_node.LPAREN():
            separators.append(lparen)
        if genexp_node := call_node.genexp():
            separators.append(genexp_node.LPAREN())
        if arguments_node := call_node.arguments():
            separators.extend(collect_argument_commas(arguments_node))

        separator_indices = [s.getSymbol().tokenIndex for s in separators]
        if token.tokenIndex < separator_indices[0]:
            # This should not happen, as the token must be inside the arguments.
            logger.warning("Token is not inside the arguments.")
            return

        i, state = token.tokenIndex, "name"
        while i not in separator_indices:
            match module.source.stream.get(i).type:
                case PythonParser.NAME:
                    if state == "name":
                        state = "separator"
                    else:
                        return
                case PythonParser.WS:
                    pass
                case _:
                    return

            i -= 1

        param_names: set[str] = set()

        # Find the keyword parameters that have not been filled yet.
        for overload in func_type.get_overloads():
            parameters = overload.get_parameters()
            result = match_arguments_to_parameters(func_args, parameters)

            for param in parameters:
                if (
                    not param.posonly
                    and param.star is None
                    and param.name not in result.values
                ):
                    param_names.add(param.name)

        for param_name in param_names:
            yield lsp.CompletionItem(
                label=f"{param_name}=",
                kind=lsp.CompletionItemKind.Variable,
            )

    def get_signature_help(
        self, uri: str, position: lsp.Position
    ) -> Optional[lsp.SignatureHelp]:
        if uri not in self.documents:
            return None

        module = self.documents[uri]
        token = token_at_position(
            self.document_tokens[uri], position, anchor="end", skip_ws=True
        )
        if token is None:
            return None

        node = node_at_token_index(module.source.tree, token.tokenIndex)

        # Find the primary node that contains a valid function call.
        if func_call := find_function_call(token, node, module.context):
            call_node, (func_type, func_args) = func_call
        else:
            return None

        overloads = func_type.get_overloads()

        # Find the index of the active argument.
        if arguments_node := call_node.arguments():
            arg_index = get_argument_index(arguments_node, token)
        else:
            arg_index = 0

        signatures = [
            get_signature_information(f, func_args, arg_index) for f in overloads
        ]
        return lsp.SignatureHelp(
            signatures=[s[0] for s in signatures],
            active_signature=max(
                range(len(signatures)), key=lambda i: signatures[i][1]
            ),
        )


def find_function_call(
    token: CommonToken, node: ParseTree, context: PythonContext
) -> Optional[tuple[PythonParser.PrimaryContext, PyFunctionCall]]:
    """
    Find the function call node that contains the given token.

    Args:
        token: The token to find the function call for.
        node: The node to start searching from.
        context: The context to use for finding the function call.

    Returns:
        A tuple containing the function call node and the function call, or None if not
        found.
    """
    while node is not None:
        call_node = find_ancestor_node(node, PythonParser.PrimaryContext)
        if call_node is None:
            return None

        func_call = context.get_function_call(call_node)
        if func_call is not None and is_inside_arguments(call_node, token):
            return call_node, func_call

        node = call_node.parentCtx

    return None


def get_signature_information(
    func_type: PyFunctionType, func_args: PyArguments, arg_index: int
) -> lsp.SignatureInformation:
    """
    Construct a signature information object for the given function type.

    Args:
        func_type: The function type to construct the signature for.
        func_args: The arguments passed to the function.
        arg_index: The index of the active argument.

    Returns:
        A signature information object for the given function type.
    """
    # Find the corresponding parameter.
    parameters = func_type.get_parameters()
    result = match_arguments_to_parameters(func_args, parameters)

    # Compute the matching score to rank the signature.
    score = (
        -10 * len(result.mismatched_args)
        - 5 * len(result.duplicate_args)
        - 2 * len(result.missing_params)
    )

    if (
        not result.mismatched_args
        and not result.duplicate_args
        and not result.missing_params
    ):
        # If all arguments match the parameters, add a bonus to the score.
        score += 5

    if arg_index < len(func_args):
        active_param_index = result.matched.get(arg_index, -1)
    else:
        active_param_index = result.next_positional
        if arg_index > 0 and result.is_complete:
            # If the user typed a comma after the last argument, but there are no
            # more parameters to fill, penalize the score.
            score -= 10

    # Construct the signature help.
    return_type = func_type.get_return_type()
    func_label, param_labels = format_signature(parameters, return_type)

    logger.info(f"Signature: {func_label}, score: {score}")

    signature = lsp.SignatureInformation(
        label=func_label,
        documentation=None,
        parameters=[lsp.ParameterInformation(label=label) for label in param_labels],
        active_parameter=active_param_index,
    )
    return signature, score


def format_signature(
    parameters: PyParameters, return_type: PyType
) -> tuple[str, list[str]]:
    """
    Format a function signature for display in the signature help.

    Args:
        parameters: The function parameters.
        return_type: The function return type.

    Returns:
        A tuple containing the formatted function label and parameter labels.
    """
    param_labels = [param.get_label() for param in parameters]

    with io.StringIO() as buf:
        slash, star = False, False

        buf.write("(")

        for i, param in enumerate(parameters):
            if i > 0:
                buf.write(", ")

            if not slash and not param.posonly:
                if i > 0:
                    buf.write("/, ")
                slash = True

            if param.star is not None:
                star = True
            elif not star and param.kwonly:
                buf.write("*, ")
                star = True

            buf.write(param_labels[i])

        if parameters and not slash:
            buf.write(", /")

        buf.write(f") -> {return_type}")
        func_label = buf.getvalue()

    return func_label, param_labels


server = PythonLanguageServer()


@server.feature(lsp.INITIALIZE)
def initialize(ls: PythonLanguageServer, params: lsp.InitializeParams):
    ls.initialize(params)


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: PythonLanguageServer, params: lsp.DidOpenTextDocumentParams):
    logger.info(f"Opened document: {params.text_document.uri}")
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.parse_document(doc)
    ls.publish_diagnostics()


@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: PythonLanguageServer, params: lsp.DidChangeTextDocumentParams):
    logger.info(f"Changed document: {params.text_document.uri}")
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.parse_document(doc)
    ls.publish_diagnostics()


@server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
def did_close(ls: PythonLanguageServer, params: lsp.DidCloseTextDocumentParams):
    logger.info(f"Closed document: {params.text_document.uri}")
    ls.close_document(params.text_document.uri)


@server.feature(
    lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
    lsp.SemanticTokensLegend(
        token_types=TOKEN_KINDS,
        token_modifiers=TOKEN_MODIFIERS,
    ),
)
def semantic_tokens_full(
    ls: PythonLanguageServer, params: lsp.SemanticTokensParams
) -> Optional[lsp.SemanticTokens]:
    logger.info(f"Requested semantic tokens: {params.text_document.uri}")
    with ls.analyzer.set_type_context():
        return ls.get_semantic_tokens(params.text_document.uri)


@server.feature(lsp.TEXT_DOCUMENT_HOVER)
def hover(ls: PythonLanguageServer, params: lsp.HoverParams) -> Optional[lsp.Hover]:
    logger.info(f"Requested hover: {params.text_document.uri} at {params.position}")
    with ls.analyzer.set_type_context():
        return ls.get_hover(params.text_document.uri, params.position)


@server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
def goto_definition(
    ls: PythonLanguageServer, params: lsp.DefinitionParams
) -> Optional[lsp.Location]:
    logger.info(
        f"Requested definition: {params.text_document.uri} at {params.position}"
    )
    return ls.get_definition(params.text_document.uri, params.position)


@server.feature(
    lsp.TEXT_DOCUMENT_COMPLETION,
    lsp.CompletionOptions(trigger_characters=["."]),
)
def completions(
    ls: PythonLanguageServer, params: lsp.CompletionParams
) -> lsp.CompletionList:
    logger.info(
        f"Requested completions: {params.text_document.uri} at {params.position}"
    )
    with ls.analyzer.set_type_context():
        items = list(ls.get_completions(params.text_document.uri, params.position))
    return lsp.CompletionList(is_incomplete=False, items=items)


@server.feature(
    lsp.TEXT_DOCUMENT_SIGNATURE_HELP,
    lsp.SignatureHelpOptions(trigger_characters=["(", ","]),
)
def signature_help(
    ls: PythonLanguageServer, params: lsp.SignatureHelpParams
) -> Optional[lsp.SignatureHelp]:
    logger.info(
        f"Requested signature help: {params.text_document.uri} at {params.position}"
    )
    with ls.analyzer.set_type_context():
        return ls.get_signature_help(params.text_document.uri, params.position)
