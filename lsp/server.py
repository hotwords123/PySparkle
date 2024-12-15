import functools
import logging
from pathlib import Path
from typing import Optional

from antlr4 import FileStream
from antlr4.Token import CommonToken
from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer
from pygls.uris import from_fs_path
from pygls.workspace import TextDocument
from typeshed_client import get_search_context

from core.analysis import PythonAnalyzer
from core.source import PythonSource, UriSourceStream
from grammar import PythonParser
from lsp.utils import (
    node_at_token_index,
    snake_case_to_camel_case,
    token_at_position,
    token_end_position,
    token_start_position,
)
from semantics.entity import PyClass, PyFunction, PyModule, PyParameter, PyVariable
from semantics.token import TokenKind, TokenModifier
from semantics.types import (
    PyClassType,
    PyFunctionType,
    PyGenericAlias,
    PyModuleType,
    PyTypeVarDef,
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


logger = logging.getLogger(__name__)


class PythonLanguageServer(LanguageServer):
    """
    A language server for Python.
    """

    def __init__(self):
        super().__init__("python-language-server", "v0.1.0")

        search_context = get_search_context()
        self.analyzer = PythonAnalyzer(search_paths=[search_context.typeshed])
        self.documents: dict[str, PyModule] = {}

    def init(self):
        self.analyzer.load_typeshed()

    def parse_document(self, doc: TextDocument):
        path = Path(doc.path)
        module_name = path.stem
        if "." in module_name:
            raise ValueError("source file name should not contain '.'")

        module = PyModule(module_name, path)
        module.loader = functools.partial(
            PythonSource.parse_uri_source, doc.source, doc.uri
        )
        self.analyzer.load_module(module, reload=True)
        self.documents[doc.uri] = module

    def close_document(self, uri: str):
        if uri in self.documents:
            self.analyzer.unload_module(self.documents[uri])
            del self.documents[uri]

    def get_semantic_tokens(self, uri: str) -> Optional[lsp.SemanticTokens]:
        if uri not in self.documents:
            return None

        module = self.documents[uri]
        data: list[int] = []

        prev_line, prev_column = 1, 0

        for token in module.source.stream.tokens:
            token: CommonToken
            if token.type in (
                PythonParser.INDENT,
                PythonParser.DEDENT,
                PythonParser.EOF,
            ):
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
        token = token_at_position(module.source.stream.tokens, position)
        if token is None:
            return None

        type_ = module.context.get_token_entity_type(token)
        if type_ is None:
            return None

        if isinstance(type_, PyModuleType):
            result = f"(module) {type_.module.name}"

        elif isinstance(type_, PyClassType):
            result = f"(class) {type_.cls.full_name}"

        elif isinstance(type_, PyFunctionType):
            result = f"({type_.func.tag}) {type_.func.full_name}"

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
        token = token_at_position(module.source.stream.tokens, position)
        if token is None:
            return None

        symbol = module.context.get_token_target(token)
        if symbol is None or symbol.token is None:
            return None

        input_stream = symbol.token.getInputStream()
        if isinstance(input_stream, FileStream):
            target_uri = from_fs_path(input_stream.fileName)
            if target_uri is None:
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
    ) -> Optional[lsp.CompletionList]:
        if uri not in self.documents:
            return None

        module = self.documents[uri]
        token = token_at_position(module.source.stream.tokens, position)
        if token is None:
            return None

        node = node_at_token_index(module.source.tree, token.tokenIndex)
        scope = module.context.get_node_scope(node)

        items: list[lsp.CompletionItem] = []

        for keyword in PYTHON_KEYWORDS:
            items.append(
                lsp.CompletionItem(
                    label=keyword,
                    kind=lsp.CompletionItemKind.Keyword,
                )
            )

        for symbol in scope.iter_symbols(parents=True):
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

            items.append(
                lsp.CompletionItem(
                    label=symbol.name,
                    kind=kind,
                    detail=symbol.full_name,
                )
            )

        return lsp.CompletionList(is_incomplete=False, items=items)


server = PythonLanguageServer()


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: PythonLanguageServer, params: lsp.DidOpenTextDocumentParams):
    logger.info(f"Opened document: {params.text_document.uri}")
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.parse_document(doc)


@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: PythonLanguageServer, params: lsp.DidChangeTextDocumentParams):
    logger.info(f"Changed document: {params.text_document.uri}")
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.parse_document(doc)


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
    return ls.get_semantic_tokens(params.text_document.uri)


@server.feature(lsp.TEXT_DOCUMENT_HOVER)
def hover(ls: PythonLanguageServer, params: lsp.HoverParams) -> Optional[lsp.Hover]:
    logger.info(f"Requested hover: {params.text_document.uri}")
    return ls.get_hover(params.text_document.uri, params.position)


@server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
def goto_definition(
    ls: PythonLanguageServer, params: lsp.DefinitionParams
) -> Optional[lsp.Location]:
    logger.info(f"Requested definition: {params.text_document.uri}")
    return ls.get_definition(params.text_document.uri, params.position)


@server.feature(
    lsp.TEXT_DOCUMENT_COMPLETION,
    lsp.CompletionOptions(trigger_characters=["."]),
)
def completions(
    ls: PythonLanguageServer, params: lsp.CompletionParams
) -> Optional[lsp.CompletionList]:
    logger.info(f"Requested completions: {params.text_document.uri}")
    return ls.get_completions(params.text_document.uri, params.position)
