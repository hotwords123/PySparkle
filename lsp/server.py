import functools
import logging
from pathlib import Path

from antlr4.Token import CommonToken
from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer
from pygls.workspace import TextDocument
from typeshed_client import get_search_context

from core.analysis import PythonAnalyzer
from core.source import PythonSource
from grammar import PythonParser
from lsp.utils import snake_case_to_camel_case
from semantics.entity import PyModule
from semantics.token import TokenKind, TokenModifier

TOKEN_KINDS = [kind.value for kind in TokenKind]
TOKEN_MODIFIERS = [
    snake_case_to_camel_case(modifier.name) for modifier in TokenModifier
]


class PythonLanguageServer(LanguageServer):
    """
    A language server for Python.
    """

    def __init__(self):
        super().__init__("python-language-server", "v0.1.0")

        search_context = get_search_context()
        self.analyzer = PythonAnalyzer(search_paths=[search_context.typeshed])
        self.analyzer.load_typeshed()

        self.documents: dict[str, PyModule] = {}

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


server = PythonLanguageServer()


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: PythonLanguageServer, params: lsp.DidOpenTextDocumentParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.parse_document(doc)


@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: PythonLanguageServer, params: lsp.DidChangeTextDocumentParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.parse_document(doc)


@server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
def did_close(ls: PythonLanguageServer, params: lsp.DidCloseTextDocumentParams):
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
    return ls.get_semantic_tokens(params.text_document.uri)
