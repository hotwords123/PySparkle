import sys
from pathlib import Path
from typing import Optional

import dominate
import dominate.tags as dom
from antlr4 import *
from antlr4.Token import CommonToken
from dominate.util import text as dom_text
from typeshed_client import get_search_context

from core.analysis import PythonAnalyzer
from grammar import PythonParser
from semantics.entity import PyFunction, PyModule
from semantics.structure import PythonContext
from semantics.symbol import Symbol, SymbolType
from semantics.token import TOKEN_KIND_MAP, TokenKind
from semantics.types import PyType


def get_token_kind(context: PythonContext, token: CommonToken) -> TokenKind:
    if token_info := context.token_info.get(token):
        if token_kind := token_info.get("kind"):
            return token_kind

        if symbol := token_info.get("symbol"):
            symbol = symbol.resolve()

            match symbol.type:
                case SymbolType.VARIABLE | SymbolType.PARAMETER:
                    return TokenKind.VARIABLE
                case SymbolType.FUNCTION:
                    pass  # May be a @property method.
                case SymbolType.CLASS:
                    return TokenKind.CLASS
                case SymbolType.GLOBAL | SymbolType.NONLOCAL | SymbolType.IMPORTED:
                    pass  # Unresolved symbols.

            if entity := symbol.resolve_entity():
                if isinstance(entity, PyModule):
                    return TokenKind.MODULE

                if isinstance(entity, PyFunction):
                    if entity.has_modifier("property"):
                        return TokenKind.VARIABLE
                    return TokenKind.FUNCTION

                return TokenKind.VARIABLE

    return TOKEN_KIND_MAP.get(token.type, TokenKind.NONE)


def get_token_target(context: PythonContext, token: CommonToken) -> Optional[Symbol]:
    if token_info := context.token_info.get(token):
        if symbol := token_info.get("symbol"):
            return symbol.resolve()

    return None


def get_token_entity_type(
    context: PythonContext, token: CommonToken
) -> Optional[PyType]:
    if token_info := context.token_info.get(token):
        if type_ := token_info.get("type"):
            return type_

        if symbol := token_info.get("symbol"):
            return symbol.get_type()

    return None


def main(args):
    source_path = Path(args.input)

    assert "." not in source_path.stem, "source file name should not contain '.'"

    search_context = get_search_context()

    analyzer = PythonAnalyzer(search_paths=[search_context.typeshed])

    if args.typeshed:
        analyzer.load_typeshed()
    else:
        analyzer.typeshed_loaded = True

    # https://typing.readthedocs.io/en/latest/spec/distributing.html#import-resolution-ordering
    search_paths = [
        source_path.parent,
        search_context.typeshed,
        *search_context.search_path,
    ]
    analyzer.importer.search_paths = search_paths

    module = PyModule(source_path.stem, source_path)
    analyzer.importer.load_module(module)

    with analyzer.set_type_context():
        doc = generate_html(module)

    if args.output:
        with open(args.output, "w", newline="") as f:
            print(doc, file=f)
    else:
        print(doc, file=sys.stdout)


def generate_html(module: PyModule) -> dominate.document:
    token_source = (module.source.lexer, module.source.input_stream)

    doc = dominate.document(title="Python Code")

    with doc.head:
        dom.link(rel="stylesheet", href="style.css")
        dom.script(src="script.js")

    with doc:
        with dom.div(cls="highlight"):
            with dom.pre():
                for token in module.source.stream.tokens:
                    token: CommonToken
                    if token.type in {
                        PythonParser.INDENT,
                        PythonParser.DEDENT,
                        PythonParser.EOF,
                    }:
                        continue

                    token_kind = get_token_kind(module.context, token)
                    if token_kind is TokenKind.NONE:
                        dom_text(token.text)
                    else:
                        attrs = dict(
                            id=f"token-{token.tokenIndex}",
                            cls=f"token token-{token_kind.value}",
                        )

                        if entity_type := get_token_entity_type(module.context, token):
                            attrs["title"] = str(entity_type)

                        if (
                            (target := get_token_target(module.context, token))
                            and (target_token := target.token)
                            and target_token.source == token_source
                        ):
                            attrs["data-target"] = f"token-{target_token.tokenIndex}"

                        dom.span(token.text, **attrs)

    return doc


def parse_args(args: Optional[list[str]] = None):
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file")
    parser.add_argument("-o", "--output", help="output file")
    parser.add_argument("--no-typeshed", dest="typeshed", action="store_false", help="do not load typeshed")
    # fmt: on

    return parser.parse_args(args)


if __name__ == "__main__":
    main(parse_args())
