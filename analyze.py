import sys
from pathlib import Path

import dominate
import dominate.tags as dom
from antlr4 import *
from antlr4.Token import CommonToken
from dominate.util import text as dom_text
from typeshed_client import get_search_context

from core.analysis import PythonAnalyzer
from grammar import PythonParser
from semantics.entity import PyModule
from semantics.structure import PythonContext
from semantics.symbol import SymbolType
from semantics.token import TOKEN_KIND_MAP, TokenKind


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
                    return TokenKind.FUNCTION
                case SymbolType.CLASS:
                    return TokenKind.CLASS
                case SymbolType.GLOBAL | SymbolType.NONLOCAL | SymbolType.IMPORTED:
                    # Unresolved symbols.
                    pass

            if entity := symbol.resolve_entity():
                if isinstance(entity, PyModule):
                    return TokenKind.MODULE

                return TokenKind.VARIABLE

    return TOKEN_KIND_MAP.get(token.type, TokenKind.NONE)


def main(args):
    source_path = Path(args.input)
    out_file = open(args.output, "w", newline="") if args.output else sys.stdout

    assert "." not in source_path.stem, "source file name should not contain '.'"

    # https://typing.readthedocs.io/en/latest/spec/distributing.html#import-resolution-ordering
    search_context = get_search_context()

    search_paths = []
    search_paths.append(source_path.parent)
    search_paths.append(search_context.typeshed)
    search_paths.extend(search_context.search_path)

    analyzer = PythonAnalyzer(search_paths)

    if args.builtins:
        analyzer.load_builtins()
    else:
        analyzer.builtins_loaded = True

    module = PyModule(source_path.stem, source_path)
    analyzer.importer.load_module(module)

    for error in module.context.errors:
        print(error, file=sys.stderr)

    doc = dominate.document(title="Python Code")

    with doc.head:
        dom.link(rel="stylesheet", href="style.css")

    with doc:
        with dom.div(cls="highlight"):
            with dom.pre():
                for token in module.source.stream.tokens:
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
                        dom.span(token.text, cls=f"token-{token_kind.value}")

    print(doc, file=out_file)

    if args.output:
        out_file.close()


def parse_args(args: list[str] = None):
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file")
    parser.add_argument("-o", "--output", help="output file")
    parser.add_argument("-b", "--builtins", help="load builtins", action="store_true")
    # fmt: on

    return parser.parse_args(args)


if __name__ == "__main__":
    main(parse_args())
