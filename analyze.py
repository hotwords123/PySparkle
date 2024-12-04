import sys
from typing import NamedTuple

import dominate
import dominate.tags as dom
from antlr4 import *
from antlr4.Token import CommonToken
from dominate.util import text as dom_text
from typeshed_client import get_stub_file

from grammar import PythonErrorStrategy, PythonLexer, PythonParser
from semantics.scope import ScopeType, SymbolTable
from semantics.structure import PythonContext
from semantics.token import TOKEN_KIND_MAP, TokenKind
from semantics.visitor import PythonVisitor


def get_rule_name(rule: RuleContext) -> str:
    return PythonParser.ruleNames[rule.getRuleIndex()]


def get_token_name(token: CommonToken) -> str:
    if token.type == Token.EOF:
        return "EOF"
    return PythonParser.symbolicNames[token.type]


class PythonSource(NamedTuple):
    input_stream: FileStream
    lexer: PythonLexer
    stream: CommonTokenStream
    parser: PythonParser
    tree: ParserRuleContext

    @classmethod
    def parse(cls, filename: str) -> "PythonSource":
        input_stream = FileStream(filename)
        lexer = PythonLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = PythonParser(stream)
        parser._errHandler = PythonErrorStrategy()
        tree = parser.file_()
        return cls(input_stream, lexer, stream, parser, tree)


def load_builtins() -> SymbolTable:
    stub_file = get_stub_file("builtins")
    source = PythonSource.parse(stub_file)

    global_scope = SymbolTable("<global>", ScopeType.GLOBAL)
    context = PythonContext(global_scope)

    visitor = PythonVisitor(context)
    visitor.fullVisit(source.tree)

    builtin_scope = SymbolTable("<builtins>", ScopeType.BUILTINS)
    for symbol in global_scope.iter_symbols(skip_imports=True, public_only=True):
        builtin_scope.define(symbol)

    return builtin_scope


def main(args):
    out_file = open(args.output, "w", newline="") if args.output else sys.stdout

    if args.builtins:
        print("Loading builtins...", file=sys.stderr)
        builtins_scope = load_builtins()
    else:
        builtins_scope = None

    print("Parsing input...", file=sys.stderr)
    source = PythonSource.parse(args.input)

    global_scope = SymbolTable("<global>", ScopeType.GLOBAL, builtins_scope)
    context = PythonContext(global_scope)

    visitor = PythonVisitor(context)
    visitor.fullVisit(source.tree)

    doc = dominate.document(title="Python Code")

    with doc.head:
        dom.link(rel="stylesheet", href="style.css")

    with doc:
        with dom.div(cls="highlight"):
            with dom.pre():
                for token in source.stream.tokens:
                    if token.type in {
                        PythonParser.INDENT,
                        PythonParser.DEDENT,
                        PythonParser.EOF,
                    }:
                        continue

                    token_kind = None
                    if token_info := context.token_info.get(token):
                        token_kind = token_info.get("kind")
                    if token_kind is None:
                        token_kind = TOKEN_KIND_MAP.get(token.type, TokenKind.NONE)

                    if token_kind == TokenKind.NONE:
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
