import sys

import dominate
import dominate.tags as dom
from antlr4 import *
from antlr4.Token import CommonToken
from dominate.util import text as dom_text

from grammar import PythonErrorStrategy, PythonLexer, PythonParser
from semantics.highlight import TOKEN_KIND_MAP, TokenKind
from semantics.visitor import PythonVisitor


def get_rule_name(rule: RuleContext) -> str:
    return PythonParser.ruleNames[rule.getRuleIndex()]


def get_token_name(token: CommonToken) -> str:
    if token.type == Token.EOF:
        return "EOF"
    return PythonParser.symbolicNames[token.type]


def main(args):
    out_file = open(args.output, "w", newline="") if args.output else sys.stdout

    input_stream = FileStream(args.input)
    lexer = PythonLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = PythonParser(stream)
    parser._errHandler = PythonErrorStrategy()
    tree = parser.file_()

    visitor = PythonVisitor()
    visitor.visit(tree)

    doc = dominate.document(title="Python Code")

    with doc.head:
        dom.link(rel="stylesheet", href="style.css")

    with doc:
        with dom.div(cls="highlight"):
            with dom.pre():
                for token in stream.tokens:
                    if token.type in {
                        PythonParser.INDENT,
                        PythonParser.DEDENT,
                        PythonParser.EOF,
                    }:
                        continue

                    if token in visitor.token_kinds:
                        token_kind = visitor.token_kinds[token]
                    else:
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
    # fmt: on

    return parser.parse_args(args)


if __name__ == "__main__":
    main(parse_args())
