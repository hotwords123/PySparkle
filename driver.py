import sys

import yaml
from antlr4 import *
from antlr4.Token import CommonToken

from grammar import PythonErrorStrategy, PythonLexer, PythonParser, PythonParserVisitor


class PythonVisitor(PythonParserVisitor):
    def visitChildren(self, node: RuleContext) -> dict:
        children = [self.visit(child) for child in node.getChildren()]
        return {get_rule_name(node): children}

    def visitTerminal(self, node: TerminalNode) -> dict:
        return {get_token_name(node.getSymbol()): node.getText()}

    def visitErrorNode(self, node: ErrorNode) -> dict:
        return {"error_" + get_token_name(node.getSymbol()): node.getText()}


def get_rule_name(rule: RuleContext) -> str:
    return PythonParser.ruleNames[rule.getRuleIndex()]


def get_token_name(token: CommonToken) -> str:
    if token.type == Token.EOF:
        return "EOF"
    return PythonParser.symbolicNames[token.type]


def main(argv):
    args = parse_args(argv[1:])

    out_file = open(args.output, "w", newline="") if args.output else sys.stdout

    input_stream = FileStream(args.input)
    lexer = PythonLexer(input_stream)
    stream = CommonTokenStream(lexer)

    if args.mode == "lex":
        stream.fill()

        for token in stream.tokens:
            token: CommonToken
            token_type = get_token_name(token)
            token_text = repr(token.text)
            print(
                f"{token.line}:{token.column}: {token_type} {token_text}",
                file=out_file,
                end="",
            )
            if token.channel != Token.DEFAULT_CHANNEL:
                print(f" (channel={token.channel})", file=out_file, end="")
            print(file=out_file)

    elif args.mode == "parse":
        parser = PythonParser(stream)
        parser._errHandler = PythonErrorStrategy()
        tree = parser.file_()

        visitor = PythonVisitor()
        result = visitor.visit(tree)
        yaml.dump(result, out_file, indent=2, sort_keys=False)

    if args.output:
        out_file.close()


def parse_args(args: list[str] = None):
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["lex", "parse"], help="mode of operation")
    parser.add_argument("input", help="input file")
    parser.add_argument("-o", "--output", help="output file")
    # fmt: on

    return parser.parse_args(args)


if __name__ == "__main__":
    main(sys.argv)
