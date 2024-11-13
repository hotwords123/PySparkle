import sys
from antlr4 import *
from grammar import PythonLexer, PythonParser, PythonErrorStrategy


def main(argv):
    input_stream = FileStream(argv[1])
    lexer = PythonLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = PythonParser(stream)
    parser._errHandler = PythonErrorStrategy()
    tree = parser.file_()

    for token in stream.tokens:
        print(token)

    def indent(depth: int):
        return "".join(["| ", ": "][i % 2] for i in range(depth))

    def walk_tree(node, depth):
        if isinstance(node, TerminalNode):
            print(
                indent(depth)
                + PythonParser.symbolicNames[node.getSymbol().type]
                + ": "
                + repr(node.getText())
            )
        elif isinstance(node, RuleContext):
            print(indent(depth) + PythonParser.ruleNames[node.getRuleIndex()])
            for child in node.getChildren():
                walk_tree(child, depth + 1)

    walk_tree(tree, 0)


if __name__ == "__main__":
    main(sys.argv)
