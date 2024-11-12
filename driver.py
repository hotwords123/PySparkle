import sys
from antlr4 import *
from grammar.PythonLexer import PythonLexer
from grammar.PythonParser import PythonParser


def main(argv):
    input_stream = FileStream(argv[1])
    lexer = PythonLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = PythonParser(stream)
    tree = parser.file_()

    for token in stream.tokens:
        print(token)

    print(tree.toStringTree(recog=parser))


if __name__ == "__main__":
    main(sys.argv)
