from typing import NamedTuple

from antlr4 import CommonTokenStream, FileStream, ParserRuleContext

from grammar import PythonErrorStrategy, PythonLexer, PythonParser


class PythonSource(NamedTuple):
    input_stream: FileStream
    lexer: PythonLexer
    stream: CommonTokenStream
    parser: PythonParser
    tree: ParserRuleContext

    @classmethod
    def parse(cls, filename: str) -> "PythonSource":
        input_stream = FileStream(filename, encoding="utf-8")
        lexer = PythonLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = PythonParser(stream)
        parser._errHandler = PythonErrorStrategy()
        tree = parser.file_()
        return cls(input_stream, lexer, stream, parser, tree)
