from pathlib import Path
from typing import NamedTuple

from antlr4 import CommonTokenStream, FileStream, InputStream, ParserRuleContext

from grammar import PythonErrorStrategy, PythonLexer, PythonParser


class PythonSource(NamedTuple):
    input_stream: InputStream
    lexer: PythonLexer
    stream: CommonTokenStream
    parser: PythonParser
    tree: ParserRuleContext

    @classmethod
    def from_stream(cls, input_stream: InputStream) -> "PythonSource":
        lexer = PythonLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = PythonParser(stream)
        parser._errHandler = PythonErrorStrategy()
        tree = parser.file_()
        return cls(input_stream, lexer, stream, parser, tree)

    @classmethod
    def parse_file(cls, filename: str | Path) -> "PythonSource":
        if isinstance(filename, Path):
            filename = str(filename)
        input_stream = FileStream(filename, encoding="utf-8")
        return cls.from_stream(input_stream)

    @classmethod
    def parse_uri_source(cls, source: str, uri: str) -> "PythonSource":
        input_stream = UriSourceStream(source, uri)
        return cls.from_stream(input_stream)


class UriSourceStream(InputStream):
    def __init__(self, source: str, uri: str):
        super().__init__(source)
        self.uri = uri
