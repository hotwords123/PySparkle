from enum import Enum, IntFlag, auto
from typing import TYPE_CHECKING, TypedDict

from grammar import PythonParser

if TYPE_CHECKING:
    from antlr4.Token import CommonToken

    from .symbol import Symbol
    from .types import PyType


class TokenInfo(TypedDict, total=False):
    kind: "TokenKind"
    symbol: "Symbol"
    type: "PyType"
    modifiers: "TokenModifier"


class TokenKind(Enum):
    NONE = "none"
    COMMENT = "comment"

    KEYWORD = "keyword"
    PRIMITIVE = "primitive"

    IDENTIFIER = "identifier"
    VARIABLE = "variable"
    FIELD = "field"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    CONSTANT = "constant"

    STRING = "string"
    NUMBER = "number"

    OPERATOR = "operator"
    DELIMITER = "delimiter"

    ERROR = "error"


class TokenModifier(IntFlag):
    READONLY = auto()
    ASYNC = auto()


TOKEN_KIND_SPEC = {
    TokenKind.COMMENT: {
        PythonParser.COMMENT,
    },
    TokenKind.PRIMITIVE: {
        PythonParser.IN,
        PythonParser.CLASS,
        PythonParser.IS,
        PythonParser.AND,
        PythonParser.LAMBDA,
        PythonParser.DEF,
        PythonParser.NONLOCAL,
        PythonParser.GLOBAL,
        PythonParser.NOT,
        PythonParser.ASYNC,
        PythonParser.OR,
    },
    TokenKind.KEYWORD: {
        PythonParser.AWAIT,
        PythonParser.ELSE,
        PythonParser.IMPORT,
        PythonParser.PASS,
        PythonParser.BREAK,
        PythonParser.EXCEPT,
        PythonParser.RAISE,
        PythonParser.FINALLY,
        PythonParser.RETURN,
        PythonParser.CONTINUE,
        PythonParser.FOR,
        PythonParser.TRY,
        PythonParser.AS,
        PythonParser.FROM,
        PythonParser.WHILE,
        PythonParser.ASSERT,
        PythonParser.DEL,
        PythonParser.WITH,
        PythonParser.ELIF,
        PythonParser.IF,
        PythonParser.YIELD,
    },
    TokenKind.IDENTIFIER: {
        PythonParser.NAME,
    },
    TokenKind.CONSTANT: {
        PythonParser.FALSE,
        PythonParser.NONE,
        PythonParser.TRUE,
    },
    TokenKind.STRING: {
        PythonParser.STRING_LITERAL,
        PythonParser.BYTES_LITERAL,
    },
    TokenKind.NUMBER: {
        PythonParser.INTEGER,
        PythonParser.FLOAT_NUMBER,
        PythonParser.IMAG_NUMBER,
    },
    TokenKind.OPERATOR: {
        PythonParser.PLUS,
        PythonParser.MINUS,
        PythonParser.STAR,
        PythonParser.DOUBLESTAR,
        PythonParser.SLASH,
        PythonParser.DOUBLESLASH,
        PythonParser.PERCENT,
        PythonParser.AT,
        PythonParser.LEFTSHIFT,
        PythonParser.RIGHTSHIFT,
        PythonParser.AMPERSAND,
        PythonParser.VBAR,
        PythonParser.CIRCUMFLEX,
        PythonParser.TILDE,
        PythonParser.COLONEQUAL,
        PythonParser.LESS,
        PythonParser.GREATER,
        PythonParser.LESSEQUAL,
        PythonParser.GREATEREQUAL,
        PythonParser.DOUBLEEQUAL,
        PythonParser.NOTEQUAL,
    },
    TokenKind.DELIMITER: {
        PythonParser.LPAREN,
        PythonParser.RPAREN,
        PythonParser.LSQB,
        PythonParser.RSQB,
        PythonParser.LBRACE,
        PythonParser.RBRACE,
        PythonParser.COMMA,
        PythonParser.COLON,
        PythonParser.EXCLAMATION,
        PythonParser.DOT,
        PythonParser.SEMICOLON,
        PythonParser.EQUAL,
        PythonParser.RARROW,
        PythonParser.PLUSEQUAL,
        PythonParser.MINUSEQUAL,
        PythonParser.STAREQUAL,
        PythonParser.SLASHEQUAL,
        PythonParser.DOUBLESLASHEQUAL,
        PythonParser.PERCENTEQUAL,
        PythonParser.ATEQUAL,
        PythonParser.AMPERSANDEQUAL,
        PythonParser.VBAREQUAL,
        PythonParser.CIRCUMFLEXEQUAL,
        PythonParser.RIGHTSHIFTEQUAL,
        PythonParser.LEFTSHIFTEQUAL,
        PythonParser.DOUBLESTAREQUAL,
        PythonParser.ELLIPSIS,
    },
    TokenKind.ERROR: {
        PythonParser.ERROR,
    },
}

TOKEN_KIND_MAP = {
    token_type: kind
    for kind, token_types in TOKEN_KIND_SPEC.items()
    for token_type in token_types
}

SYNTHETIC_TOKEN_TYPES = {
    PythonParser.INDENT,
    PythonParser.DEDENT,
    PythonParser.EOF,
}

BLANK_TOKEN_TYPES = {
    *SYNTHETIC_TOKEN_TYPES,
    PythonParser.NEWLINE,
    PythonParser.WS,
}


def is_synthetic_token(token: "CommonToken") -> bool:
    """
    Returns whether a token is synthetic, i.e. not part of the source code.
    """
    return (
        token.type in SYNTHETIC_TOKEN_TYPES
        or (token.type == PythonParser.NEWLINE and token.text == "<NEWLINE>")
        or token.tokenIndex == -1
    )


def is_blank_token(token: "CommonToken") -> bool:
    """
    Returns whether a token is blank, i.e. contains no text or whitespace only.
    """
    return token.type in BLANK_TOKEN_TYPES
