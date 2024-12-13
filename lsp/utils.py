import bisect
from typing import Optional

from antlr4.Token import CommonToken
from lsprotocol import types as lsp

from semantics.base import get_token_end_position as _get_token_end_position


def snake_case_to_camel_case(snake_str: str) -> str:
    first, *rest = snake_str.split("_")
    return first.lower() + "".join(x.title() for x in rest)


def token_start_position(token: CommonToken) -> lsp.Position:
    return lsp.Position(token.line - 1, token.column)


def token_end_position(token: CommonToken) -> lsp.Position:
    line, column = _get_token_end_position(token)
    return lsp.Position(line - 1, column)


def token_at_position(
    tokens: list[CommonToken], position: lsp.Position
) -> Optional[CommonToken]:
    index = bisect.bisect_right(tokens, position, key=token_start_position) - 1

    try:
        token = tokens[index]
    except IndexError:
        return None

    if token_start_position(token) <= position < token_end_position(token):
        return token

    return None
