import bisect
from typing import Literal, Optional

from antlr4 import ParserRuleContext
from antlr4.Token import CommonToken
from antlr4.tree.Tree import ParseTree
from lsprotocol import types as lsp

from grammar import PythonParser
from semantics.base import get_token_end_position as _get_token_end_position


def snake_case_to_camel_case(snake_str: str) -> str:
    first, *rest = snake_str.split("_")
    return first.lower() + "".join(x.title() for x in rest)


def token_start_position(token: CommonToken) -> lsp.Position:
    return lsp.Position(token.line - 1, token.column)


def token_end_position(token: CommonToken) -> lsp.Position:
    line, column = _get_token_end_position(token)
    return lsp.Position(line - 1, column)


def token_contains_position(token: CommonToken, position: lsp.Position) -> bool:
    start = token_start_position(token)
    end = token_end_position(token)
    return start <= position <= end


def token_at_position(
    tokens: list[CommonToken],
    position: lsp.Position,
    anchor: Literal["start", "end"] = "start",
    strict: bool = False,
    skip_ws: bool = False,
) -> Optional[CommonToken]:
    """
    Find the token at the given position, using binary search.

    The tokens are expected to be sorted by start position.

    Args:
        tokens: A list of tokens to search through.
        position: The position to look for.
        anchor: Whether to consider the start or end position of the token.
        strict: Whether to require the token to contain the position.
        skip_ws: Whether to skip whitespace tokens.

    Returns:
        The token at the given position, or None if no token is found.
    """
    match anchor:
        case "start":
            index = bisect.bisect_right(tokens, position, key=token_start_position) - 1
        case "end":
            index = bisect.bisect_left(tokens, position, key=token_end_position)
        case _:
            raise ValueError(f"Invalid anchor: {anchor!r}")

    try:
        token = tokens[index]
    except IndexError:
        return None

    if skip_ws:
        while token.type == PythonParser.WS:
            index += 1 if anchor == "start" else -1
            try:
                token = tokens[index]
            except IndexError:
                return None

    elif strict and not token_contains_position(token, position):
        return None

    return token


def node_at_token_index(tree: ParseTree, token_index: int) -> ParseTree:
    """
    Find the innermost node that contains the token at the given index.

    Args:
        tree: The tree to search through.
        index: The index of the token to look for.

    Returns:
        The innermost node that contains the token at the given index.
    """
    while isinstance(tree, ParserRuleContext):
        for child in tree.getChildren():
            start_index, stop_index = child.getSourceInterval()
            if start_index <= token_index <= stop_index:
                tree = child
                break
        else:
            break

    return tree
