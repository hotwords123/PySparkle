import bisect
from typing import Literal, Optional

from antlr4 import ParserRuleContext
from antlr4.Token import CommonToken
from antlr4.tree.Tree import ParseTree, TerminalNode
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


def find_ancestor_node[
    T: ParserRuleContext
](tree: ParseTree, parent_type: type[T]) -> Optional[T]:
    """
    Find the nearest ancestor of the given type.

    Args:
        tree: The tree to search through.
        parent_type: The type of the ancestor to look for.

    Returns:
        The nearest ancestor of the given type, or None if not found.
    """
    while tree is not None:
        if isinstance(tree, parent_type):
            return tree
        tree = tree.parentCtx

    return None


def is_inside_arguments(
    call_node: PythonParser.PrimaryContext, token: CommonToken
) -> bool:
    """
    Return whether the given token is inside the arguments of a call expression.

    Args:
        call_node: The call expression node.
        token: The token to check.

    Returns:
        Whether the token is inside the arguments of the call expression.
    """
    if genexp := call_node.genexp():
        # primary: primary genexp
        # genexp: '(' namedExpression forIfClauses ')';
        lparen, rparen = genexp.LPAREN(), genexp.RPAREN()
    else:
        # primary: primary '(' arguments? ')'
        lparen, rparen = call_node.LPAREN(), call_node.RPAREN()

    if lparen is None or rparen is None:
        return False

    return (
        lparen.getSymbol().tokenIndex
        <= token.tokenIndex
        < rparen.getSymbol().tokenIndex
    )


def collect_argument_commas(
    arguments_node: PythonParser.ArgumentsContext,
) -> list[TerminalNode]:
    """
    Collect the comma tokens between arguments in the given arguments node.

    Args:
        arguments_node: The arguments node to search through.

    Returns:
        A list of comma tokens between arguments.
    """
    commas: list[TerminalNode] = []

    # arguments: args ','? | kwargs;
    if args_node := arguments_node.args():
        commas.extend(args_node.COMMA())
        # args: arg (',' arg)* (',' kwargs)? | kwargs;
        if kwargs_node := args_node.kwargs():
            # kwargs
            #   : kwargOrStarred (',' kwargOrStarred)* (',' kwargOrDoubleStarred)?
            #   | kwargOrDoubleStarred (',' kwargOrDoubleStarred)*;
            commas.extend(kwargs_node.COMMA())
    if comma := arguments_node.COMMA():
        commas.append(comma)

    return commas


def get_argument_index(
    arguments_node: PythonParser.ArgumentsContext, token: CommonToken
) -> int:
    """
    Find the index of the active argument in the given arguments node.

    Args:
        arguments_node: The arguments node to search through.
        token: The token to find the argument index of.

    Returns:
        The index of the active argument.
    """
    # Collect the comma tokens between arguments.
    commas = collect_argument_commas(arguments_node)

    # Collect the token indices of the commas.
    comma_indices = [comma.getSymbol().tokenIndex for comma in commas]

    # Find the index of the active argument.
    return bisect.bisect_right(comma_indices, token.tokenIndex)
