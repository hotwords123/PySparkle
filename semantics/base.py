from typing import Optional

from antlr4 import CommonTokenStream, ParserRuleContext
from antlr4.Token import CommonToken
from antlr4.tree.Tree import ParseTree, TerminalNode

from semantics.token import is_synthetic_token


class SemanticError(Exception):
    def __init__(
        self,
        message: str,
        token: Optional[CommonToken] = None,
        end_token: Optional[CommonToken] = None,
    ):
        self.message = message
        if token:
            if end_token:
                self.range = shrink_token_range(token, end_token)
            else:
                self.range = token, token
        else:
            self.range = None

    def __str__(self):
        message = self.message
        if self.range is not None:
            start, end = self.range
            end_line, end_column = get_token_end_position(end)
            message += f" at {start.line}:{start.column} to {end_line}:{end_column}"
        return message

    def set_context(self, node: ParseTree):
        if isinstance(node, TerminalNode):
            self.range = node.getSymbol(), node.getSymbol()
        elif isinstance(node, ParserRuleContext):
            self.range = shrink_token_range(node.start, node.stop)

    def with_context(self, node: ParseTree):
        self.set_context(node)
        return self


class PySyntaxError(SemanticError):
    pass


def get_token_end_position(token: CommonToken) -> tuple[int, int]:
    """
    Returns the end position of a token in the source code.

    Args:
        token: The token to get the end position of.

    Returns:
        A tuple of the line and column of the end position.
    """
    if is_synthetic_token(token):
        # Synthetic tokens do not correspond to any text in the source code.
        return token.line, token.column

    lines = token.text.split("\n")
    if len(lines) == 1:
        return token.line, token.column + len(token.text)
    else:
        return token.line + len(lines) - 1, len(lines[-1])


def get_node_source(node: ParseTree) -> str:
    """
    Returns the source code of a node in the parse tree.

    Args:
        node: The node to get the source code of.

    Returns:
        The source code of the node.
    """
    if isinstance(node, ParserRuleContext):
        start, stop = node.start, node.stop
        return start.getInputStream().getText(start.start, stop.stop)

    return node.getText()


def shrink_token_range(
    start: CommonToken, end: CommonToken
) -> tuple[CommonToken, CommonToken]:
    """
    Shrinks a token range to exclude any hidden tokens.

    Args:
        start: The start token of the range.
        end: The end token of the range.

    Returns:
        A tuple of the start and end tokens of the reduced range.
    """
    if not isinstance(stream := start.getTokenSource(), CommonTokenStream):
        return start, end

    start_index, end_index = start.tokenIndex, end.tokenIndex
    while (
        start_index < end_index
        and stream.get(start_index).channel == CommonToken.HIDDEN_CHANNEL
    ):
        start_index += 1
    while (
        end_index > start_index
        and stream.get(end_index).channel == CommonToken.HIDDEN_CHANNEL
    ):
        end_index -= 1

    return stream.get(start_index), stream.get(end_index)
