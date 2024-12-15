from typing import Optional

from antlr4 import ParserRuleContext
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
        self.range: Optional[tuple[CommonToken, CommonToken]] = (
            (token, end_token or token) if token else None
        )

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
            self.range = node.start, node.stop

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
