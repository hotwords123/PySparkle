from typing import Optional

from antlr4 import ParserRuleContext
from antlr4.Token import CommonToken
from antlr4.tree.Tree import ParseTree, TerminalNode


class SemanticError(Exception):
    def __init__(
        self,
        message: str,
        token: Optional[CommonToken] = None,
        end_token: Optional[CommonToken] = None,
    ):
        self.message = message
        self.token = token
        self.end_token = end_token

    def __str__(self):
        message = self.message
        if self.token is not None:
            message += f" at {self.token.line}:{self.token.column}"
            if self.end_token is not None:
                message += f" to {self.end_token.line}:{self.end_token.column}"
        return message

    def set_context(self, node: ParseTree):
        if isinstance(node, TerminalNode):
            self.token = node.getSymbol()
            self.end_token = None
        elif isinstance(node, ParserRuleContext):
            self.token = node.start
            self.end_token = node.stop


class PySyntaxError(SemanticError):
    pass
