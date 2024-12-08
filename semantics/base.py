from typing import Optional

from antlr4.Token import CommonToken


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
