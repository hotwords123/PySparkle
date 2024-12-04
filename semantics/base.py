from typing import Optional

from antlr4.Token import CommonToken


class SemanticError(Exception):
    def __init__(self, message: str, token: Optional[CommonToken] = None):
        self.message = message
        self.token = token

    def __str__(self):
        return f"{self.message} at {self.token.line}:{self.token.column}"
