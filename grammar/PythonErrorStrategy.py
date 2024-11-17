from antlr4 import Token
from antlr4.error.ErrorStrategy import DefaultErrorStrategy
from antlr4.error.Errors import InputMismatchException

from .PythonParser import PythonParser


class PythonErrorStrategy(DefaultErrorStrategy):
    STRUCURAL_TOKENS = (
        PythonParser.NEWLINE,
        PythonParser.INDENT,
        PythonParser.DEDENT,
        PythonParser.COLON,
        Token.EOF,
    )

    STATEMENT_RULES = {
        PythonParser.RULE_statement,
        PythonParser.RULE_compound_stmt,
        PythonParser.RULE_simple_stmts,
        PythonParser.RULE_function_def,
        PythonParser.RULE_if_stmt,
        PythonParser.RULE_class_def,
        PythonParser.RULE_with_stmt,
        PythonParser.RULE_for_stmt,
        PythonParser.RULE_try_stmt,
        PythonParser.RULE_while_stmt,
        # PythonParser.RULE_match_stmt,
    }

    def singleTokenDeletion(self, recognizer: PythonParser):
        cur_token_type = recognizer.getTokenStream().LA(1)
        # Do not perform deletion if the current token is structural.
        if cur_token_type in self.STRUCURAL_TOKENS:
            return None

        return super().singleTokenDeletion(recognizer)

    def singleTokenInsertion(self, recognizer: PythonParser):
        expected_tokens = set(self.getExpectedTokens(recognizer))
        # Do not perform insertion if all expected tokens are structural.
        if all(token in self.STRUCURAL_TOKENS for token in expected_tokens):
            return None

        return super().singleTokenInsertion(recognizer)

    def consumePastUnless(
        self, recognizer: PythonParser, past: set[int], unless: set[int]
    ):
        while True:
            ttype = recognizer.getTokenStream().LA(1)
            if ttype == Token.EOF or ttype in unless:
                break
            recognizer.consume()
            if ttype in past:
                break

    def recover(self, recognizer: PythonParser, e: InputMismatchException):
        rule_index = recognizer._ctx.getRuleIndex()
        if rule_index in self.STATEMENT_RULES:
            # Consume the rest of the line, unless an EOF or DEDENT is encountered.
            self.consumePastUnless(
                recognizer, {PythonParser.NEWLINE}, {PythonParser.DEDENT}
            )
            return

        return super().recover(recognizer, e)