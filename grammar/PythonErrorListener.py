import logging

from antlr4.error.ErrorListener import ErrorListener
from antlr4.error.Errors import RecognitionException

logger = logging.getLogger(__name__)


class PythonErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        if isinstance(e, RecognitionException):
            logger.info(f"syntax error at {line}:{column}: {msg}")
            e.message = msg
