import itertools
import sys
from typing import TextIO

from antlr4 import InputStream, Lexer, Token
from antlr4.Recognizer import ProxyErrorListener
from antlr4.Token import CommonToken


class PythonLexerBase(Lexer):
    def __init__(self, input: InputStream, output: TextIO = sys.stdout):
        super().__init__(input, output)

        # The current token.
        self.__cur_token: CommonToken | None
        # The buffer of lookahead tokens.
        self.__lookahead_tokens: list[CommonToken]

        # The queue of pending tokens to emit.
        self.__pending_tokens: list[CommonToken]
        # Whether we are at the beginning of a logical line.
        self.__at_logical_newline: bool
        # Whether the current line is blank.
        self.__blank_line: bool

        # The stack of indentation levels.
        self.__indent_stack: list[int]
        # The number of opened brackets (round, square, and curly).
        self.__opened_brackets: int

        self.__init()

    def nextToken(self) -> CommonToken:
        if not self.__pending_tokens:
            if self.__eof:
                return self.__cur_token
            self.__fill_pending_tokens()
        return self.__pending_tokens.pop(0)

    def reset(self):
        super().reset()
        self.__init()

    def __init(self):
        self.__cur_token = None
        self.__lookahead_tokens = []

        self.__pending_tokens = []
        self.__at_logical_newline = True
        self.__blank_line = False

        self.__indent_stack = [0]
        self.__opened_brackets = 0

    @property
    def __eof(self) -> bool:
        return self.__cur_token and self.__cur_token.type == Token.EOF

    def __advance(self):
        if self.__lookahead_tokens:
            self.__cur_token = self.__lookahead_tokens.pop(0)
        elif not self.__eof:
            self.__cur_token = super().nextToken()

    def __lookahead(self, n: int) -> CommonToken:
        if n == 0:
            return self.__cur_token

        last_token = (
            self.__lookahead_tokens[-1] if self.__lookahead_tokens else self.__cur_token
        )

        while len(self.__lookahead_tokens) < n:
            if last_token.type == Token.EOF:
                return last_token
            last_token = super().nextToken()
            self.__lookahead_tokens.append(last_token)

        return self.__lookahead_tokens[n - 1]

    def __fill_pending_tokens(self):
        self.__advance()

        if self.__at_logical_newline:
            self.__blank_line = self.__check_blank_line()
            if not self.__blank_line:
                self.__handle_indentation()

        match self.__cur_token.type:
            case self.LPAREN | self.LSQB | self.LBRACE:
                self.__opened_brackets += 1
            case self.RPAREN | self.RSQB | self.RBRACE:
                self.__opened_brackets -= 1
            case self.NEWLINE:
                self.__handle_NEWLINE()
            case self.ERROR:
                self.__report_error("invalid character")
            case Token.EOF:
                self.__handle_EOF()

        self.__emit_token(self.__cur_token)

        if self.__cur_token.type != self.NEWLINE:
            self.__at_logical_newline = False

    def __emit_token(self, token: CommonToken):
        self.__pending_tokens.append(token)

    def __hide_current_token(self):
        self.__cur_token.channel = Token.HIDDEN_CHANNEL

    def __create_and_emit_token(
        self, token_type: int, channel: int = Token.DEFAULT_CHANNEL, text: str = ""
    ):
        token = CommonToken(
            self.__cur_token.source,
            token_type,
            channel,
            start=self.__cur_token.start,
            stop=self.__cur_token.start - 1,
        )
        token.line = self.__cur_token.line
        token.column = self.__cur_token.column
        token.text = text
        self.__emit_token(token)

    def __check_blank_line(self) -> bool:
        for i in itertools.count():
            match self.__lookahead(i).type:
                case self.WS | self.COMMENT | self.EXPLICIT_LINE_JOINING:
                    continue
                case self.NEWLINE | Token.EOF:
                    return True
                case _:
                    return False

    def __handle_NEWLINE(self):
        # Implicit line joining inside brackets.
        if self.__opened_brackets > 0:
            self.__hide_current_token()
            self.__at_logical_newline = False
            return

        self.__at_logical_newline = True
        # Ignore blank lines.
        if self.__blank_line:
            self.__hide_current_token()

    def __handle_indentation(self):
        if self.__cur_token.type == self.WS:
            level = self.__get_indentation_level(self.__cur_token.text)
        else:
            level = 0

        if level > self.__indent_stack[-1]:
            self.__indent_stack.append(level)
            self.__create_and_emit_token(self.INDENT, text="<INDENT>")
        else:
            while level < self.__indent_stack[-1]:
                self.__indent_stack.pop()
                self.__create_and_emit_token(self.DEDENT, text="<DEDENT>")
            if level != self.__indent_stack[-1]:
                self.__report_error("inconsistent indentation")
                self.__create_and_emit_token(self.ERROR, text="<ERROR>")

    def __get_indentation_level(self, ws: str) -> int:
        TAB_LENGTH = 8

        length = 0
        for c in ws:
            match c:
                case " ":
                    length += 1
                case "\t":
                    length += TAB_LENGTH - length % TAB_LENGTH
                case "\f":
                    length = 0
                case _:
                    raise ValueError(f"unexpected character: {c!r}")
        return length

    def __handle_EOF(self):
        if not self.__blank_line:
            self.__create_and_emit_token(self.NEWLINE, text="<NEWLINE>")

        while len(self.__indent_stack) > 1:
            self.__indent_stack.pop()
            self.__create_and_emit_token(self.DEDENT, text="<DEDENT>")

    def __report_error(self, message: str):
        listener: ProxyErrorListener = self.getErrorListenerDispatch()
        listener.syntaxError(
            self,
            self.__cur_token,
            self.__cur_token.line,
            self.__cur_token.column,
            message,
            None,
        )
