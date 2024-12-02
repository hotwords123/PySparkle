from antlr4.RuleContext import RuleContext
from antlr4.Token import CommonToken
from antlr4.tree.Tree import ErrorNode, TerminalNode

from grammar import PythonParserVisitor

from .highlight import TOKEN_KIND_MAP, TokenKind
from .symbol import SymbolTable, VariableSymbol


class PythonVisitor(PythonParserVisitor):

    def __init__(self):
        self.builtins_scope = SymbolTable("<builtins>")
        self.global_scope = SymbolTable("<global>", self.builtins_scope)
        self.current_scope = self.global_scope

        self.token_kinds: dict[CommonToken, TokenKind] = {}

    def push_scope(self, name: str):
        self.current_scope = SymbolTable(name, self.current_scope)

    def pop_scope(self):
        assert self.current_scope.parent is not None
        self.current_scope = self.current_scope.parent

    def visitChildren(self, node: RuleContext):
        for child in node.getChildren():
            self.visit(child)

    def visitTerminal(self, node: TerminalNode):
        token = node.getSymbol()
        token_kind = TOKEN_KIND_MAP.get(token.type, TokenKind.NONE)
        self.token_kinds[token] = token_kind

    def visitErrorNode(self, node: ErrorNode):
        token = node.getSymbol()
        self.token_kinds[token] = TokenKind.ERROR
