from contextlib import contextmanager
from typing import Iterator, Unpack

from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Token import CommonToken
from antlr4.tree.Tree import TerminalNode

from .scope import ScopeType, SymbolTable
from .token import TokenInfo


class PythonContext:
    """
    Represents the context of a Python module being analyzed.
    """

    def __init__(self, global_scope: SymbolTable):
        self.global_scope: SymbolTable = global_scope
        self.current_scope: SymbolTable = global_scope
        self.scopes: dict[ParserRuleContext, SymbolTable] = {}

        self.token_info: dict[CommonToken, TokenInfo] = {}

        self.errors: list[Exception] = []

    @contextmanager
    def scope_guard(self, scope: SymbolTable) -> Iterator[SymbolTable]:
        old_scope = self.current_scope
        self.current_scope = scope
        try:
            yield scope
        finally:
            self.current_scope = old_scope

    def new_scope(self, ctx: ParserRuleContext, name: str, scope_type: ScopeType):
        parent_scope = self.current_scope
        if parent_scope.scope_type is ScopeType.CLASS:
            parent_scope = parent_scope.parent

        scope = SymbolTable(name, scope_type, parent_scope)
        self.scopes[ctx] = scope
        return self.scope_guard(scope)

    def scope_of(self, ctx: ParserRuleContext):
        assert (scope := self.scopes[ctx]) is not None
        return self.scope_guard(scope)

    def parent_scope(self):
        assert (scope := self.current_scope.parent) is not None
        return self.scope_guard(scope)

    def set_node_info(self, node: TerminalNode, /, **kwargs: Unpack[TokenInfo]):
        token = node.getSymbol()
        self.token_info.setdefault(token, TokenInfo()).update(**kwargs)

    @contextmanager
    def wrap_errors(self, error_cls: type[Exception]) -> Iterator[None]:
        try:
            yield
        except error_cls as e:
            self.errors.append(e)
