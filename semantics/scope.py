from enum import Enum
from typing import Optional

from .base import SemanticError
from .symbol import Symbol, SymbolType


class ScopeType(Enum):
    BUILTINS = "builtins"
    GLOBAL = "global"
    LOCAL = "local"
    CLASS = "class"
    LAMBDA = "lambda"
    COMPREHENSION = "comprehension"


class SymbolTable:
    def __init__(
        self, name: str, scope_type: ScopeType, parent: Optional["SymbolTable"] = None
    ):
        self.name = name
        self.scope_type = scope_type
        self.parent = parent
        self.symbols: dict[str, Symbol] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.symbols

    def __getitem__(self, name: str) -> Symbol:
        return self.symbols[name]

    def get(self, name: str) -> Optional[Symbol]:
        return self.symbols.get(name)

    def define(self, symbol: Symbol):
        """
        Defines a symbol in the current scope. Raises an error if the symbol is already
        defined in the current scope.

        Args:
            symbol: The symbol to define.
        """
        if symbol.name in self.symbols:
            raise DuplicateSymbolError(symbol, self)
        self.symbols[symbol.name] = symbol
        # print(f"Defined symbol {symbol.name} in scope {self.name}")

    def lookup(
        self, name: str, parents: bool = True, globals: bool = True
    ) -> Optional[Symbol]:
        """
        Looks up a symbol in the current scope and possibly its parents. Returns None if the
        symbol is not found.

        Args:
            name: The name of the symbol to look up.
            parents: Whether to look up the symbol in the parent scopes.
            globals: Whether to look up the symbol in the global scope.
        """
        scope = self
        while scope is not None:
            if not globals and scope.scope_type is ScopeType.GLOBAL:
                break
            if name in scope.symbols:
                return scope.symbols[name]
            if not parents:
                break
            scope = scope.parent
        return None


class DuplicateSymbolError(SemanticError):
    def __init__(self, symbol: Symbol, scope: SymbolTable):
        super().__init__(
            f"Symbol {symbol.name} already defined in scope {scope.name}", symbol.token
        )
        self.symbol = symbol
        self.scope = scope
