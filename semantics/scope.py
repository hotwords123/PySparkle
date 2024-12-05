from enum import Enum
from typing import Iterable, Optional

from antlr4.Token import CommonToken

from .base import SemanticError
from .symbol import Symbol


class ScopeType(Enum):
    BUILTINS = "builtins"
    GLOBAL = "global"
    LOCAL = "local"
    CLASS = "class"
    OBJECT = "object"
    LAMBDA = "lambda"
    COMPREHENSION = "comprehension"


class SymbolTable:
    def __init__(
        self, name: str, scope_type: ScopeType, parent: Optional["SymbolTable"] = None
    ):
        self.name = name
        self.scope_type = scope_type
        self.parent = parent
        self._symbols: dict[str, Symbol] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._symbols

    def __getitem__(self, name: str) -> Symbol:
        return self._symbols[name]

    def get(self, name: str) -> Optional[Symbol]:
        return self._symbols.get(name)

    def define(self, symbol: Symbol):
        """
        Defines a symbol in the current scope. Raises an error if the symbol is already
        defined in the current scope.

        Args:
            symbol: The symbol to define.
        """
        if symbol.name in self._symbols:
            raise PyDuplicateSymbolError(symbol, self)
        self._symbols[symbol.name] = symbol
        # print(f"Defined symbol {symbol.name} in scope {self.name}")

    def lookup(
        self,
        name: str,
        parents: bool = True,
        globals: bool = True,
        raise_from: Optional[CommonToken] = None,
    ) -> Optional[Symbol]:
        """
        Looks up a symbol in the current scope and possibly its parents. Returns None if the
        symbol is not found.

        Args:
            name: The name of the symbol to look up.
            parents: Whether to look up the symbol in the parent scopes.
            globals: Whether to look up the symbol in the global scope.
            raise_from: The token to raise an error from if the symbol is not found.

        Returns:
            symbol: The symbol if found, or None if not found.
        """
        scope = self
        while scope is not None:
            if not globals and scope.scope_type is ScopeType.GLOBAL:
                break
            if name in scope._symbols:
                return scope._symbols[name]
            if not parents:
                break
            scope = scope.parent
        if raise_from is not None:
            raise PySymbolNotFoundError(name, raise_from, self)
        return None

    def symbols(
        self,
        parents: bool = False,
        public_only: bool = False,
    ) -> Iterable[Symbol]:
        """
        Returns an iterable of all symbols in the scope and possibly its parent scopes.

        Args:
            parents: Whether to iterate over the parent scopes.
            public_only: Whether to only iterate over public symbols.

        Returns:
            symbols: An iterable of symbols.
        """
        if public_only:
            assert (
                not parents and self.scope_type is ScopeType.GLOBAL
            ), "Public symbols are only defined in the global scope"

        scope = self
        while scope is not None:
            for symbol in self._symbols.values():
                if public_only and not symbol.public:
                    continue
                yield symbol

            if not parents:
                break
            scope = scope.parent


class PyNameError(SemanticError):
    def __init__(self, token: CommonToken, scope: SymbolTable, message: str):
        super().__init__(message, token)
        self.scope = scope


class PyDuplicateSymbolError(PyNameError):
    def __init__(self, symbol: Symbol, scope: SymbolTable):
        super().__init__(
            symbol.token,
            scope,
            f"Symbol {symbol.name} already defined in scope {scope.name}",
        )
        self.symbol = symbol


class PySymbolNotFoundError(PyNameError):
    def __init__(self, name: str, token: CommonToken, scope: SymbolTable):
        super().__init__(token, scope, f"Symbol {name} not found in scope {scope.name}")
