from enum import Enum
from typing import Iterator, Optional

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
        self,
        name: str,
        scope_type: ScopeType,
        parent: Optional["SymbolTable"] = None,
        *,
        full_name: Optional[str] = None,
    ):
        self.name = name
        self.full_name = full_name or name
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
            raise PyDuplicateSymbolError(symbol, self)
        self.symbols[symbol.name] = symbol
        symbol.set_parent_scope(self)

    def lookup(
        self,
        name: str,
        parents: bool = True,
        globals: bool = True,
        raise_error: bool = False,
    ) -> Optional[Symbol]:
        """
        Looks up a symbol in the current scope and possibly its parents. Returns None if the
        symbol is not found.

        Args:
            name: The name of the symbol to look up.
            parents: Whether to look up the symbol in the parent scopes.
            globals: Whether to look up the symbol in the global scope.
            raise_error: Whether to raise an error if the symbol is not found.

        Returns:
            symbol: The symbol if found, or None if not found.
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
        if raise_error:
            raise PySymbolNotFoundError(name, self)
        return None

    def iter_symbols(
        self,
        parents: bool = False,
        public_only: bool = False,
    ) -> Iterator[Symbol]:
        """
        Yields the symbols in the current scope and possibly its parents.

        Args:
            parents: Whether to iterate over the parent scopes.
            public_only: Whether to only iterate over public symbols.

        Yields:
            symbol: The symbol in the scope.
        """
        if public_only:
            assert (
                not parents and self.scope_type is ScopeType.GLOBAL
            ), "Public symbols are only defined in the global scope"

        scope = self
        while scope is not None:
            for symbol in self.symbols.values():
                if public_only and not symbol.public:
                    continue
                yield symbol

            if not parents:
                break
            scope = scope.parent


class PyNameError(SemanticError):
    def __init__(self, scope: SymbolTable, message: str):
        super().__init__(message)
        self.scope = scope


class PyDuplicateSymbolError(PyNameError):
    def __init__(self, symbol: Symbol, scope: SymbolTable):
        super().__init__(
            scope,
            f"Symbol {symbol.name} already defined in scope {scope.name}",
        )
        self.symbol = symbol


class PySymbolNotFoundError(PyNameError):
    def __init__(self, name: str, scope: SymbolTable):
        super().__init__(scope, f"Symbol {name} not found in scope {scope.name}")
        self.name = name
