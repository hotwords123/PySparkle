from enum import Enum
from typing import TYPE_CHECKING, Optional

from antlr4.Token import CommonToken
from antlr4.tree.Tree import TerminalNode

if TYPE_CHECKING:
    from .entity import PyEntity


class SymbolType(Enum):
    VARIABLE = "variable"
    PARAMETER = "parameter"
    FUNCTION = "function"
    CLASS = "class"
    IMPORTED = "import"
    GLOBAL = "global"
    NONLOCAL = "nonlocal"


class Symbol:
    def __init__(
        self,
        type_: SymbolType,
        name: str,
        node: Optional[TerminalNode] = None,
        *,
        public: Optional[bool] = None,
        target: Optional["Symbol"] = None,
        entity: Optional["PyEntity"] = None,
    ):
        self.name = name
        self.type = type_
        self.public = public

        self.node = node
        self.token: Optional[CommonToken] = node and node.getSymbol()

        self.target = target
        self.entity = entity

    def __str__(self):
        return f"<{self.__class__.__name__}(name={self.name}, type={self.type})>"

    def is_outer(self) -> bool:
        """
        Returns whether the symbol is an outer symbol (global or nonlocal).
        """
        return self.type in (SymbolType.GLOBAL, SymbolType.NONLOCAL)

    def resolve(self) -> "Symbol":
        """
        Resolves the symbol to its target.
        """
        symbol = self
        while symbol.target is not None:
            symbol = symbol.target
        return symbol
