from enum import Enum
from typing import TYPE_CHECKING, Optional

from antlr4.Token import CommonToken
from antlr4.tree.Tree import TerminalNode

if TYPE_CHECKING:
    from .entity import PyEntity
    from .scope import SymbolTable
    from .types import PyType


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
        type: SymbolType,
        name: str,
        node: Optional[TerminalNode] = None,
        *,
        public: Optional[bool] = None,
        target: Optional["Symbol"] = None,
        entity: Optional["PyEntity"] = None,
        full_name: Optional[str] = None,
    ):
        self.name = name
        self.type = type
        self.public = public

        self.node = node
        self.token: Optional[CommonToken] = node and node.getSymbol()

        self.target = target
        self.entity = entity
        self._full_name = full_name

        self.set_entity_name()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} type={self.type}>"

    @property
    def full_name(self) -> str:
        """
        Returns the fully-qualified name of the symbol.
        """
        return self._full_name or self.name

    def set_parent_scope(self, scope: "SymbolTable"):
        if self._full_name is None:
            self._full_name = f"{scope.full_name}.{self.name}"
            self.set_entity_name()

    def set_entity(self, entity: "PyEntity"):
        self.entity = entity
        self.set_entity_name()

    def set_entity_name(self):
        if self.entity is not None and self._full_name is not None:
            self.entity.set_full_name(self.full_name)

    def copy(
        self,
        node: Optional[TerminalNode] = ...,
        public: Optional[bool] = ...,
        target: Optional["Symbol"] = ...,
    ) -> "Symbol":
        """
        Returns a copy of the symbol.

        Args:
            node: The new node for the symbol.
            public: The new public flag for the symbol.
            target: The new target for the symbol.
        """
        return Symbol(
            self.type,
            self.name,
            self.node if node is Ellipsis else node,
            public=self.public if public is Ellipsis else public,
            target=self.target if target is Ellipsis else target,
            entity=self.entity,
        )

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
        visited = {self}
        while symbol.target is not None:
            symbol = symbol.target
            if symbol in visited:
                break  # Cycles may occur due to circular imports
            visited.add(symbol)
        return symbol

    def resolve_entity(self) -> Optional["PyEntity"]:
        """
        Resolves the symbol to its entity.
        """
        symbol = self
        visited = set()
        while True:
            if symbol.entity is not None:
                return symbol.entity
            visited.add(symbol)
            symbol = symbol.target
            if symbol is None or symbol in visited:
                return None

    def get_type(self) -> "PyType":
        """
        Returns the type of the symbol's entity, if any.
        """
        from .types import PyType

        if entity := self.resolve_entity():
            return entity.get_type()

        return PyType.ANY
