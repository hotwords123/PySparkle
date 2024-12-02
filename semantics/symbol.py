from enum import Enum
from typing import Optional


class SymbolType(Enum):
    VARIABLE = "variable"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"


class Symbol:
    def __init__(self, name: str, type_: SymbolType):
        self.name = name
        self.type = type_

    def __str__(self):
        return f"<{self.__class__.__name__}(name={self.name}, type={self.type})>"


class SymbolTable:
    def __init__(self, name: str, parent: Optional["SymbolTable"] = None):
        self.name = name
        self.parent = parent
        self.symbols = {}

    def define(self, name: str, value: Symbol):
        assert name not in self.symbols, f"Symbol {name} already defined"
        self.symbols[name] = value

    def lookup(self, name: str, parents: bool = True) -> Optional[Symbol]:
        if name in self.symbols:
            return self.symbols[name]
        if parents and self.parent is not None:
            return self.parent.lookup(name)
        return None


class VariableSymbol(Symbol):
    def __init__(self, name: str):
        super().__init__(name, SymbolType.VARIABLE)


class FunctionSymbol(Symbol):
    def __init__(self, name: str):
        super().__init__(name, SymbolType.FUNCTION)


class ClassSymbol(Symbol):
    def __init__(self, name: str):
        super().__init__(name, SymbolType.CLASS)


class ModuleSymbol(Symbol):
    def __init__(self, name: str):
        super().__init__(name, SymbolType.MODULE)
