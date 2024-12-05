import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterable, Optional

from .scope import SymbolTable
from .symbol import Symbol, SymbolType

if TYPE_CHECKING:
    from .entity import PyClass, PyFunction, PyModule


class _TypeContext(threading.local):
    def __init__(self):
        self.contexts: dict[str, SymbolTable] = {}


_type_context = _TypeContext()


@contextmanager
def set_type_context(contexts: dict[str, SymbolTable]):
    _type_context.contexts = contexts
    yield
    _type_context.contexts = {}


def get_context_symbol(name: str) -> Optional[Symbol]:
    """
    Retrieves a symbol from the current type context by name.

    Args:
        name: The name of the symbol to retrieve. If the name is fully qualified, it
            should be in the form `module.name`. Otherwise, the symbol is looked up in
            the builtins context.
    """
    if "." in name:
        context, name = name.rsplit(".", 1)
    else:
        context = "builtins"

    if scope := _type_context.contexts.get(context):
        return scope.get(name)
    return None


def get_context_cls(name: str) -> Optional["PyClass"]:
    from .entity import PyClass

    if (
        (symbol := get_context_symbol(name))
        and symbol.type is SymbolType.CLASS
        and (entity := symbol.entity)
        and isinstance(entity, PyClass)
    ):
        return entity

    return None


class PyType:
    """
    A Python type.
    """

    ANY: "PyType"

    def get_attr(self, name: str) -> Optional[Symbol]:
        """
        Finds an attribute of the type by name.
        """
        return None

    def attrs(self) -> Iterable[Symbol]:
        """
        Returns an iterable of all attributes of the type.
        """
        visited: set[str] = set()
        for symbol in self._attrs():
            if symbol.name not in visited:
                visited.add(symbol.name)
                yield symbol

    def _attrs(self) -> Iterable[Symbol]:
        return ()

    def get_return_type(self) -> "PyType":
        """
        Returns the type obtained by calling the type.
        """
        return PyType.ANY

    def get_subscripted_type(self) -> "PyType":
        """
        Returns the type obtained by subscripting the type.
        """
        return PyType.ANY

    def get_awaited_type(self) -> "PyType":
        """
        Returns the type obtained by awaiting the type.
        """
        return PyType.ANY


PyType.ANY = PyType()


class PyModuleType(PyType):
    def __init__(self, module: "PyModule"):
        self.module = module

    def get_attr(self, name: str) -> Optional[Symbol]:
        return self.module.context.global_scope.get(name)

    def _attrs(self) -> Iterable[Symbol]:
        yield from self.module.context.global_scope.symbols()


class PyClassType(PyType):
    def __init__(self, cls: "PyClass"):
        self.cls = cls

    def get_attr(self, name: str) -> Optional[Symbol]:
        # TODO: Look up in the class hierarchy.
        return self.cls.scope.get(name)

    def _attrs(self) -> Iterable[Symbol]:
        yield from self.cls.scope.symbols()

    def get_return_type(self) -> PyType:
        return PyInstanceType(self.cls)

    def get_subscripted_type(self) -> PyType:
        return self.cls.get_method_return_type("__class_getitem__")

    @classmethod
    def from_builtin(cls, name: str) -> PyType:
        if cls := get_context_cls(name):
            return PyClassType(cls)
        return PyType.ANY


class PyInstanceType(PyType):
    def __init__(self, cls: "PyClass"):
        self.cls = cls

    def get_attr(self, name: str) -> Optional[Symbol]:
        # TODO: Look up in the class hierarchy.
        if symbol := self.cls.instance_scope.get(name):
            return symbol
        return self.cls.scope.get(name)

    def _attrs(self) -> Iterable[Symbol]:
        yield from self.cls.instance_scope.symbols()
        yield from self.cls.scope.symbols()

    def get_return_type(self) -> PyType:
        return self.cls.get_method_return_type("__call__")

    def get_subscripted_type(self) -> PyType:
        return self.cls.get_method_return_type("__getitem__")

    def get_awaited_type(self) -> PyType:
        return self.cls.get_method_return_type("__await__")

    @classmethod
    def from_builtin(cls, name: str) -> PyType:
        if cls := get_context_cls(name):
            return PyInstanceType(cls)
        return PyType.ANY


class PyFunctionType(PyType):
    def __init__(self, func: "PyFunction"):
        self.func = func

    def get_attr(self, name: str) -> Optional[Symbol]:
        if function_cls := get_context_cls("function"):
            return PyClassType(function_cls).get_attr(name)
        return None

    def _attrs(self) -> Iterable[Symbol]:
        if function_cls := get_context_cls("function"):
            yield from PyClassType(function_cls)._attrs()

    def get_return_type(self) -> PyType:
        return self.func.return_type or PyType.ANY


type SimpleLiteral = bool | int | float | complex | str | bytes
type CompoundLiteral = (
    SimpleLiteral
    | tuple[CompoundLiteral]
    | list[CompoundLiteral]
    | set[CompoundLiteral]
    | dict[CompoundLiteral, CompoundLiteral]
)

BUILTIN_LITERAL_TYPES: set[type] = {
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    tuple,
    list,
    set,
    dict,
}


class PyLiteralType(PyType):
    def __init__(self, value: CompoundLiteral):
        self.value = value

    @property
    def value_type(self) -> type:
        return type(self.value)

    @property
    def value_type_name(self) -> str:
        return self.value_type.__name__

    def get_attr(self, name: str) -> Optional[Symbol]:
        if self.value is None:
            return PyInstanceType.from_builtin("types.NoneType").get_attr(name)

        if self.value_type in BUILTIN_LITERAL_TYPES:
            return PyInstanceType.from_builtin(self.value_type_name).get_attr(name)

    def _attrs(self) -> Iterable[Symbol]:
        if self.value is None:
            yield from PyInstanceType.from_builtin("types.NoneType")._attrs()

        if self.value_type in BUILTIN_LITERAL_TYPES:
            yield from PyInstanceType.from_builtin(self.value_type_name)._attrs()
