"""
This module defines the types used in the type system of the Python semantic analyzer.

References
- https://peps.python.org/pep-0483/
- https://peps.python.org/pep-0484/
- https://docs.python.org/3/library/typing.html
- https://docs.python.org/3/library/types.html
"""

import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from types import EllipsisType, NoneType
from typing import TYPE_CHECKING, Iterable, Optional, final

from .scope import SymbolTable
from .symbol import Symbol, SymbolType

if TYPE_CHECKING:
    from .entity import PyClass, PyEntity, PyFunction, PyModule
    from .structure import PythonContext


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


def get_context_entity(name: str) -> Optional["PyEntity"]:
    if symbol := get_context_symbol(name):
        return symbol.resolve_entity()
    return None


def get_context_cls(name: str) -> Optional["PyClass"]:
    from .entity import PyClass

    if isinstance(entity := get_context_entity(name), PyClass):
        return entity

    return None


def get_context_func(name: str) -> Optional["PyFunction"]:
    from .entity import PyFunction

    if isinstance(entity := get_context_entity(name), PyFunction):
        return entity

    return None


class PyType(ABC):
    """
    A Python type.
    """

    ANY: "_PyAnyType"

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns the string representation of the type.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __eq__(self, other: object) -> bool:
        """
        Checks if two types are equal.
        """
        return True if self is other else NotImplemented

    @property
    def entity(self) -> Optional["PyEntity"]:
        """
        Returns the entity associated with the type, if any.
        """
        return None

    def attr_scopes(self) -> Iterable[SymbolTable]:
        """
        Returns an iterable of all attribute scopes of the type in order of precedence.

        The default implementation of `get_attr` and `attrs` uses this method to lookup
        and iterate over the attributes of the type.
        """
        return ()

    def get_attr(self, name: str) -> Optional[Symbol]:
        """
        Finds an attribute of the type by name.
        """
        for scope in self.attr_scopes():
            if symbol := scope.get(name):
                return symbol

        return None

    def attrs(self) -> Iterable[Symbol]:
        """
        Returns an iterable of all attributes of the type, removing duplicates.
        """
        visited: set[str] = set()
        for symbol in self._attrs():
            if symbol.name not in visited:
                visited.add(symbol.name)
                yield symbol

    def _attrs(self) -> Iterable[Symbol]:
        """
        Returns an iterable of all attributes of the type.
        """
        for scope in self.attr_scopes():
            yield from scope.symbols()

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

    def get_annotated_type(self, context: "PythonContext") -> "PyType":
        """
        Returns the type of the annotation.

        This is used for type annotations, e.g. determining the type of a variable from
        its annotation.

        Args:
            context: The context in which the annotation is used.
        """
        return PyType.ANY

    def get_inferred_type(self) -> "PyType":
        """
        Returns the type inferred from the type.

        This is used for type inference, e.g. determining the type of a variable from
        its initializer.
        """
        return self


@final
class _PyAnyType(PyType):
    def __str__(self) -> str:
        return "Any"

    def __eq__(self, other: object) -> bool:
        return other is PyType.ANY


PyType.ANY = _PyAnyType()


@final
class PyModuleType(PyType):
    def __init__(self, module: "PyModule"):
        self.module = module

    def __str__(self) -> str:
        return f"<module {self.module.name!r}>"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} module={self.module!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyModuleType) and self.module is other.module

    @property
    def entity(self) -> "PyEntity":
        return self.module

    def attr_scopes(self) -> Iterable[SymbolTable]:
        yield self.module.context.global_scope

        if module_cls := get_context_cls("types.ModuleType"):
            yield from module_cls.mro_scopes()


@final
class PyClassType(PyType):
    def __init__(self, cls: "PyClass"):
        self.cls = cls

    def __str__(self) -> str:
        return f"<class {self.cls.name!r}>"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} class={self.cls!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyClassType) and self.cls is other.cls

    @property
    def entity(self) -> "PyEntity":
        return self.cls

    def attr_scopes(self) -> Iterable[SymbolTable]:
        return self.cls.mro_scopes()

    def get_return_type(self) -> PyType:
        return PyInstanceType(self.cls)

    def get_subscripted_type(self) -> PyType:
        return self.cls.get_method_return_type("__class_getitem__")

    def get_annotated_type(self, context: "PythonContext") -> PyType:
        return self.cls.get_instance_type()

    @classmethod
    def from_builtin(cls, name: str) -> PyType:
        if entity := get_context_cls(name):
            return cls(entity)
        return PyType.ANY


class PyInstanceType(PyType):
    def __init__(self, cls: "PyClass"):
        self.cls = cls

    def __str__(self) -> str:
        if self.cls is get_context_cls("types.NoneType"):
            return "None"

        return self.cls.name

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} class={self.cls!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyInstanceType) and self.cls is other.cls

    def attr_scopes(self) -> Iterable[SymbolTable]:
        return self.cls.mro_scopes(instance=True)

    def get_return_type(self) -> PyType:
        return self.cls.get_method_return_type("__call__")

    def get_subscripted_type(self) -> PyType:
        return self.cls.get_method_return_type("__getitem__")

    def get_awaited_type(self) -> PyType:
        return self.cls.get_method_return_type("__await__")

    @classmethod
    def from_builtin(cls, name: str) -> PyType:
        if entity := get_context_cls(name):
            return cls(entity)
        return PyType.ANY


@final
class PySelfType(PyInstanceType):
    def __init__(self, cls: "PyClass"):
        super().__init__(cls)

    def __str__(self) -> str:
        return f"Self@{super().__str__()}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PySelfType) and super().__eq__(other)


@final
class PyFunctionType(PyType):
    def __init__(self, func: "PyFunction"):
        self.func = func

    def __str__(self) -> str:
        return str(self.func)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} function={self.func!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyFunctionType) and self.func is other.func

    @property
    def entity(self) -> "PyEntity":
        return self.func

    def attr_scopes(self) -> Iterable[SymbolTable]:
        if function_cls := get_context_cls("types.FunctionType"):
            return PyClassType(function_cls).attr_scopes()
        return ()

    def get_return_type(self) -> PyType:
        return self.func.return_type or PyType.ANY

    def get_inferred_type(self) -> PyType:
        # TODO: Infer the callable type from the function.
        return self

    @staticmethod
    def from_builtin(name: str) -> PyType:
        if func := get_context_func(name):
            return PyFunctionType(func)
        return PyType.ANY


type SimpleLiteral = None | bool | int | float | complex | str | bytes | EllipsisType
type CompoundLiteral = (
    SimpleLiteral
    | tuple[CompoundLiteral]
    | list[CompoundLiteral]
    | set[CompoundLiteral]
    | dict[CompoundLiteral, CompoundLiteral]
)

SIMPLE_LITERAL_TYPES: set[type] = {
    NoneType,
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    EllipsisType,
}
COMPUND_LITERAL_TYPES: set[type] = {tuple, list, set, dict}
LITERAL_TYPES = SIMPLE_LITERAL_TYPES | COMPUND_LITERAL_TYPES

LITERAL_TYPE_MAP: dict[type, str] = {
    x: f"types.{x.__name__}" if x in (NoneType, EllipsisType) else x.__name__
    for x in LITERAL_TYPES
}


@final
class PyLiteralType(PyType):
    def __init__(self, value: CompoundLiteral):
        self.value = value

    def __str__(self) -> str:
        if self.value is None:
            return "None"

        if self.value is ...:
            return "EllipsisType"

        return f"Literal[{self.value!r}]"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} value={self.value!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyLiteralType) and self.value == other.value

    @property
    def value_type(self) -> type:
        return type(self.value)

    @property
    def value_type_name(self) -> str:
        return self.value_type.__name__

    def get_base_type(self) -> PyInstanceType:
        """
        Returns the base type of the literal.
        """
        return PyInstanceType.from_builtin(LITERAL_TYPE_MAP[self.value_type])

    def attr_scopes(self) -> Iterable[SymbolTable]:
        return self.get_base_type().attr_scopes()

    def get_annotated_type(self, context: "PythonContext") -> PyType:
        # None stands its own type in type annotations.
        if self.value is None:
            return PyInstanceType.from_builtin("types.NoneType")

        # String literals can represent forward references.
        if self.value_type is str:
            # TODO: Parse the string literal as tree and resolve the type.
            if symbol := context.current_scope.lookup(self.value):
                return symbol.get_type().get_annotated_type(context)

        # Other literals cannot be directly used for type annotations.
        return PyType.ANY

    def get_inferred_type(self) -> PyType:
        # TODO: Handle compound literals.
        return self.get_base_type()
