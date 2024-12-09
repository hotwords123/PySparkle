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
from typing import TYPE_CHECKING, Iterable, Iterator, Literal, Optional, final, overload

from .scope import ScopeType, SymbolTable
from .symbol import Symbol

if TYPE_CHECKING:
    from .entity import PyClass, PyEntity, PyFunction, PyModule
    from .structure import PythonContext


class _TypeContext(threading.local):
    def __init__(self):
        self.stubs: dict[str, SymbolTable] = {}


_type_context = _TypeContext()


@contextmanager
def set_type_context(stubs: dict[str, SymbolTable]):
    _type_context.stubs = stubs
    yield
    _type_context.stubs = {}


def get_stub_symbol(name: str) -> Optional[Symbol]:
    """
    Retrieves a symbol from the current type context by name.

    Args:
        name: The name of the symbol to retrieve. If the name is fully qualified, it
            should be in the form `module.name`. Otherwise, the symbol is looked up in
            the builtins context.
    """
    context, name = name.rsplit(".", 1)
    if scope := _type_context.stubs.get(context):
        return scope.get(name)
    return None


def get_stub_entity(name: str) -> Optional["PyEntity"]:
    if symbol := get_stub_symbol(name):
        return symbol.resolve_entity()
    return None


_dummy_classes: dict[str, "PyClass"] = {}


@overload
def get_stub_class(name: str, dummy: Literal[False] = ...) -> Optional["PyClass"]: ...


@overload
def get_stub_class(name: str, dummy: Literal[True]) -> "PyClass": ...


def get_stub_class(name: str, dummy: bool = False) -> Optional["PyClass"]:
    from .entity import PyClass

    if isinstance(entity := get_stub_entity(name), PyClass):
        return entity

    if dummy:
        if name in _dummy_classes:
            return _dummy_classes[name]

        cls_name = name.rsplit(".", 1)[-1]
        dummy_scope = SymbolTable(f"<class '{cls_name}'>", ScopeType.CLASS)
        dummy_cls = PyClass(cls_name, dummy_scope)
        _dummy_classes[name] = dummy_cls
        return dummy_cls

    return None


def get_stub_func(name: str) -> Optional["PyFunction"]:
    from .entity import PyFunction

    if isinstance(entity := get_stub_entity(name), PyFunction):
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

    def attr_scopes(self) -> Iterator[SymbolTable]:
        """
        Yields the attribute scopes of the type in order of precedence.

        The default implementation of `get_attr` and `attrs` uses this method to lookup
        and iterate over the attributes of the type.
        """
        return iter(())

    def get_attr(self, name: str) -> Optional[Symbol]:
        """
        Finds an attribute of the type by name.
        """
        for scope in self.attr_scopes():
            if symbol := scope.get(name):
                return symbol

        return None

    def attrs(self) -> Iterator[Symbol]:
        """
        Yields the attributes of the type, removing duplicates.
        """
        visited: set[str] = set()
        for symbol in self._attrs():
            if symbol.name not in visited:
                visited.add(symbol.name)
                yield symbol

    def _attrs(self) -> Iterator[Symbol]:
        """
        Yields the attributes of the type, possibly with duplicates.
        """
        for scope in self.attr_scopes():
            yield from scope.iter_symbols()

    def get_return_type(self) -> "PyType":
        """
        Returns the type obtained by calling the type.
        """
        return PyType.ANY

    def get_subscripted_type(self, key: "PyType") -> "PyType":
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


class PyInstanceBase(PyType, ABC):
    """
    A base class for instance types.
    """

    @abstractmethod
    def get_cls(self) -> "PyClass":
        """
        Returns the class of the instance.
        """
        pass

    def attr_scopes(self) -> Iterator[SymbolTable]:
        return self.get_cls().mro_scopes(instance=True)

    def get_return_type(self) -> PyType:
        return self.get_cls().get_method_return_type("__call__")

    def get_subscripted_type(self, key: PyType) -> PyType:
        return self.get_cls().get_method_return_type("__getitem__")

    def get_awaited_type(self) -> PyType:
        return self.get_cls().get_method_return_type("__await__")


@final
class PyModuleType(PyInstanceBase):
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

    def get_cls(self) -> "PyClass":
        return get_stub_class("types.ModuleType", dummy=True)

    def attr_scopes(self) -> Iterator[SymbolTable]:
        yield self.module.context.global_scope
        yield from super().attr_scopes()


@final
class PyClassType(PyInstanceBase):
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

    def get_cls(self) -> "PyClass":
        return get_stub_class("builtins.type", dummy=True)

    def attr_scopes(self) -> Iterator[SymbolTable]:
        yield from self.cls.mro_scopes()
        yield from super().attr_scopes()

    def get_return_type(self) -> PyType:
        return PyInstanceType(self.cls)

    def get_subscripted_type(self, key: PyType) -> PyType:
        # TODO: Generic alias.
        return self.cls.get_method_return_type("__class_getitem__")

    def get_annotated_type(self, context: "PythonContext") -> PyType:
        return self.cls.get_instance_type()

    @classmethod
    def from_stub(cls, name: str) -> PyType:
        if entity := get_stub_class(name):
            return cls(entity)
        return PyType.ANY


class PyInstanceType(PyInstanceBase):
    def __init__(self, cls: "PyClass"):
        self.cls = cls

    def __str__(self) -> str:
        return self.cls.name

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} class={self.cls!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyInstanceType) and self.cls is other.cls

    def get_cls(self) -> "PyClass":
        return self.cls

    @classmethod
    def from_stub(cls, name: str) -> PyType:
        if entity := get_stub_class(name):
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

    def get_inferred_type(self) -> PyType:
        # The special handling of instance types only applies to the `self` parameter,
        # not to its aliases.
        return PyInstanceType(self.cls)


@final
class PyFunctionType(PyInstanceBase):
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

    def get_cls(self) -> "PyClass":
        return get_stub_class("types.FunctionType", dummy=True)

    def get_return_type(self) -> PyType:
        return self.func.return_type or PyType.ANY

    def get_inferred_type(self) -> PyType:
        # TODO: Infer the callable type from the function.
        return self

    @staticmethod
    def from_stub(name: str) -> PyType:
        if func := get_stub_func(name):
            return PyFunctionType(func)
        return PyType.ANY


@final
class PyNoneType(PyInstanceBase):
    def __str__(self) -> str:
        return "None"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyNoneType)

    def get_cls(self) -> "PyClass":
        return get_stub_class("types.NoneType", dummy=True)

    def get_annotated_type(self, context: "PythonContext") -> PyType:
        # None stands its own type in type annotations.
        return self


@final
class PyEllipsisType(PyInstanceBase):
    def __str__(self) -> str:
        return "EllipsisType"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyEllipsisType)

    def get_cls(self) -> "PyClass":
        return get_stub_class("types.EllipsisType", dummy=True)


type SimpleLiteral = bool | int | float | complex | str | bytes

SIMPLE_LITERAL_TYPES: set[type] = {bool, int, float, complex, str, bytes}

LITERAL_TYPE_MAP: dict[type, str] = {
    x: f"builtins.{x.__name__}" for x in SIMPLE_LITERAL_TYPES
}


@final
class PyLiteralType(PyInstanceBase):
    def __init__(self, value: SimpleLiteral):
        self.value = value

    def __str__(self) -> str:
        return f"Literal[{self.value!r}]"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} value={self.value!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyLiteralType) and self.value == other.value

    def get_cls(self) -> "PyClass":
        return get_stub_class(LITERAL_TYPE_MAP[type(self.value)], dummy=True)

    def get_annotated_type(self, context: "PythonContext") -> PyType:
        # String literals can represent forward references.
        if type(self.value) is str:
            # TODO: Parse the string literal as tree and resolve the type.
            if symbol := context.current_scope.lookup(self.value):
                return symbol.get_type().get_annotated_type(context)

        # Other literals cannot be directly used for type annotations.
        return PyType.ANY

    def get_inferred_type(self) -> PyType:
        return self.get_cls().get_instance_type()


class PyTupleType(PyInstanceBase):
    def __init__(self, types: Iterable[PyType]):
        """
        Creates a tuple type from the given types. The types must not contain PyUnpack.
        """
        self.types = tuple(types)
        assert all(not isinstance(x, PyUnpack) for x in self.types)

    def __str__(self) -> str:
        return f"tuple[{', '.join(map(str, self.types))}]"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} types={self.types!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyTupleType) and self.types == other.types

    def get_cls(self) -> "PyClass":
        return get_stub_class("builtins.tuple", dummy=True)

    def get_subscripted_type(self, key: PyType) -> PyType:
        if isinstance(key, PyLiteralType) and type(key.value) is int:
            index = key.value
            if 0 <= index < len(self.types):
                return self.types[index]

        if isinstance(key, PyInstanceType) and key.cls is get_stub_class(
            "builtins.int"
        ):
            "TODO: Return the union of all types."

        # TODO: Handle slices.

        return PyType.ANY

    def get_inferred_type(self) -> PyType:
        return PyTupleType(tuple(x.get_inferred_type() for x in self.types))

    @staticmethod
    def from_starred(item_types: Iterable[PyType]) -> PyType:
        """
        Creates a tuple type from starred expressions, unpacking the starred items.

        Args:
            item_types: The types of the items in the tuple, including starred items.

        Returns:
            The tuple type.
        """
        unpacked_types: list[PyType] = []

        for item_type in item_types:
            if isinstance(item_type, PyUnpack):
                if inner_types := item_type.get_unpacked_types():
                    # Unpack the starred item if possible.
                    unpacked_types.extend(inner_types)
                else:
                    # Otherwise, fall back to a normal tuple type.
                    return PyInstanceType.from_stub("builtins.tuple")
            else:
                # For normal items, simply add them to the tuple.
                unpacked_types.append(item_type)

        return PyTupleType(item_types)


@final
class PyUnpack(PyType):
    """
    Wrapper type for unpacked types in star expressions.
    """

    def __init__(self, inner: PyType):
        self.inner = inner

    def __str__(self) -> str:
        return f"*{self.inner}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} inner={self.inner!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyUnpack) and self.inner == other.inner

    def get_unpacked_types(self) -> Optional[tuple["PyType", ...]]:
        """
        Returns the exact types of the unpacked items, if known.

        Currently, this only works for PyTupleType.
        """
        if isinstance(self.inner, PyTupleType):
            return self.inner.types

        return None


@final
class PyPackedTuple(PyTupleType):
    """
    Helper type for star targets in assignment. Represents an intermediate tuple type
    generated for a starred target. Will be converted to a list when actually assigned
    to the target.
    """

    def to_list_type(self) -> PyType:
        """
        Converts the packed tuple to a list type.
        """
        # TODO: Return a generic list of the union of all types once implemented.
        return PyInstanceType.from_stub("builtins.list")
