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
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    final,
    overload,
)

from .scope import ScopeType, SymbolTable
from .symbol import Symbol, SymbolType

if TYPE_CHECKING:
    from .entity import PyClass, PyEntity, PyFunction, PyModule


class _TypeContext(threading.local):
    def __init__(self):
        self.stubs: dict[str, SymbolTable] = {}
        self.forward_ref_evaluator: Callable[[str], "PyType"] = lambda _: PyType.ANY


_type_context = _TypeContext()


@contextmanager
def set_type_context(stubs: dict[str, SymbolTable]):
    old_stubs = _type_context.stubs
    _type_context.stubs = stubs
    try:
        yield
    finally:
        _type_context.stubs = old_stubs


@contextmanager
def set_forward_ref_evaluator(callback: Callable[[str], "PyType"]):
    old_evaluator = _type_context.forward_ref_evaluator
    _type_context.forward_ref_evaluator = callback
    try:
        yield
    finally:
        _type_context.forward_ref_evaluator = old_evaluator


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


class TypedScope(NamedTuple):
    """
    A typed version of a symbol table.

    Attributes:
        scope: The symbol table.
        transform: The type transform to apply to the types of the symbols in the scope.
            This is used to handle type variables in the presence of generic types.
    """

    scope: SymbolTable
    transform: "PyTypeTransform"

    def get(self, name: str) -> Optional["TypedSymbol"]:
        if symbol := self.scope.get(name):
            return TypedSymbol(symbol, self.transform.visit_type(symbol.get_type()))
        return None

    def iter_symbols(self) -> Iterator["TypedSymbol"]:
        for symbol in self.scope.iter_symbols():
            yield TypedSymbol(symbol, self.transform.visit_type(symbol.get_type()))


class TypedSymbol(NamedTuple):
    """
    A symbol with its type.

    Attributes:
        symbol: The symbol.
        type: The type of the symbol. The type may not equal `symbol.get_type()` in the
            presence of type variables.
    """

    symbol: Symbol
    type: "PyType"


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

    def attr_scopes(self) -> Iterator[TypedScope]:
        """
        Yields the attribute scopes of the type in order of precedence.

        The default implementation of `get_attr` and `iter_attrs` uses this method to
        lookup and iterate over the attributes of the type.
        """
        return iter(())

    def get_attr(self, name: str) -> Optional[TypedSymbol]:
        """
        Finds an attribute of the type by name.
        """
        for scope in self.attr_scopes():
            if attr := scope.get(name):
                return attr

        return None

    def iter_attrs(self) -> Iterator[TypedSymbol]:
        """
        Yields the attributes of the type, removing duplicates.
        """
        visited: set[str] = set()
        for attr in self._iter_attrs():
            if attr.symbol.name not in visited:
                visited.add(attr.symbol.name)
                yield attr

    def _iter_attrs(self) -> Iterator[TypedSymbol]:
        """
        Yields the attributes of the type, possibly with duplicates.
        """
        for scope in self.attr_scopes():
            yield from scope.iter_symbols()

    def get_return_type(self, args: "PyArguments") -> "PyType":
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

    def get_annotated_type(self) -> "PyType":
        """
        Returns the type of the annotation.

        This is used for type annotations, e.g. determining the type of a variable from
        its annotation.
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


class PyTypeVar:
    def __init__(self, name: str, bound: Optional[PyType] = None):
        # TODO: variances
        self.name = name
        self.bound = bound

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} bound={self.bound!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyTypeVar) and self.name == other.name


PyTypeArgMap = dict[str, PyType]
PyTypeArgs = Sequence[PyType]


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

    def get_type_args(self) -> Optional[PyTypeArgs]:
        """
        Returns the type arguments of the instance.
        """
        return None

    def attr_scopes(self) -> Iterator[TypedScope]:
        yield from self.instance_attr_scopes()
        yield from self.class_attr_scopes()

    def class_attr_scopes(self) -> Iterator[TypedScope]:
        cls, type_args = self.get_cls(), self.get_type_args()

        for base_cls in cls.mro:
            yield TypedScope(
                base_cls.scope, get_base_substitutor(cls, base_cls, type_args)
            )

    def instance_attr_scopes(self) -> Iterator[TypedScope]:
        cls, type_args = self.get_cls(), self.get_type_args()

        for base_cls in cls.mro:
            yield TypedScope(
                base_cls.instance_scope, get_base_substitutor(cls, base_cls, type_args)
            )

    def lookup_method(self, name: str) -> Optional["PyFunctionType"]:
        for scope in self.attr_scopes():
            if attr := scope.get(name):
                if isinstance(attr.type, PyFunctionType):
                    return attr.type

        return None

    def get_method_return_type(self, name: str, args: "PyArguments") -> PyType:
        if method := self.lookup_method(name):
            return method.get_return_type(args)

        return PyType.ANY

    def get_return_type(self, args: "PyArguments") -> PyType:
        return self.get_method_return_type("__call__", args)

    def get_subscripted_type(self, key: PyType) -> PyType:
        return self.get_method_return_type("__getitem__", PyArguments(args=[key]))

    def get_awaited_type(self) -> PyType:
        return self.get_method_return_type("__await__", PyArguments())


class PyTypeVarDef(PyInstanceBase):
    """
    The type of a type variable definition, e.g. `TypeVar('T')`.
    """

    def __init__(self, var: PyTypeVar):
        self.var = var

    def __str__(self) -> str:
        return f"<tvar {self.var.name!r}>"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} var={self.var!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyTypeVarDef) and self.var == other.var

    def get_cls(self) -> "PyClass":
        return get_stub_class("typing.TypeVar", dummy=True)

    def get_annotated_type(self) -> PyType:
        return PyTypeVarType(self.var)


class PyTypeVarType(PyType):
    """
    The type of a type variable, e.g. `T` in `list[T]`.

    Currently, this type serves as a placeholder for type variables in the type system.
    """

    def __init__(self, var: PyTypeVar):
        self.var = var

    def __str__(self) -> str:
        return str(self.var)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} var={self.var!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyTypeVarType) and self.var == other.var


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

    def attr_scopes(self) -> Iterator[TypedScope]:
        yield TypedScope(self.module.context.global_scope, DummyTypeTransform())
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

    def attr_scopes(self) -> Iterator[TypedScope]:
        for base_cls in self.cls.mro:
            yield TypedScope(base_cls.scope, get_base_substitutor(self.cls, base_cls))
        yield from super().attr_scopes()

    def get_return_type(self, args: "PyArguments") -> PyType:
        if self.cls is get_stub_class("typing.TypeVar"):
            # TODO: Actually handle function calls.
            if (
                args.args
                and isinstance(arg := args.args[0], PyLiteralType)
                and type(arg.value) is str
            ):
                return PyTypeVarDef(PyTypeVar(arg.value))
            else:
                # TODO: Report an errornous call.
                return PyType.ANY

        return PyInstanceType(self.cls)

    def get_subscripted_type(self, key: PyType) -> PyType:
        if isinstance(key, PyTupleType):
            args = tuple(t.get_annotated_type() for t in key.types)
        else:
            args = (key.get_annotated_type(),)
        return PyGenericAlias(self.cls, args)

    def get_annotated_type(self) -> PyType:
        return self.cls.get_instance_type()

    @classmethod
    def from_stub(cls, name: str) -> PyType:
        if entity := get_stub_class(name):
            return cls(entity)
        return PyType.ANY


@final
class PyGenericAlias(PyInstanceBase):
    """
    The type of a generic alias, e.g. `list[int]`, `dict[str, T]`.
    """

    def __init__(self, cls: "PyClass", args: PyTypeArgs):
        # TODO: Check the number of type arguments.
        self.cls = cls
        self.args = tuple(args)

    def __str__(self) -> str:
        return f"{self.cls.name}[{', '.join(map(str, self.args))}]"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} cls={self.cls!r} args={self.args!r}>"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PyGenericAlias)
            and self.cls is other.cls
            and self.args == other.args
        )

    def get_cls(self) -> "PyClass":
        return get_stub_class("types.GenericAlias", dummy=True)

    def get_instance_type(self) -> "PyInstanceType":
        # TODO: Handle special forms.
        return PyInstanceType(self.cls, self.args)

    def get_return_type(self, args: "PyArguments") -> PyType:
        return self.get_instance_type()

    def get_subscripted_type(self, key: PyType) -> PyType:
        # TODO: Generic alias like `list[T]` can be further subscripted.
        return PyType.ANY

    def get_annotated_type(self) -> PyType:
        return self.get_instance_type()

    @classmethod
    def from_stub(cls, name: str, args: PyTypeArgs) -> PyType:
        if entity := get_stub_class(name):
            return cls(entity, args)
        return PyType.ANY


class PyInstanceType(PyInstanceBase):
    """
    An instance type in the form of `C[T1, T2, ...]`. If the type arguments are not
    provided, they are assumed to be `Any` for each type parameter of the class.

    The type parameters can be concrete types or contain type variables.
    """

    def __init__(self, cls: "PyClass", type_args: Optional[PyTypeArgs] = None):
        self.cls = cls
        self.type_args = tuple(type_args) if type_args is not None else None

    def __str__(self) -> str:
        name = self.cls.name
        if self.type_args:
            name += f"[{', '.join(map(str, self.type_args))}]"
        return name

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} class={self.cls!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyInstanceType) and self.cls is other.cls

    def get_cls(self) -> "PyClass":
        return self.cls

    def get_type_args(self) -> Optional[PyTypeArgs]:
        return self.type_args

    @classmethod
    def from_stub(cls, name: str, type_args: Optional[PyTypeArgs] = None) -> PyType:
        if entity := get_stub_class(name):
            return cls(entity, type_args)
        return PyType.ANY


@final
class PySelfType(PyInstanceType):
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
    def __init__(self, func: "PyFunction", mapping: Optional[PyTypeArgMap] = None):
        self.func = func
        self.mapping = mapping if mapping is not None else {}

    def __str__(self) -> str:
        name = str(self.func)
        if self.mapping:
            name += f"[{', '.join(f'{k}={v}' for k, v in self.mapping.items())}]"
        return name

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} function={self.func!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyFunctionType) and self.func is other.func

    @property
    def entity(self) -> "PyEntity":
        return self.func

    def get_cls(self) -> "PyClass":
        return get_stub_class("types.FunctionType", dummy=True)

    def get_return_type(self, args: "PyArguments") -> PyType:
        visitor = SubstituteTypeVars(self.mapping)
        return visitor.visit_type(self.func.return_type or PyType.ANY)

    def get_inferred_type(self) -> PyType:
        # TODO: Infer the callable type from the function.
        return self

    def get_parameter(self, name: str) -> Optional[TypedSymbol]:
        if symbol := self.func.scope.get(name):
            if symbol.type == SymbolType.PARAMETER:
                visitor = SubstituteTypeVars(self.mapping)
                return TypedSymbol(symbol, visitor.visit_type(symbol.get_type()))

        return None

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

    def get_annotated_type(self) -> PyType:
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

    def get_annotated_type(self) -> PyType:
        # String literals can represent forward references.
        if type(self.value) is str:
            evaluated_type = _type_context.forward_ref_evaluator(self.value)
            return evaluated_type.get_annotated_type()

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

    def get_type_args(self) -> Optional[Iterable[PyType]]:
        # TODO: Return a union of all enclosed types.
        if self.types and all(t == self.types[0] for t in self.types):
            return (self.types[0],)
        return None

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


class PyKeywordArgument(NamedTuple):
    name: str
    type: PyType


class PyArguments:
    def __init__(
        self,
        *,
        args: Optional[list[PyType]] = None,
        kwargs: Optional[list[PyKeywordArgument]] = None,
        double_stars: Optional[list[PyType]] = None,
    ):
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else []
        self.double_stars = double_stars if double_stars is not None else []


def get_type_arg_map(
    type_params: Sequence[PyTypeVar], type_args: Optional[PyTypeArgs]
) -> PyTypeArgMap:
    """
    Creates a mapping of type variable names to types from the given type arguments.

    Args:
        type_params: The type parameters of the class or function.
        type_args: The type arguments provided for the type parameters.

    Returns:
        mapping: The mapping of type variable names to types.
    """
    if type_args is None or len(type_params) != len(type_args):
        return {}

    return {param.name: arg for param, arg in zip(type_params, type_args)}


def get_base_type_args(
    cls: "PyClass", base_cls: "PyClass", type_args: Optional[PyTypeArgs] = None
) -> Optional[PyTypeArgs]:
    """
    Calculates the type arguments of a base class in the context of a derived class.

    Args:
        cls: The derived class.
        base_cls: The base class.
        type_args: The type arguments of the derived class.
    """
    if base_cls is cls:
        return type_args

    mapping = get_type_arg_map(cls.type_params, type_args)

    for base in cls.bases:
        # Find the first parent class that is a subclass of the base class.
        if base_cls in base.cls.mro:
            visitor = SubstituteTypeVars(mapping)
            type_args = visitor.visit_type_args(base.get_type_args())
            return get_base_type_args(base.cls, base_cls, type_args)

    raise ValueError(f"Base class {base_cls.name} not found in the MRO of {cls.name}")


def get_base_substitutor(
    cls: "PyClass", base_cls: "PyClass", type_args: Optional[PyTypeArgs] = None
) -> "SubstituteTypeVars":
    """
    Creates a type substitutor for the given base class in the context of the derived class.

    Args:
        cls: The derived class.
        base_cls: The base class.
        type_args: The type arguments of the derived class.
    """
    if not base_cls.type_params:
        # Fast path for classes without type parameters.
        return DummyTypeTransform()

    type_args = get_base_type_args(cls, base_cls, type_args)
    mapping = get_type_arg_map(base_cls.type_params, type_args)
    return SubstituteTypeVars(mapping)


class CollectTypeVars:
    """
    A visitor that collects type variables from types.
    """

    def __init__(self):
        self.type_vars: list[PyTypeVar] = []
        self.seen_names: set[str] = set()

    def visit_type(self, type: PyType) -> None:
        if isinstance(type, PyTypeVarType):
            if type.var.name not in self.seen_names:
                self.type_vars.append(type.var)
                self.seen_names.add(type.var.name)
                return

        if isinstance(type, PyInstanceType):
            return self.visit_type_args(type.type_args)

        if isinstance(type, PyGenericAlias):
            return self.visit_type_args(type.args)

        if isinstance(type, PyTupleType):
            return self.visit_type_args(type.types)

    def visit_type_args(self, type_args: Optional[PyTypeArgs]) -> None:
        if type_args is None:
            return

        for type in type_args:
            self.visit_type(type)


class PyTypeTransform(ABC):
    """
    A visitor that transforms types.
    """

    @abstractmethod
    def visit_type(self, type: PyType) -> PyType: ...

    def visit_type_args(self, type_args: Optional[PyTypeArgs]) -> Optional[PyTypeArgs]:
        if type_args is None:
            return None

        return tuple(self.visit_type(t) for t in type_args)


class DummyTypeTransform(PyTypeTransform):
    """
    A dummy type transform that returns the input type as is.
    """

    def visit_type(self, type: PyType) -> PyType:
        return type


class SubstituteTypeVars(PyTypeTransform):
    """
    A visitor that substitutes type variables in types with the corresponding types
    from the given mapping.
    """

    def __init__(self, mapping: PyTypeArgMap, keep_unknown: bool = False):
        """
        Args:
            mapping: The mapping of type variable names to types.
            keep_unknown: Whether to keep unknown type variables as is. If False, they
                are replaced with `Any`.
        """
        self.mapping = mapping
        self.keep_unknown = keep_unknown

    def visit_type(self, type: PyType) -> PyType:
        if isinstance(type, PyTypeVarType):
            if type.var.name in self.mapping:
                return self.mapping[type.var.name]

            return type if self.keep_unknown else PyType.ANY

        if isinstance(type, PyInstanceType):
            type_args = self.visit_type_args(type.type_args)
            return PyInstanceType(type.cls, type_args)

        if isinstance(type, PyGenericAlias):
            args = self.visit_type_args(type.args)
            return PyGenericAlias(type.cls, args)

        if isinstance(type, PyFunctionType):
            return PyFunctionType(type.func, self.mapping)

        if isinstance(type, PyTupleType):
            types = tuple(self.visit_type(t) for t in type.types)
            return PyTupleType(types)

        return type
