"""
This module defines the types used in the type system of the Python semantic analyzer.

References
- https://peps.python.org/pep-0483/
- https://peps.python.org/pep-0484/
- https://docs.python.org/3/library/typing.html
- https://docs.python.org/3/library/types.html
"""

import logging
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

from .base import SemanticError
from .scope import ScopeType, SymbolTable
from .symbol import Symbol

if TYPE_CHECKING:
    from .entity import PyClass, PyEntity, PyFunction, PyModule, PyParameters

logger = logging.getLogger(__name__)


class PyTypeError(SemanticError):
    pass


class _TypeContext(threading.local):
    def __init__(self):
        self.stubs: dict[str, SymbolTable] = {}
        self.forward_ref_evaluator: Callable[[str], "PyType"] = lambda _: PyType.ANY
        self.error_reporter: Optional[Callable[[PyTypeError], None]] = None


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


@contextmanager
def set_error_reporter(callback: Optional[Callable[[PyTypeError], None]]):
    old_reporter = _type_context.error_reporter
    _type_context.error_reporter = callback
    try:
        yield
    finally:
        _type_context.error_reporter = old_reporter


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

    logger.warning(f"Stub class {name} not found")

    if dummy:
        if name in _dummy_classes:
            return _dummy_classes[name]

        cls_name = name.rsplit(".", 1)[-1]
        dummy_scope = SymbolTable(cls_name, ScopeType.CLASS, full_name=name)
        dummy_cls = PyClass(cls_name, dummy_scope)
        _dummy_classes[name] = dummy_cls
        return dummy_cls

    return None


def get_stub_func(name: str) -> Optional["PyFunction"]:
    from .entity import PyFunction

    if isinstance(entity := get_stub_entity(name), PyFunction):
        return entity

    logger.warning(f"Stub function {name} not found")
    return None


def report_type_error(error: PyTypeError):
    if reporter := _type_context.error_reporter:
        reporter(error)


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
    NONE: "_PyNoneType"
    ELLIPSIS: "_PyEllipsisType"
    NEVER: "_PyNeverType"

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

    def can_be_truthy(self) -> bool:
        """
        Checks if the type can be evaluated as True in a boolean context.
        """
        return True

    def can_be_falsy(self) -> bool:
        """
        Checks if the type can be evaluated as False in a boolean context.
        """
        return True

    def extract_truthy(self) -> "PyType":
        """
        Returns the type of the truthy value of the type.
        """
        return self if self.can_be_truthy() else PyType.NEVER

    def extract_falsy(self) -> "PyType":
        """
        Returns the type of the falsy value of the type.
        """
        return self if self.can_be_falsy() else PyType.NEVER

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

    def get_conjunction_type(self, other: "PyType") -> "PyType":
        """
        Returns the type of the expression `self and other`, which is equivalent to
        `other if self else self`.
        """
        return PyUnionType.from_items(
            (
                self.extract_falsy(),
                other if self.can_be_truthy() else PyType.NEVER,
            )
        )

    def get_disjunction_type(self, other: "PyType") -> "PyType":
        """
        Returns the type of the expression `self or other`, which is equivalent to
        `self if self else other`.
        """
        return PyUnionType.from_items(
            (
                self.extract_truthy(),
                other if self.can_be_falsy() else PyType.NEVER,
            )
        )

    def get_inversion_type(self) -> "PyType":
        """
        Returns the type of the expression `not self`.
        """
        match self.can_be_truthy(), self.can_be_falsy():
            case True, True:
                return PyInstanceType.from_stub("builtins.bool")
            case True, False:
                return PyLiteralType(False)
            case False, True:
                return PyLiteralType(True)
            case False, False:
                return PyType.NEVER

    def check_protocol(self, protocol: "PyClass") -> Optional["PyTypeArgs"]:
        """
        Checks if the type conforms to the given protocol.

        Args:
            protocol: The protocol class to check against.

        Returns:
            type_args: The type arguments of the protocol if the type conforms to the
                protocol, or None otherwise. If the protocol has no type parameters, an
                empty tuple is returned.
        """
        return None

    def check_protocol_or_any(self, protocol: "PyClass") -> "PyTypeArgs":
        """ """
        if (type_args := self.check_protocol(protocol)) is not None:
            return type_args
        else:
            return protocol.default_type_args()

    def get_iterated_type(self, is_async: bool = False) -> "PyType":
        """
        Returns the type of the elements of the iterable.

        Args:
            is_async: Whether the iteration is asynchronous.
        """
        (item_type,) = self.check_protocol_or_any(
            get_stub_class(
                "typing.AsyncIterable" if is_async else "typing.Iterable", dummy=True
            )
        )
        return item_type

    def get_mapped_types(self) -> "PyKvPair":
        """
        Returns the type of the key-value pairs of the mapping.
        """
        key_type, value_type = self.check_protocol_or_any(
            get_stub_class("typing.Mapping", dummy=True)
        )
        return PyKvPair(key_type, value_type)

    def get_awaited_type(self) -> "PyType":
        """
        Returns the type of the awaited value.
        """
        (result_type,) = self.check_protocol_or_any(
            get_stub_class("typing.Awaitable", dummy=True)
        )
        return result_type

    def get_callable_type(self) -> Optional["PyFunctionType"]:
        """
        Returns the type of the callable, if the type is callable.
        """
        return None


@final
class _PyAnyType(PyType):
    def __str__(self) -> str:
        return "Any"

    def __eq__(self, other: object) -> bool:
        return self is other


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
                base_cls.scope,
                PyTypeTransform.chain(
                    get_base_substitutor(cls, base_cls, type_args),
                    BindMethod(bind_to="instance"),
                ),
            )

    def instance_attr_scopes(self) -> Iterator[TypedScope]:
        cls, type_args = self.get_cls(), self.get_type_args()

        for base_cls in cls.mro:
            yield TypedScope(
                base_cls.instance_scope, get_base_substitutor(cls, base_cls, type_args)
            )

    def lookup_method(self, name: str) -> Optional["PyFunctionType"]:
        for scope in self.class_attr_scopes():
            if attr := scope.get(name):
                if isinstance(attr.type, PyFunctionType):
                    return attr.type

        return None

    def get_method_return_type(self, name: str, args: "PyArguments") -> PyType:
        if method := self.lookup_method(name):
            return method.get_return_type(args)

        return PyType.ANY

    def can_be_truthy(self):
        # Special handling for bool types to avoid infinite recursion.
        if self.get_cls().full_name == "builtins.bool":
            return True

        if method := self.lookup_method("__bool__"):
            return method.get_return_type(PyArguments()).can_be_truthy()

        return True

    def can_be_falsy(self) -> bool:
        if self.get_cls().full_name == "builtins.bool":
            return True

        if method := self.lookup_method("__bool__"):
            return method.get_return_type(PyArguments()).can_be_falsy()

        if method := self.lookup_method("__len__"):
            return True

        return False

    def get_return_type(self, args: "PyArguments") -> PyType:
        return self.get_method_return_type("__call__", args)

    def get_subscripted_type(self, key: PyType) -> PyType:
        return self.get_method_return_type("__getitem__", PyArguments([key]))

    def check_protocol(self, protocol: "PyClass") -> Optional[PyTypeArgs]:
        assert protocol.has_modifier("protocol")

        # Check nominal subtyping.
        if protocol in self.get_cls().mro:
            type_args = get_base_type_args(
                self.get_cls(), protocol, self.get_type_args()
            )
            return type_args if type_args is not None else protocol.default_type_args()

        # TODO: Check structural subtyping.
        return None

    def get_callable_type(self) -> Optional["PyFunctionType"]:
        return self.lookup_method("__call__")

    def get_constructor_type(self) -> Optional["PyFunctionType"]:
        return self.lookup_method("__init__") or self.lookup_method("__new__")


class PyTypeVarDef(PyInstanceBase):
    """
    The type of a type variable definition, e.g. `TypeVar('T')`.
    """

    def __init__(self, var: PyTypeVar):
        self.var = var

    def __str__(self) -> str:
        return str(self.var)

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
            yield TypedScope(
                base_cls.scope,
                PyTypeTransform.chain(
                    get_base_substitutor(self.cls, base_cls),
                    BindMethod(bind_to="class"),
                ),
            )
        yield from super().attr_scopes()

    def get_return_type(self, args: "PyArguments") -> PyType:
        if self.cls.full_name == "typing.TypeVar":
            result = match_arguments_to_parameters(
                args, self.get_callable_type().get_parameters()
            )

            name = result.values.get("name")
            if name and isinstance(name, PyLiteralType) and type(name.value) is str:
                return PyTypeVarDef(PyTypeVar(name.value))
            else:
                report_type_error(PyTypeError("Invalid type variable definition"))
                return PyType.ANY

        return PyInstanceType(self.cls)

    def get_subscripted_type(self, key: PyType) -> PyType:
        args = key.types if isinstance(key, PyTupleType) else (key,)

        # In the form Literal[X], X is interpreted as a literal value, not a type
        # annotation. We need to handle this special case here.
        if self.cls.full_name != "typing.Literal":
            # Keep the ellipsis type as is.
            args = tuple(
                t if t is PyType.ELLIPSIS else t.get_annotated_type() for t in args
            )

        return PyGenericAlias(self.cls, args)

    def get_annotated_type(self) -> PyType:
        return self.cls.get_instance_type()

    @classmethod
    def from_stub(cls, name: str) -> PyType:
        if entity := get_stub_class(name):
            return cls(entity)
        return PyType.ANY

    def get_callable_type(self) -> Optional["PyFunctionType"]:
        return PyInstanceType(self.cls).get_constructor_type()


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

    def get_instance_type(
        self, convert_tuples: bool = True
    ) -> "PyInstanceType | PyTupleType":
        if self.cls.full_name == "builtins.tuple":
            if len(self.args) == 2 and self.args[1] is PyType.ELLIPSIS:
                # A homogeneous tuple type is represented as tuple[T, ...].
                return PyInstanceType(self.cls, (self.args[0],))
            else:
                # A tuple type with multiple items is represented as tuple[T1, ..., Tn].
                tuple_type = PyTupleType(self.args)
                return tuple_type if convert_tuples else tuple_type.get_fallback_type()

        return PyInstanceType(self.cls, self.args)

    def get_return_type(self, args: "PyArguments") -> PyType:
        return self.get_instance_type()

    def get_subscripted_type(self, key: PyType) -> PyType:
        # TODO: Generic alias like `list[T]` can be further subscripted.
        return PyType.ANY

    def get_annotated_type(self) -> PyType:
        # Check if the alias is a special form.
        if self.cls.full_name == "typing.Union":
            if self.args:
                return PyUnionType.from_items(self.args)
            else:
                report_type_error(
                    PyTypeError("Union type must have at least one argument")
                )
                return PyType.ANY

        elif self.cls.full_name == "typing.Optional":
            if len(self.args) == 1:
                return PyUnionType.optional(self.args[0])
            else:
                report_type_error(
                    PyTypeError("Optional type must have exactly one argument")
                )
                return PyType.ANY

        elif self.cls.full_name == "typing.Literal":
            if self.args and all(isinstance(x, PyLiteralType) for x in self.args):
                return PyUnionType.from_items(self.args)
            else:
                report_type_error(
                    PyTypeError("Literal type arguments must be literals")
                )
                return PyType.ANY

        return self.get_instance_type()

    @classmethod
    def from_stub(cls, name: str, args: PyTypeArgs) -> PyType:
        if entity := get_stub_class(name):
            return cls(entity, args)
        return PyType.ANY

    def get_callable_type(self) -> Optional["PyFunctionType"]:
        return self.get_instance_type().get_constructor_type()


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
            if self.cls.full_name == "builtins.tuple" and len(self.type_args) == 1:
                # A homogeneous tuple type is represented as tuple[T, ...].
                name += f"[{self.type_args[0]}, ...]"
            else:
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

    def extract_truthy(self) -> PyType:
        if self.cls.full_name == "builtins.bool":
            return PyLiteralType(True)

        return super().extract_truthy()

    def extract_falsy(self) -> PyType:
        match self.cls.full_name:
            case "builtins.bool":
                return PyLiteralType(False)
            case "builtins.int":
                return PyLiteralType(0)
            case "builtins.str":
                return PyLiteralType("")
            case "builtins.bytes":
                return PyLiteralType(b"")

        return super().extract_falsy()

    @classmethod
    def from_stub(cls, name: str, type_args: Optional[PyTypeArgs] = None) -> PyType:
        if entity := get_stub_class(name):
            return cls(entity, type_args)
        return PyType.ANY


@final
class PySelfType(PyInstanceType):
    def __init__(self, cls: "PyClass"):
        # The `self` type is generic if the class has type parameters.
        if cls.type_params:
            type_args = tuple(PyTypeVarType(t) for t in cls.type_params)
        else:
            type_args = None
        super().__init__(cls, type_args)

    def __str__(self) -> str:
        return f"Self@{super().__str__()}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PySelfType) and super().__eq__(other)

    def get_inferred_type(self) -> PyType:
        # The special handling of instance types only applies to the `self` parameter,
        # not to its aliases.
        return PyInstanceType(self.cls, self.type_args)


@final
class PyFunctionType(PyInstanceBase):
    def __init__(
        self,
        func: "PyFunction",
        mapping: Optional[PyTypeArgMap] = None,
        is_bound: bool = False,
    ):
        self.func = func
        self.mapping = mapping if mapping is not None else {}
        self.is_bound = is_bound

    def __str__(self) -> str:
        tag = f"bound {self.func.tag}" if self.is_bound else self.func.tag
        name = f"<{tag} {self.func.detailed_name!r}>"
        if self.mapping:
            name += f"[{', '.join(f'{k}={v}' for k, v in self.mapping.items())}]"
        return name

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} function={self.func!r} "
            "mapping={self.mapping!r} is_bound={self.is_bound}>"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PyFunctionType)
            and self.func is other.func
            and self.mapping == other.mapping
            and self.is_bound == other.is_bound
        )

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

    def get_parameters(self) -> "PyParameters":
        visitor = SubstituteTypeVars(self.mapping)
        parameters = self.func.parameters.transform(visitor)
        if self.is_bound:
            parameters.remove_bound_param()
        return parameters

    @staticmethod
    def from_stub(name: str) -> PyType:
        if func := get_stub_func(name):
            return PyFunctionType(func)
        return PyType.ANY

    def get_callable_type(self) -> "PyFunctionType":
        return self


@final
class _PyNoneType(PyInstanceBase):
    def __str__(self) -> str:
        return "None"

    def __eq__(self, other: object) -> bool:
        return self is other

    def get_cls(self) -> "PyClass":
        return get_stub_class("types.NoneType", dummy=True)

    def can_be_truthy(self) -> Literal[False]:
        return False

    def can_be_falsy(self) -> Literal[True]:
        return True

    def get_annotated_type(self) -> PyType:
        # None stands its own type in type annotations.
        return self


PyType.NONE = _PyNoneType()


@final
class _PyEllipsisType(PyInstanceBase):
    def __str__(self) -> str:
        return "EllipsisType"

    def __eq__(self, other: object) -> bool:
        return self is other

    def get_cls(self) -> "PyClass":
        return get_stub_class("types.EllipsisType", dummy=True)

    def can_be_truthy(self) -> Literal[True]:
        return True

    def can_be_falsy(self) -> Literal[False]:
        return False


PyType.ELLIPSIS = _PyEllipsisType()


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

    def can_be_truthy(self) -> bool:
        return bool(self.value)

    def can_be_falsy(self) -> bool:
        return not self.value

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
    """
    A tuple type in the form of `tuple[T1, ..., Tn]`. Note that the homogeneous tuple
    type `tuple[T, ...]` is handled by PyInstanceType with type argument (T,).
    """

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

    def get_type_args(self) -> Optional[PyTypeArgs]:
        # tuple[T1, ..., Tn] -> tuple[T1 | ... | Tn, ...]
        return (self.get_item_type(),)

    def can_be_truthy(self) -> bool:
        return bool(self.types)

    def can_be_falsy(self) -> bool:
        return not self.types

    def get_subscripted_type(self, key: PyType) -> PyType:
        if isinstance(key, PyLiteralType) and type(key.value) is int:
            index = key.value
            if 0 <= index < len(self.types):
                return self.types[index]

        if isinstance(key, PyInstanceType) and key.cls.full_name == "builtins.int":
            return self.get_item_type()

        # TODO: Handle slices.

        return super().get_subscripted_type(key)

    def get_inferred_type(self) -> PyType:
        return PyTupleType(tuple(x.get_inferred_type() for x in self.types))

    def get_item_type(self) -> PyType:
        return PyUnionType.from_items(self.types)

    def get_fallback_type(self) -> PyInstanceType:
        return PyInstanceType(self.get_cls(), self.get_type_args())

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
                    # Fully unpack the starred item if possible.
                    unpacked_types.extend(inner_types)
                else:
                    # Otherwise, use the common type of the unpacked items.
                    unpacked_type = item_type.get_unpacked_type()
                    return PyInstanceType.from_stub("builtins.tuple", (unpacked_type,))
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

    def get_unpacked_type(self) -> PyType:
        """
        Returns the type of any single unpacked item.
        """
        if isinstance(self.inner, PyTupleType):
            return self.inner.get_item_type()

        # Check if the inner type is an iterable.
        return self.inner.get_iterated_type()


@final
class PyUnpackKv(PyType):
    """
    Wrapper type for unpacked key-value pairs in star expressions.
    """

    def __init__(self, inner: PyType):
        self.inner = inner

    def __str__(self) -> str:
        return "**"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} inner={self.inner!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyUnpackKv) and self.inner == other.inner

    def get_unpacked_kvpair(self) -> "PyKvPair":
        """
        Returns the key-value pair of the unpacked item.
        """
        # The inner type should be a Mapping type.
        return self.inner.get_mapped_types()


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
        # PackedTuple[T1, ..., Tn] -> list[T1 | ... | Tn]
        return PyInstanceType.from_stub("builtins.list", (self.get_item_type(),))


@final
class PyUnionType(PyType):
    """
    A union type in the form of `Union[T1, T2, ...]`.
    """

    def __init__(self, items: tuple[PyType, ...]):
        """
        Creates a union type from the given types.

        In most cases, do not use this constructor directly. Instead, use `from_items`.
        """
        self.items = items

    def __str__(self) -> str:
        return " | ".join(map(str, self.items))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} items={self.items!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PyUnionType) and self.items == other.items

    def attr_scopes(self) -> Iterator[TypedScope]:
        # TODO: Handle conflicts between items.
        for item in self.items:
            yield from item.attr_scopes()

    def get_attr(self, name: str) -> Optional[TypedSymbol]:
        # TODO: Currently, only the first matching attribute is returned.
        for item in self.items:
            if attr := item.get_attr(name):
                return attr

        return None

    def can_be_truthy(self) -> bool:
        return any(t.can_be_truthy() for t in self.items)

    def can_be_falsy(self) -> bool:
        return any(t.can_be_falsy() for t in self.items)

    def extract_truthy(self) -> PyType:
        return PyUnionType.from_items(t.extract_truthy() for t in self.items)

    def extract_falsy(self) -> PyType:
        return PyUnionType.from_items(t.extract_falsy() for t in self.items)

    def get_return_type(self, args: "PyArguments") -> PyType:
        return PyUnionType.from_items(t.get_return_type(args) for t in self.items)

    def get_subscripted_type(self, key: PyType) -> PyType:
        return PyUnionType.from_items(t.get_subscripted_type(key) for t in self.items)

    def get_inferred_type(self) -> PyType:
        return PyUnionType.from_items(t.get_inferred_type() for t in self.items)

    def check_protocol(self, protocol: "PyClass") -> Optional[PyTypeArgs]:
        # TODO: Currently, only the first matching type is returned.
        for item in self.items:
            if (type_args := item.check_protocol(protocol)) is not None:
                return type_args

        return None

    def get_callable_type(self) -> Optional["PyFunctionType"]:
        # TODO: Currently, only the first matching type is returned.
        for item in self.items:
            if callable_type := item.get_callable_type():
                return callable_type

        return None

    @staticmethod
    def from_items(items: Iterable[PyType]) -> PyType:
        """
        Creates a union type from the given types. This method is used to handle
        duplicate types and nested unions. Also, it returns a single type if the union
        contains only one type.

        Args:
            items: The types to include in the union.

        Returns:
            The union type.
        """
        unique_items: list[PyType] = []

        for item in items:
            if isinstance(item, PyUnionType):
                unique_items.extend(item.items)
            elif item not in unique_items:
                unique_items.append(item)

        if PyType.NEVER in unique_items:
            unique_items.remove(PyType.NEVER)

        if not unique_items:
            return PyType.NEVER

        if len(unique_items) == 1:
            return unique_items[0]

        return PyUnionType(tuple(unique_items))

    @staticmethod
    def optional(item: PyType) -> PyType:
        """
        Creates an optional type Optional[T] from the given type T.

        Args:
            item: The type to make optional.

        Returns:
            The optional type.
        """
        return PyUnionType.from_items((item, PyType.NONE))


class _PyNeverType(PyType):
    """
    A type that contains no values. This type is used to represent the result of
    functions that never return, such as `sys.exit()`, or to indicate unreachable code.
    """

    def __str__(self) -> str:
        return "Never"

    def __eq__(self, other: object) -> bool:
        return self is other

    def can_be_truthy(self) -> Literal[False]:
        return False

    def can_be_falsy(self) -> Literal[False]:
        return False

    def get_return_type(self, args: "PyArguments") -> PyType:
        return PyType.NEVER

    def get_subscripted_type(self, key: PyType) -> PyType:
        return PyType.NEVER


PyType.NEVER = _PyNeverType()


class PyKeywordArgument(NamedTuple):
    name: str
    type: PyType


class PyArguments(list[PyType | PyKeywordArgument]):
    """
    A list of arguments in a function call or definition.

    The list can contain positional arguments, keyword arguments, and unpacked items or
    key-value pairs.
    """

    def get_positionals(self) -> list[PyType]:
        return [
            x for x in self if isinstance(x, PyType) and not isinstance(x, PyUnpackKv)
        ]

    def get_keywords(self) -> list[PyKeywordArgument]:
        return [x for x in self if isinstance(x, PyKeywordArgument)]


class ArgumentMatch(NamedTuple):
    """
    Stores the result of matching arguments to parameters in a function call.
    """

    matched: dict[int, int]
    """The mapping of argument indices to parameter indices."""
    values: dict[str, PyType]
    """The mapping of keyword argument names to their types."""
    next_positional: int
    """The index of the next positional parameter to match, or -1 if no more."""


def match_arguments_to_parameters(
    arguments: PyArguments, parameters: "PyParameters"
) -> ArgumentMatch:
    """
    Matches the arguments against the given parameters.

    Args:
        arguments: The arguments to match.
        parameters: The parameters to match against.

    Returns:
        result: The result of the matching process.
    """
    pos_params = parameters.get_positionals()
    star_args: list[PyType] = []

    matched: dict[int, int] = {}
    values: dict[str, PyType] = {}
    next_positional = 0 if pos_params else -1

    for i, arg in enumerate(arguments):
        if isinstance(arg, PyKeywordArgument):
            # Keyword arguments are matched by name.
            if param := parameters.get_keyword(arg.name):
                matched[i] = parameters.index(param)
                if param.star is None:
                    values[param.name] = arg.type

        elif isinstance(arg, PyUnpackKv):
            # Cannot match unpacked key-value pairs.
            pass

        else:
            # Positional arguments are matched by position.
            if next_positional == -1:
                continue

            if isinstance(arg, PyUnpack):
                # Unpack the starred item if possible.
                unpacked = arg.get_unpacked_types()
                if unpacked is None:
                    next_positional = -1
                    continue
            else:
                unpacked = (arg,)

            matched[i] = next_positional

            for item in unpacked:
                param = pos_params[next_positional]
                if param.star == "*":
                    star_args.append(item)
                else:
                    values[param.name] = item
                    next_positional += 1
                    if next_positional >= len(pos_params):
                        next_positional = -1
                        break

    if star_param := parameters.get_starred():
        values[star_param.name] = PyTupleType.from_starred(star_args)

    return ArgumentMatch(matched, values, next_positional)


class PyKvPair(NamedTuple):
    key: PyType
    value: PyType


PyDictDisplayItem = PyUnpackKv | PyKvPair


def infer_list_display(item_types: Iterable[PyType]) -> Optional[PyTypeArgs]:
    """
    Infers the type arguments of a list or set display from the types of its items,
    possibly with unpacked items. For example, [1, *(2, 3)] -> (int,).

    Args:
        item_types: The types of the items in the display.

    Returns:
        type_args: The inferred type arguments, or None if the element type cannot be
            inferred.
    """
    unpacked_types: list[PyType] = []

    for item_type in item_types:
        if isinstance(item_type, PyUnpack):
            # Unpack the starred item if possible.
            unpacked_types.append(item_type.get_unpacked_type())
        else:
            # For normal items, simply add them to the tuple.
            unpacked_types.append(item_type)

    # If there are no items, the element type cannot be inferred.
    if not unpacked_types:
        return None

    return (PyUnionType.from_items(t.get_inferred_type() for t in unpacked_types),)


def infer_dict_display(items: Iterable[PyDictDisplayItem]) -> Optional[PyTypeArgs]:
    """
    Similar to `infer_type_args_from_display`, but for dictionary displays.
    """
    key_types: list[PyType] = []
    value_types: list[PyType] = []

    for item in items:
        if isinstance(item, PyUnpackKv):
            # The unpacked item should be a Mapping type.
            key_type, value_type = item.get_unpacked_kvpair()
            key_types.append(key_type)
            value_types.append(value_type)
        else:
            key_types.append(item.key)
            value_types.append(item.value)

    if not key_types:
        return None

    return (
        PyUnionType.from_items(t.get_inferred_type() for t in key_types),
        PyUnionType.from_items(t.get_inferred_type() for t in value_types),
    )


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

        if isinstance(type, PyUnionType):
            return self.visit_type_args(type.items)

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

    def visit_maybe_type(self, type: Optional[PyType]) -> Optional[PyType]:
        return self.visit_type(type) if type is not None else None

    @staticmethod
    def chain(*transforms: "PyTypeTransform") -> "PyTypeTransform":
        """
        Chains multiple type transforms together.
        """
        return ChainedTypeTransform(*transforms)


class DummyTypeTransform(PyTypeTransform):
    """
    A dummy type transform that returns the input type as is.
    """

    def visit_type(self, type: PyType) -> PyType:
        return type


class ChainedTypeTransform(PyTypeTransform):
    """
    A type transform that chains multiple type transforms.
    """

    def __init__(self, *transforms: PyTypeTransform):
        self.transforms = transforms

    def visit_type(self, type: PyType) -> PyType:
        for transform in self.transforms:
            type = transform.visit_type(type)

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
            return PyFunctionType(type.func, self.mapping, type.is_bound)

        if isinstance(type, PyTupleType):
            types = tuple(self.visit_type(t) for t in type.types)
            return PyTupleType(types)

        if isinstance(type, PyUnionType):
            items = tuple(self.visit_type(t) for t in type.items)
            return PyUnionType.from_items(items)

        return type


class BindMethod(PyTypeTransform):
    """
    A visitor that marks methods as bound methods.
    """

    def __init__(self, bind_to: Literal["instance", "class"] = "instance"):
        self.bound_to = bind_to

    def _should_bind(self, func: "PyFunction") -> bool:
        if func.has_modifier("staticmethod"):
            # Static methods are not bound.
            return False

        if func.has_modifier("classmethod"):
            # Class methods are always bound.
            return True

        # Bound methods are only created for instances.
        return self.bound_to == "instance"

    def visit_type(self, type: PyType) -> PyType:
        if isinstance(type, PyFunctionType) and self._should_bind(type.func):
            return PyFunctionType(type.func, type.mapping, is_bound=True)

        return type
