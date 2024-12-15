from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Literal, Optional

from .base import get_node_source
from .scope import ScopeType, SymbolTable
from .types import (
    CollectTypeVars,
    PyArguments,
    PyClassType,
    PyFunctionType,
    PyGenericAlias,
    PyInstanceType,
    PyKeywordArgument,
    PyModuleType,
    PySelfType,
    PyType,
    PyTypeArgs,
    PyTypeError,
    PyTypeTransform,
    PyTypeVar,
    PyTypeVarType,
    get_stub_class,
    report_type_error,
)

if TYPE_CHECKING:
    from core.source import PythonSource
    from grammar import PythonParser

    from .structure import PythonContext


class PyEntity(ABC):
    """
    A Python entity.
    """

    def __init__(self, name: str, full_name: Optional[str] = None):
        self.name = name
        self._full_name = full_name

    @property
    def full_name(self) -> str:
        """
        Returns the fully-qualified name of the entity.
        """
        return self._full_name or self.name

    def set_full_name(self, full_name: str):
        if self._full_name is None:
            self._full_name = full_name

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the entity.
        """
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name!r}>"

    @abstractmethod
    def get_type(self) -> PyType: ...


class PyModule(PyEntity):
    """
    A Python module.
    """

    def __init__(self, name: str, path: Optional[Path]):
        """
        Args:
            name: The fully qualified module name.
            path: The path to the module file.
        """
        super().__init__(name, full_name=name)
        self.path = path

        self.loader: Optional[Callable[[], PythonSource]] = None
        self.source: Optional[PythonSource] = None
        self.context: Optional[PythonContext] = None
        self.loaded = False

    def __str__(self):
        return f"<module {self.name!r}>"

    def __repr__(self):
        s = f"<{self.__class__.__name__} {self.name!r}"
        if self.path is not None:
            s += f" from {self.path}"
        return s + ">"

    @property
    def package(self) -> str:
        return ".".join(self.name.split(".")[:-1])

    def get_type(self) -> PyModuleType:
        return PyModuleType(self)


class PyPackage(PyModule):
    """
    A Python package.
    """

    def __init__(self, name: str, path: Optional[Path], import_paths: list[Path]):
        super().__init__(name, path)
        self.import_paths = import_paths


class _ModifiersMixin:
    def init_modifiers(self):
        self.modifiers: set[str] = set()

    def has_modifier(self, *modifiers: str) -> bool:
        """
        Checks if the entity has all the specified modifiers.
        """
        return all(modifier in self.modifiers for modifier in modifiers)

    def set_modifier(self, *modifiers: str):
        """
        Sets the specified modifiers on the entity.
        """
        self.modifiers.update(modifiers)


class PyClass(_ModifiersMixin, PyEntity):
    def __init__(self, name: str, scope: "SymbolTable"):
        super().__init__(name)
        self.scope = scope
        self.instance_scope = SymbolTable(
            "<members>", ScopeType.OBJECT, full_name=f"{scope.full_name}.<members>"
        )
        self.decorators: list[PyType] = []
        self.arguments: Optional[PyArguments] = None
        self.bases: list[PyInstanceType] = []
        self.type_params: list[PyTypeVar] = []

        self._mro: Optional[list[PyClass]] = None
        self._computing_mro = False

        self.init_modifiers()

    def __str__(self) -> str:
        return f"<class {self.name!r}>"

    def get_type(self) -> PyClassType:
        return PyClassType(self)

    def get_instance_type(self) -> PyType:
        if self.full_name == "types.NoneType":
            return PyType.NONE
        if self.full_name == "types.EllipsisType":
            return PyType.ELLIPSIS
        return PyInstanceType(self)

    def get_self_type(self) -> PySelfType:
        return PySelfType(self)

    def default_type_args(self) -> PyTypeArgs:
        return tuple(PyType.ANY for _ in self.type_params)

    def mro_scopes(self, instance: bool = False) -> Iterator[SymbolTable]:
        """
        Yields the scopes of the class and its bases in MRO order.

        Args:
            instance: Whether to yield the instance scopes of the classes.

        Yields:
            scope: The scope of a class or its instance.
        """
        if instance:
            for cls in self.mro:
                yield cls.instance_scope

        for cls in self.mro:
            yield cls.scope

    def lookup_method(self, name: str) -> Optional["PyFunction"]:
        """
        Looks up a method in the class and its bases.
        """
        for scope in self.mro_scopes():
            if symbol := scope.get(name):
                if isinstance(entity := symbol.resolve_entity(), PyFunction):
                    return entity

        return None

    def get_method_return_type(self, name: str) -> PyType:
        if method := self.lookup_method(name):
            return method.return_type or PyType.ANY
        return PyType.ANY

    @property
    def mro(self) -> list["PyClass"]:
        if self._mro is None:
            if self.arguments is None:
                # The class is not fully defined yet. Return a placeholder.
                return [self]

            if self._computing_mro:
                report_type_error(
                    PyTypeError(
                        f"Recursive inheritance detected (in class {self.full_name!r})"
                    )
                )
                return [self]

            self._computing_mro = True
            try:
                self._mro = self.compute_mro()
            except PyTypeError as e:
                report_type_error(e)
                self._mro = [self]
            finally:
                self._computing_mro = False

        return self._mro

    def compute_mro(self) -> list["PyClass"]:
        """
        Computes the method resolution order for the class according to the C3
        linearization algorithm.

        References:
        - https://docs.python.org/3/howto/mro.html
        """
        # Collect the MROs of the base classes.
        mro_lists: list[list[PyClass]] = []
        for base in self.bases:
            mro_lists.append(base.cls.mro.copy())

        # Add the base classes to preserve the orderings.
        if self.bases:
            mro_lists.append([base.cls for base in self.bases])

        # The MRO always starts with the class itself.
        result: list[PyClass] = [self]

        while mro_lists:
            # Find the first head element that is not in the tail of any other list.
            for mro_list in mro_lists:
                head = mro_list[0]
                if any(head in l[1:] for l in mro_lists):
                    continue
                break
            else:
                raise PyTypeError(
                    f"Cannot create a consistent MRO (in class {self.full_name!r})"
                )

            # Add the head element to the MRO.
            result.append(head)

            # Remove the head element from the lists.
            for mro_list in mro_lists:
                if mro_list[0] is head:
                    del mro_list[0]

            # Remove the lists that are now empty.
            mro_lists = [l for l in mro_lists if l]

        # All classes have `object` as the last base class.
        if object_cls := get_stub_class("builtins.object"):
            if result[-1] is not object_cls:
                result.append(object_cls)

        return result

    def parse_arguments(self, ctx: "PythonParser.ArgumentsContext"):
        """
        Parses the base classes and the type parameters of the class from the
        arguments of the class definition.
        """
        assert self.arguments is not None
        assert not self.bases and not self.type_params

        # Whether the class has a generic base class.
        has_generic_base = False

        # NOTE: In theory, we should handle starred arguments here, but in
        # practice, syntaxes like `class C(*args):` are hardly ever used.
        for arg in self.arguments:
            if isinstance(arg, PyClassType):
                self.bases.append(PyInstanceType(arg.cls))
            elif isinstance(arg, PyGenericAlias):
                # TODO: Verify that the generic alias is a valid base class.
                has_generic_base = True
                self.bases.append(arg.get_instance_type(convert_tuples=False))
            elif isinstance(arg, PyKeywordArgument):
                # Meta-classes are not supported.
                pass
            else:
                report_type_error(
                    PyTypeError(
                        f"invalid base class {arg} in class definition",
                    ).with_context(ctx)
                )

        # Check if the class has typing.Generic as a base class.
        if any(
            (generic_base := base).cls.full_name == "typing.Generic"
            for base in self.bases
        ):
            # Collect the type variable arguments as type parameters.
            for arg_type in generic_base.type_args:
                if isinstance(arg_type, PyTypeVarType):
                    if arg_type.var not in self.type_params:
                        self.type_params.append(arg_type.var)
                    else:
                        report_type_error(
                            PyTypeError(
                                f"duplicate type variable argument {arg_type} to Generic "
                                f"(in class {self.full_name!r})",
                            ).with_context(ctx)
                        )
                else:
                    report_type_error(
                        PyTypeError(
                            f"invalid type argument {arg_type} to Generic "
                            f"(in class {self.full_name!r})",
                        ).with_context(ctx)
                    )

        elif has_generic_base:
            # Collect the type variables from the base classes.
            visitor = CollectTypeVars()
            for base in self.bases:
                if base.type_args is not None:
                    visitor.visit_type_args(base.type_args)

            self.type_params.extend(visitor.type_vars)

            # Add typing.Generic as an implicit base class.
            type_args = tuple(PyTypeVarType(var) for var in visitor.type_vars)
            generic_type = PyInstanceType.from_stub("typing.Generic", type_args)
            self.bases.append(generic_type)

        # Check if the class is a protocol.
        if any(base.cls.full_name == "typing.Protocol" for base in self.bases):
            self.set_modifier("protocol")

            # Verify that all base classes are protocols.
            for base in self.bases:
                if base.cls.full_name not in (
                    "typing.Protocol",
                    "typing.Generic",
                    "builtins.object",
                ) and not base.cls.has_modifier("protocol"):
                    report_type_error(
                        PyTypeError(
                            f"non-protocol base class {base.cls.full_name} in protocol "
                            f"(in class {self.full_name!r})",
                        ).with_context(ctx)
                    )


class PyFunction(_ModifiersMixin, PyEntity):
    def __init__(self, name: str, scope: "SymbolTable", cls: Optional[PyClass] = None):
        super().__init__(name)
        self.scope = scope
        self.cls = cls
        self.parameters = PyParameters()
        self.return_type: Optional[PyType] = None
        self.decorators: list[PyType] = []

        self.init_modifiers()

        # Overloaded functions.
        self.overloads: list[PyFunction] = []

        # The types returned by a return statement. Used for type inference.
        self.returned_types: list[PyType] = []

    @property
    def tag(self) -> str:
        if self.cls is not None:
            if self.has_modifier("property"):
                return "property"
            if self.has_modifier("classmethod"):
                return "classmethod"
            if self.has_modifier("staticmethod"):
                return "staticmethod"
            return "method"
        return "function"

    @property
    def detailed_name(self) -> str:
        return f"{self.cls.name}.{self.name}" if self.cls else self.name

    def __str__(self) -> str:
        return f"<{self.tag} {self.detailed_name!r}>"

    def get_type(self) -> PyFunctionType:
        return PyFunctionType(self)

    @property
    def is_method(self) -> bool:
        return self.cls is not None


class PyLambda(PyFunction):
    def __init__(self, scope: "SymbolTable"):
        super().__init__("<lambda>", scope)

    def __str__(self):
        return "<lambda>"

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


class PyVariable(PyEntity):
    def __init__(self, name: str, type: Optional[PyType] = None):
        super().__init__(name)
        self.type = type

    def __str__(self):
        return f"<variable {self.name}>"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name!r} type={self.type!r}>"

    def get_type(self) -> PyType:
        return self.type or PyType.ANY


class PyParameter(PyVariable):
    def __init__(
        self,
        name: str,
        type: Optional[PyType] = None,
        *,
        kwonly: bool = False,
        posonly: bool = False,
        star: Optional[Literal["*", "**"]] = None,
        annotation: Optional["PythonParser.AnnotationContext"] = None,
        star_annotation: Optional["PythonParser.StarAnnotationContext"] = None,
        default: Optional["PythonParser.DefaultContext"] = None,
    ):
        """
        Args:
            posonly: Whether the parameter is positional-only.
            kwonly: Whether the parameter is keyword-only.
            star: Whether the parameter is a star or double-star parameter.
            annotation: The type annotation of the parameter.
            star_annotation: The starred type annotation of the parameter.
            default: The default value of the parameter.
        """
        super().__init__(name, type)

        self.kwonly = kwonly
        self.posonly = posonly
        self.star: Optional[Literal["*", "**"]] = star
        self.annotation = annotation
        self.star_annotation = star_annotation
        self.default = default

    def __str__(self):
        return f"<parameter {self.star or ''}{self.name}>"

    def get_label(self) -> str:
        label = f"{self.star or ''}{self.name}"

        if self.type:
            label += f": {self.type}"
        elif self.annotation:
            label += f": {get_node_source(self.annotation.expression())}"
        elif self.star_annotation:
            label += f": {get_node_source(self.star_annotation.starredExpression())}"

        if self.default:
            label += f" = {get_node_source(self.default.expression())}"

        return label


class PyParameters(list[PyParameter]):
    """
    A list of function parameters.
    """

    def __str__(self):
        return ", ".join(str(param) for param in self)

    def copy(self) -> "PyParameters":
        return PyParameters(self)

    def get_positionals(self) -> list[PyParameter]:
        """
        Returns the positional parameters, including starred parameters.
        """
        params: list[PyParameter] = []
        for param in self:
            if param.kwonly or param.star == "**":
                break
            params.append(param)
        return params

    def get_positional(self, index: int) -> Optional[PyParameter]:
        """
        Returns the positional parameter at the specified index.

        If a starred parameter is encountered before the index, it is returned instead.
        """
        for i, param in enumerate(self):
            if param.kwonly or param.star == "**":
                break
            if param.star == "*":
                return param
            if i == index:
                return param
        return None

    def get_keyword(self, name: str) -> Optional[PyParameter]:
        """
        Returns the keyword parameter with the specified name.

        If no such parameter is found, but a double-starred parameter is found, it is
        returned instead.
        """
        kwargs: Optional[PyParameter] = None
        for param in self:
            if param.star == "**":
                kwargs = param
            elif param.name == name and not param.posonly and param.star is None:
                return param
        return kwargs

    def get_starred(self) -> Optional[PyParameter]:
        """
        Returns the starred parameter if present.
        """
        for param in self:
            if param.star == "*":
                return param
        return None

    def get_double_starred(self) -> Optional[PyParameter]:
        """
        Returns the double-starred parameter if present.
        """
        for param in self:
            if param.star == "**":
                return param
        return None

    def has_bound_param(self) -> bool:
        """
        Returns whether the parameters have a bound parameter.

        The bound parameter is the first parameter, and it should not be keyword-only
        or starred. This parameter is usually named `self` in methods, and `cls` in
        classmethods.
        """
        return bool(self) and not self[0].kwonly and not self[0].star

    def get_bound_param(self) -> Optional[PyParameter]:
        """
        Returns the bound parameter of the parameters.
        """
        return self[0] if self.has_bound_param() else None

    def remove_bound_param(self):
        """
        Removes the bound parameter from the parameters.
        """
        if self.has_bound_param():
            self.pop(0)

    def transform(self, transform: PyTypeTransform) -> "PyParameters":
        return PyParameters(
            PyParameter(
                param.name,
                transform.visit_type(param.type) if param.type else None,
                kwonly=param.kwonly,
                posonly=param.posonly,
                star=param.star,
                annotation=param.annotation,
                star_annotation=param.star_annotation,
                default=param.default,
            )
            for param in self
        )
