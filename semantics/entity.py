from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal, Optional

from .scope import ScopeType, SymbolTable
from .types import (
    PyClassType,
    PyEllipsisType,
    PyFunctionType,
    PyInstanceType,
    PyModuleType,
    PyNoneType,
    PySelfType,
    PyType,
    get_stub_class,
)

if TYPE_CHECKING:
    from core.source import PythonSource
    from grammar import PythonParser

    from .structure import PyArguments, PythonContext


class PyEntity(ABC):
    """
    A Python entity.
    """

    def __init__(self, name: str):
        self.name = name

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
        super().__init__(name)
        self.path = path

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
    __slots__ = ("modifiers",)

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
        self.instance_scope = SymbolTable(f"<object '{name}'>", ScopeType.OBJECT)
        self.decorators: list[PyType] = []
        self.arguments: Optional["PyArguments"] = None
        self.bases: list[PyClass] = []
        self.mro: list[PyClass] = []

        self.init_modifiers()

    def __str__(self) -> str:
        return f"<class {self.name!r}>"

    def get_type(self) -> PyClassType:
        return PyClassType(self)

    def get_instance_type(self) -> PyInstanceType:
        if self is get_stub_class("types.NoneType"):
            return PyNoneType()
        if self is get_stub_class("types.EllipsisType"):
            return PyEllipsisType()
        return PyInstanceType(self)

    def get_self_type(self) -> PySelfType:
        return PySelfType(self)

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

    def compute_mro(self):
        """
        Computes the method resolution order for the class according to the C3
        linearization algorithm.

        References:
        - https://docs.python.org/3/howto/mro.html
        """
        # Collect the MROs of the base classes.
        mro_lists: list[list[PyClass]] = []
        for base in self.bases:
            if not base.mro:
                raise PyTypeError(
                    f"Cannot inherit from base class {base.name!r} before it is defined"
                    f" (in class {self.name!r})"
                )
            mro_lists.append(base.mro.copy())

        # Add the base classes to preserve the orderings.
        if self.bases:
            mro_lists.append([base for base in self.bases])

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
                    f"Cannot create a consistent MRO (in class {self.name!r})"
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
        if (object_cls := get_stub_class("builtins.object")) and result[
            -1
        ] is not object_cls:
            result.append(object_cls)

        self.mro = result


class PyFunction(_ModifiersMixin, PyEntity):
    def __init__(self, name: str, scope: "SymbolTable", cls: Optional[PyClass] = None):
        super().__init__(name)
        self.scope = scope
        self.cls = cls
        self.parameters: list[PyParameter] = []
        self.return_type: Optional[PyType] = None
        self.decorators: list[PyType] = []

        self.init_modifiers()

    def __str__(self) -> str:
        if self.is_method:
            if self.has_modifier("property"):
                tag = "property"
            elif self.has_modifier("classmethod"):
                tag = "classmethod"
            elif self.has_modifier("staticmethod"):
                tag = "staticmethod"
            else:
                tag = "method"

            return f"<{tag} {self.cls.name}.{self.name}>"

        return f"<function {self.name}>"

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
        self.star = star
        self.annotation = annotation
        self.star_annotation = star_annotation
        self.default = default

    def __str__(self):
        return f"<parameter {self.star or ''}{self.name}>"


class PyTypeError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
