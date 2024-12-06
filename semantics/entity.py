from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .scope import ScopeType, SymbolTable
from .types import PyClassType, PyFunctionType, PyInstanceType, PyModuleType, PyType

if TYPE_CHECKING:
    from core.source import PythonSource

    from .structure import PyParameterSpec, PythonContext


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

        self.init_modifiers()

    def __str__(self) -> str:
        return f"<class {self.name!r}>"

    def get_type(self) -> PyClassType:
        return PyClassType(self)

    def get_instance_type(self) -> PyInstanceType:
        return PyInstanceType(self)

    def get_method(self, name: str) -> Optional["PyFunction"]:
        # TODO: Implement method lookup.
        if symbol := self.scope.get(name):
            if isinstance(entity := symbol.resolve_entity(), PyFunction):
                return entity
        return None

    def get_method_return_type(self, name: str) -> PyType:
        if method := self.get_method(name):
            return method.return_type or PyType.ANY
        return PyType.ANY


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
        spec: "PyParameterSpec",
        type: Optional[PyType] = None,
    ):
        super().__init__(name, type)
        self.spec = spec

    def __str__(self):
        return f"<parameter {self.spec.star or ''}{self.name}>"

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} {self.name!r}"
            f" spec={self.spec!r} type={self.type!r}>"
        )
