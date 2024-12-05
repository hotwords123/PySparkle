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


class PyClass(PyEntity):
    def __init__(self, name: str, scope: "SymbolTable"):
        super().__init__(name)
        self.scope = scope
        self.instance_scope = SymbolTable(f"<object '{name}'>", ScopeType.OBJECT)

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


class PyFunction(PyEntity):
    def __init__(self, name: str, scope: "SymbolTable"):
        super().__init__(name)
        self.scope = scope
        self.parameters: list[PyParameter] = []
        self.return_type: Optional[PyType] = None

    def get_type(self) -> PyFunctionType:
        return PyFunctionType(self)


class PyLambda(PyFunction):
    pass


class PyVariable(PyEntity):
    def __init__(self, name: str, type: Optional[PyType] = None):
        super().__init__(name)
        self.type = type

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
