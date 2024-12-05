from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .scope import SymbolTable

if TYPE_CHECKING:
    from core.source import PythonSource

    from .structure import PythonContext


class PyEntity(ABC):
    """
    A Python entity.
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name!r}>"


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


class PyFunction(PyEntity):
    def __init__(self, name: str, scope: "SymbolTable"):
        super().__init__(name)
        self.scope = scope


class PyVariable(PyEntity):
    pass


class PyParameter(PyVariable):
    pass
