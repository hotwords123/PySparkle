import functools
from pathlib import Path
from typing import Callable

from core.source import PythonSource
from semantics.base import SemanticError
from semantics.entity import PyModule, PyPackage


class ModuleManager:
    """
    Manages the finding and loading of Python modules.
    """

    def __init__(self, search_paths: list[Path], loader: Callable[[PyModule], None]):
        """
        Args:
            search_paths: A list of paths to search for modules in.
            loader: The module loader to use for loading modules.
        """
        self.search_paths = search_paths
        self.loader = loader
        self.suffixes = [".py", ".pyi"]
        self.modules: dict[str, PyModule] = {}

    def __getitem__(self, name: str) -> PyModule:
        return self.import_module(name)

    def __contains__(self, name: str) -> bool:
        return name in self.modules

    def import_module(self, name: str) -> PyModule:
        """
        Imports a Python module by name. If the module is already loaded, it is returned
        immediately. Otherwise, the module is searched for in the search paths and loaded
        if found.

        Args:
            name: The fully qualified module name.
        """
        assert name, "module name must not be empty"
        if name in self.modules:
            return self.modules[name]

        module = self.find_module(name)
        if module.path is not None:
            module.loader = functools.partial(PythonSource.parse_file, module.path)

        self.load_module(module)
        return module

    def find_module(self, name: str) -> PyModule:
        *parent_names, last_name = name.split(".")
        if parent_names:
            # For submodules, use the parent package's import paths.
            package = self.import_module(".".join(parent_names))
            if not isinstance(package, PyPackage):
                raise PyImportError(package.name, f"{package.name!r} is not a package")
            import_paths = package.import_paths
        else:
            # For top-level modules, use the global search paths.
            import_paths = self.search_paths

        # According to PEP 420, while looking for a module or package, these paths are
        # searched in order:
        # 1. `<parent>/name/__init__.py` for a regular package.
        # 2. `<parent>/name.{extension}` for a module.
        # 3. `<parent>/name` for a namespace package. The path is recorded and the
        #    search continues.

        found_paths: list[Path] = []

        for parent_path in import_paths:
            package_path = parent_path / last_name
            for suffix in self.suffixes:
                init_path = package_path / f"__init__{suffix}"
                if init_path.is_file():
                    return PyPackage(name, init_path, [package_path])

            for suffix in self.suffixes:
                module_path = parent_path / f"{last_name}{suffix}"
                if module_path.is_file():
                    return PyModule(name, module_path)

            if package_path.is_dir():
                found_paths.append(package_path)

        if found_paths:
            return PyPackage(name, None, found_paths)

        raise PyImportError(name, f"module {name!r} not found")

    def load_module(self, module: PyModule):
        self.modules[module.name] = module

        try:
            self.loader(module)
        except:
            del self.modules[module.name]
            raise

        module.loaded = True

    def unload_module(self, module: PyModule):
        if module.name in self.modules:
            del self.modules[module.name]


class PyImportError(SemanticError):
    def __init__(self, module_name: str, message: str):
        super().__init__(message)
        self.module_name = module_name
