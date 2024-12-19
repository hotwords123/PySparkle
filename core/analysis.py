import dataclasses
import logging
from pathlib import Path
from typing import Final, Optional

from typeshed_client import SearchContext

from semantics.entity import PyClass, PyModule, PyPackage
from semantics.scope import PyDuplicateSymbolError, ScopeType, SymbolTable
from semantics.structure import PyImportFrom, PyImportName, PythonContext
from semantics.symbol import Symbol, SymbolType
from semantics.types import PyArguments, get_stub_class, set_type_context
from semantics.visitor import PythonVisitor

from .modules import ModuleManager, PyImportError

# Special forms that require special handling in the Python analyzer.
special_form_names: Final = {
    "typing": {
        "Union",
        "Generic",
        "Protocol",
        "Callable",
        "Type",
        "NoReturn",
        "ClassVar",
        "Optional",
        "Tuple",
        "Final",
        "Literal",
        # Introduced in Python 3.11.
        "Self",
        "Never",
        "Unpack",
        "Required",
        "NotRequired",
        "LiteralString",
        # Introduced in Python 3.10.
        "Concatenate",
        "TypeAlias",
        "TypeGuard",
        # Introduced in Python 3.9.
        "Annotated",
    },
    "typing_extensions": {
        "Protocol",
        "Final",
        "Literal",
        "Annotated",
        "Concatenate",
        "TypeAlias",
        "TypeGuard",
        "Self",
        "Never",
        "Required",
        "NotRequired",
        "LiteralString",
        "Unpack",
        "ReadOnly",
        "TypeIs",
    },
}

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PythonAnalyzerConfig:
    """
    Configuration for the Python analyzer.
    """

    report_errors: bool = True
    """Whether to report errors during analysis."""
    load_imports: bool = True
    """Whether to load imports during analysis."""


class PythonAnalyzer:
    def __init__(
        self,
        root_paths: Optional[list[Path]] = None,
        search_context: Optional[SearchContext] = None,
        config: Optional[PythonAnalyzerConfig] = None,
    ):
        """
        Initializes a Python analyzer.

        Args:
            root_paths: The root paths to search for Python modules.
            search_context: The search context for the typeshed stubs.
            config: The configuration for the analyzer.
        """
        self.root_paths = root_paths or []
        self.importer = ModuleManager(
            self.root_paths, self._load_module, search_context=search_context
        )
        self.config = config or PythonAnalyzerConfig()

        self.builtin_scope = SymbolTable("builtins", ScopeType.BUILTINS)
        self.type_stubs: dict[str, SymbolTable] = {}
        self.typeshed_loaded = False

        self._pending_second_pass: list[PyModule] = []
        self._module_cls: Optional[PyClass] = None

    def create_module(self, path: Path) -> PyModule:
        """
        Creates a module entity for a given file path.

        The module is not loaded or added to the module manager. In order to load the
        module, the `loader` attribute on the module must be set to a function that
        returns the source of the module.
        """
        for parent_path in self.root_paths:
            # Try to find the root path that contains the module.
            if path.is_relative_to(parent_path):
                relative_path = path.relative_to(parent_path)
                parts = relative_path.parts
                break
        else:
            # Otherwise, use a temporary name.
            parts = ("@", path.stem)

        # Remove any file extensions and convert to module name.
        parts = [part.split(".")[0] for part in parts]

        if parts[-1] == "__init__":
            # For packages, remove the last part.
            module_name = ".".join(parts[:-1])
            return PyPackage(module_name, path, [path.parent])
        else:
            module_name = ".".join(parts)
            return PyModule(module_name, path)

    def load_module(self, module: PyModule, reload: bool = False) -> bool:
        """
        Loads a Python module.
        """
        if module in self.importer:
            if not reload:
                return False
            self.unload_module(module)
        self.importer.load_module(module)
        return True

    def unload_module(self, module: PyModule):
        """
        Unloads a Python module.
        """
        self.importer.unload_module(module)

    def _load_module(self, module: PyModule):
        """
        The internal method for loading a Python module.
        """
        global_scope = self._build_global_scope(module)
        module.context = PythonContext(global_scope)

        if module.loader is not None:
            logger.info(f"Loading {module} from {module.path}")
            module.source = module.loader()
            module.loader = None

            visitor = PythonVisitor(module.context)
            visitor.first_pass(module.source.tree)

            if self.config.load_imports:
                self._load_imports(module)

            if self.typeshed_loaded:
                with self.set_type_context():
                    visitor.second_pass(module.source.tree)
                self._report_module_errors(module)
            else:
                self._pending_second_pass.append(module)

    def _report_module_errors(self, module: PyModule):
        """
        Reports the errors within a Python module.
        """
        if self.config.report_errors and module.context.errors:
            logger.info(f"In {module}:")
            for error in module.context.errors:
                logger.error(f"  {error}")

    def _build_global_scope(self, module: PyModule) -> SymbolTable:
        """
        Builds the global scope for a Python module.
        """
        scope = SymbolTable(module.name, ScopeType.GLOBAL, self.builtin_scope)

        self._define_module_attrs(scope)

        if module.name in special_form_names:
            # Define special forms as classes in the global scope.
            # This is a hack to make the subscript syntax work for them.
            for name in special_form_names[module.name]:
                cls_scope = SymbolTable(
                    name, ScopeType.CLASS, full_name=f"{module.name}.{name}"
                )
                cls = PyClass(name, cls_scope)
                # XXX: This is a hack to deal with the aliasing of special forms.
                cls.set_full_name(f"typing.{name}")
                cls.set_modifier("special")
                cls.arguments = PyArguments()
                symbol = Symbol(SymbolType.CLASS, name, entity=cls)
                scope.define(symbol)

        return scope

    def _load_imports(self, module: PyModule):
        """
        Loads the imports for a Python module.
        """
        for stmt in module.context.imports:
            try:
                if isinstance(stmt, PyImportName):
                    self._load_import_name(stmt)

                elif isinstance(stmt, PyImportFrom):
                    if isinstance(module, PyPackage):
                        base_name = module.name
                    else:
                        base_name = module.package
                    self._load_import_from(base_name, module.context, stmt)
            except PyImportError as e:
                module.context.errors.append(e.with_context(stmt.ctx))

    def _load_import_name(self, stmt: PyImportName):
        """
        Loads an import-name statement for a Python module.

        Args:
            context: The context of the Python module.
            stmt: The import-name statement to load.
        """
        # Import the top-level module object.
        module = self.importer.import_module(stmt.path[0])
        stmt.path_symbols[0].set_entity(module)

        # Import the submodules as attributes.
        module = self._import_and_define_submodules(
            module, stmt.path[1:], stmt.path_symbols[1:]
        )

        # If an alias is provided, the symbol refers to the imported module object.
        if stmt.alias_symbol is not None:
            stmt.alias_symbol.set_entity(module)

    def _load_import_from(
        self, base_name: str, context: PythonContext, stmt: PyImportFrom
    ):
        """
        Loads an import-from statement for a Python module.

        Args:
            base_name: The name of the module that the import statement is relative to.
            context: The context of the Python module.
            stmt: The import-from statement to load.
        """
        if stmt.relative is not None:
            # Ensure that the relative import is valid.
            if not base_name:
                raise PyImportError(
                    "attempted relative import with no known parent package",
                )

            # Resolve the full path of the module to import from.
            base_path = base_name.split(".")
            if (offset := len(base_path) - stmt.relative) <= 0:
                raise PyImportError(
                    "attempted relative import beyond top-level package",
                )

            # Import the module that the import statement is relative to.
            module = self.importer.import_module(".".join(base_path[:offset]))
            module = self._import_and_define_submodules(
                module, stmt.path, stmt.path_symbols
            )

        else:
            # Import the top-level module object.
            module = self.importer.import_module(stmt.path[0])
            stmt.path_symbols[0].set_entity(module)

            # Import the submodules as attributes.
            module = self._import_and_define_submodules(
                module, stmt.path[1:], stmt.path_symbols[1:]
            )

        imported_scope = module.context.global_scope

        if stmt.targets.as_names is not None:
            # Import specific symbols from the module.
            for name, _, symbols in stmt.targets.as_names:
                target_symbol = imported_scope.get(name)

                # If the imported module has an attribute by the name, import it.
                # Note that we also check if the target symbol is the imported symbol.
                # This may happen if a module is importing itself, for example in
                # package __init__.py files.
                if target_symbol is not None and target_symbol not in symbols:
                    for symbol in symbols:
                        symbol.target = target_symbol
                    continue

                # Otherwise, attempt to import a submodule with the name.
                try:
                    submodule = self._import_and_define_module(module, name)
                except PyImportError:
                    context.errors.append(
                        PyImportError(
                            f"Cannot import name {name!r} from {module.name!r}"
                        ).with_context(stmt.ctx)
                    )
                    continue

                for symbol in symbols:
                    symbol.set_entity(submodule)

        else:
            # Import all public symbols from the module.
            for target_symbol in imported_scope.iter_symbols(public_only=True):
                alias_symbol = Symbol(
                    SymbolType.IMPORTED, target_symbol.name, target=target_symbol
                )

                try:
                    context.global_scope.define(alias_symbol)
                except PyDuplicateSymbolError as e:
                    context.errors.append(e.with_context(stmt.ctx))

    def _import_and_define_submodules(
        self, module: PyModule, path: list[str], symbols: list[Symbol]
    ) -> PyModule:
        """
        Imports and defines submodules in the global scope of a module.

        Args:
            module: The top-level module to import the submodules into.
            path: The list of submodule names to import.
            symbols: The list of symbols for each submodule name.

        Returns:
            module: The last submodule that was imported.
        """
        for i, name in enumerate(path):
            module = self._import_and_define_module(module, name)
            symbols[i].set_entity(module)
        return module

    def _import_and_define_module(self, module: PyModule, name: str) -> PyModule:
        """
        Imports and defines a submodule in the global scope of a module.

        Args:
            module: The module to import the submodule into.
            name: The name of the submodule to import.

        Returns:
            submodule: The imported submodule.
        """
        submodule = self.importer.import_module(f"{module.name}.{name}")

        if name not in module.context.global_scope:
            symbol = Symbol(SymbolType.IMPORTED, name, entity=submodule)
            module.context.global_scope.define(symbol)

        return submodule

    def load_typeshed(self):
        """
        Loads the built-in symbols from the typeshed stubs.
        """
        # Temporarily disable the user search paths.
        self.importer.search_paths = []
        try:
            self._load_typeshed()
        finally:
            self.importer.search_paths = self.root_paths

    def _load_typeshed(self):
        builtins_module = self.importer.import_module("builtins")

        for symbol in builtins_module.context.global_scope.iter_symbols(
            public_only=True
        ):
            self.builtin_scope.define(
                symbol.copy(node=None, public=None, target=symbol)
            )

        for module_name in ("builtins", "types", "typing", "abc"):
            module = self.importer.import_module(module_name)
            self.type_stubs[module_name] = module.context.global_scope

        with self.set_type_context():
            self._module_cls = get_stub_class("types.ModuleType")
            assert self._module_cls is not None, "types.ModuleType not found"

        for module in self.importer.modules.values():
            self._define_module_attrs(module.context.global_scope)

        self.typeshed_loaded = True

        with self.set_type_context():
            for module in self._pending_second_pass:
                visitor = PythonVisitor(module.context)
                visitor.second_pass(module.source.tree)
                self._report_module_errors(module)

        self._pending_second_pass.clear()

    def _define_module_attrs(self, scope: SymbolTable):
        """
        Defines the attributes of a module object in the global scope.
        """
        if self._module_cls is None:
            return

        # Define the built-in symbols on module objects.
        # https://docs.python.org/3/reference/datamodel.html#module-objects
        for symbol in self._module_cls.scope.iter_symbols():
            if symbol.name not in ("__init__", "__getattr__"):
                scope.define(symbol.copy(node=None, public=None, target=symbol))

    def set_type_context(self):
        """
        Sets the context for the type analyzer.
        """
        return set_type_context(self.type_stubs)
