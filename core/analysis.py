import sys
from pathlib import Path
from typing import Optional

from semantics.entity import ModuleEntity
from semantics.scope import PyDuplicateSymbolError, ScopeType, SymbolTable
from semantics.structure import PyImportFrom, PyImportName, PythonContext
from semantics.symbol import Symbol, SymbolType
from semantics.visitor import PythonVisitor

from .modules import ModuleManager, PyImportError, PyModule, PyPackage
from .source import PythonSource


class PythonAnalyzer:
    def __init__(self, search_paths: list[Path]):
        self.builtin_scope = SymbolTable("<builtins>", ScopeType.BUILTINS)
        self.importer = ModuleManager(search_paths, self.load_module)
        self.builtins_loaded = False
        self.pending_second_pass: list[PyModule] = []

    def load_module(self, module: PyModule):
        """
        Loads a Python module.
        """
        global_scope = self.build_global_scope(module)
        module.context = PythonContext(global_scope)

        if module.path is not None:
            print(f"Loading module {module.name!r} from {module.path}", file=sys.stderr)

            module.source = PythonSource.parse(module.path)

            visitor = PythonVisitor(module.context)
            visitor.first_pass(module.source.tree)

            self.load_imports(module)

            if self.builtins_loaded:
                visitor.second_pass(module.source.tree)
            else:
                self.pending_second_pass.append(module)

    def build_global_scope(self, module: PyModule) -> SymbolTable:
        """
        Builds the global scope for a Python module.
        """
        scope = SymbolTable(
            f"<module {module.name!r}>", ScopeType.GLOBAL, self.builtin_scope
        )

        # Define the built-in symbols on module objects.
        # https://docs.python.org/3/library/stdtypes.html#module-objects

        scope.define(Symbol(SymbolType.VARIABLE, "__name__"))
        scope.define(Symbol(SymbolType.VARIABLE, "__spec__"))
        scope.define(Symbol(SymbolType.VARIABLE, "__package__"))
        scope.define(Symbol(SymbolType.VARIABLE, "__loader__"))

        if isinstance(module, PyPackage):
            scope.define(Symbol(SymbolType.VARIABLE, "__path__"))

        if module.path is not None:
            scope.define(Symbol(SymbolType.VARIABLE, "__file__"))
            scope.define(Symbol(SymbolType.VARIABLE, "__cached__"))

        scope.define(Symbol(SymbolType.VARIABLE, "__doc__"))
        scope.define(Symbol(SymbolType.VARIABLE, "__annotations__"))

        scope.define(Symbol(SymbolType.VARIABLE, "__dict__"))

        return scope

    def load_imports(self, module: PyModule):
        """
        Loads the imports for a Python module.
        """
        for stmt in module.context.imports:
            if isinstance(stmt, PyImportName):
                self.load_import_name(module.context, stmt)
            elif isinstance(stmt, PyImportFrom):
                self.load_import_from(module.name, module.context, stmt)

    def load_import_name(self, context: PythonContext, stmt: PyImportName):
        """
        Loads an import-name statement for a Python module.

        Args:
            context: The context of the Python module.
            stmt: The import-name statement to load.
        """
        try:
            imported_module = self.importer.import_module(".".join(stmt.path))
        except PyImportError as e:
            context.errors.append(e)
            return

        if stmt.alias is not None:
            # If an alias is provided, the symbol refers to the imported
            # module object.
            stmt.symbol.entity = ModuleEntity(imported_module)

        else:
            # Otherwise, the symbol refers to the top-level module object,
            # and the submodules are imported as attributes.
            module = self.importer[stmt.path[0]]
            stmt.symbol.entity = ModuleEntity(module)

            for name in stmt.path[1:]:
                module = self.import_and_define_module(module, name)
                assert module is not None

    def load_import_from(
        self, base_name: str, context: PythonContext, stmt: PyImportFrom
    ):
        """
        Loads an import-from statement for a Python module.

        Args:
            base_name: The name of the module that the import statement is relative to.
            context: The context of the Python module.
            stmt: The import-from statement to load.
        """
        try:
            if stmt.relative is not None:
                # Resolve the full path of the module to import from.
                base_path = base_name.split(".")
                if (offset := len(base_path) - stmt.relative) <= 0:
                    raise PyImportError(
                        "." * (1 + stmt.relative) + ".".join(stmt.path),
                        f"Attempted relative import beyond top-level package",
                    )
                path = base_path[:offset] + stmt.path

            else:
                path = stmt.path

            imported_module = self.importer.import_module(".".join(path))

        except PyImportError as e:
            context.errors.append(e)
            return

        imported_scope = imported_module.context.global_scope

        if stmt.targets.as_names is not None:
            # Import specific symbols from the module.
            for name, _, symbol in stmt.targets.as_names:
                if target_symbol := imported_scope.get(name):
                    # If the imported module has an attribute by the name, import it.
                    symbol.target = target_symbol

                elif self.import_and_define_module(imported_module, name):
                    # Otherwise, attempt to import a submodule with the name.
                    symbol.target = imported_scope[name]

        else:
            # Import all public symbols from the module.
            for target_symbol in imported_scope.iter_symbols(public_only=True):
                symbol = Symbol(
                    SymbolType.IMPORTED, target_symbol.name, target=target_symbol
                )

                with context.wrap_errors(PyDuplicateSymbolError):
                    context.global_scope.define(symbol)

    def import_and_define_module(
        self, module: PyModule, name: str
    ) -> Optional[PyModule]:
        """
        Imports and defines a submodule in the global scope of a module.

        Args:
            module: The module to import the submodule into.
            name: The name of the submodule to import.

        Returns:
            submodule: The imported submodule, or None if the import failed.
        """
        try:
            submodule = self.importer.import_module(f"{module.name}.{name}")
        except PyImportError as e:
            module.context.errors.append(e)
            return None

        entity = ModuleEntity(submodule)
        symbol = Symbol(SymbolType.IMPORTED, name, entity=entity)

        with module.context.wrap_errors(PyDuplicateSymbolError):
            module.context.global_scope.define(symbol)

        return submodule

    def load_builtins(self):
        """
        Loads the built-in symbols.
        """
        builtins_module = self.importer.import_module("builtins")

        for symbol in builtins_module.context.global_scope.iter_symbols(
            public_only=True
        ):
            self.builtin_scope.define(symbol)

        self.builtins_loaded = True

        for module in self.pending_second_pass:
            visitor = PythonVisitor(module.context)
            visitor.second_pass(module.source.tree)

        self.pending_second_pass.clear()
