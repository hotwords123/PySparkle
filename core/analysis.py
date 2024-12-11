import sys
from pathlib import Path
from typing import Final, Optional

from semantics.entity import PyClass, PyModule, PyPackage
from semantics.scope import PyDuplicateSymbolError, ScopeType, SymbolTable
from semantics.structure import PyImportFrom, PyImportName, PythonContext
from semantics.symbol import Symbol, SymbolType
from semantics.types import get_stub_class, set_type_context
from semantics.visitor import PythonVisitor

from .modules import ModuleManager, PyImportError
from .source import PythonSource

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
    },
    "typing_extensions": {
        "Protocol",
        "Final",
        "Literal",
    },
}


class PythonAnalyzer:
    def __init__(self, search_paths: list[Path], report_errors: bool = True):
        self.builtin_scope = SymbolTable("builtins", ScopeType.BUILTINS)
        self.type_stubs: dict[str, SymbolTable] = {}
        self.importer = ModuleManager(search_paths, self.load_module)
        self.report_errors = report_errors
        self.typeshed_loaded = False
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

            if self.typeshed_loaded:
                with self.set_type_context():
                    visitor.second_pass(module.source.tree)
                self.report_module_errors(module)
            else:
                self.pending_second_pass.append(module)

    def report_module_errors(self, module: PyModule):
        """
        Reports the errors within a Python module.
        """
        if not self.report_errors:
            return

        if module.context.errors:
            print(f"In {module}:", file=sys.stderr)

            for error in module.context.errors:
                print(f"  {error}", file=sys.stderr)

            print(file=sys.stderr)

    def build_global_scope(self, module: PyModule) -> SymbolTable:
        """
        Builds the global scope for a Python module.
        """
        scope = SymbolTable(module.name, ScopeType.GLOBAL, self.builtin_scope)

        # Define the built-in symbols on module objects.
        # https://docs.python.org/3/library/stdtypes.html#module-objects

        scope.define(Symbol(SymbolType.VARIABLE, "__doc__"))
        scope.define(Symbol(SymbolType.VARIABLE, "__annotations__"))

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
                cls.modifiers.add("special")
                symbol = Symbol(SymbolType.CLASS, name, entity=cls)
                scope.define(symbol)

        return scope

    def load_imports(self, module: PyModule):
        """
        Loads the imports for a Python module.
        """
        for stmt in module.context.imports:
            if isinstance(stmt, PyImportName):
                self.load_import_name(module.context, stmt)
            elif isinstance(stmt, PyImportFrom):
                if isinstance(module, PyPackage):
                    base_name = module.name
                else:
                    base_name = module.package
                self.load_import_from(base_name, module.context, stmt)

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
            stmt.symbol.set_entity(imported_module)

        else:
            # Otherwise, the symbol refers to the top-level module object,
            # and the submodules are imported as attributes.
            module = self.importer[stmt.path[0]]
            stmt.symbol.set_entity(module)

            for name in stmt.path[1:]:
                module = self.import_and_define_module(module, name, context)
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
                    if symbol is not target_symbol:
                        symbol.target = target_symbol

                    # Special case: importing from self.
                    if (
                        imported_module.name == base_name
                        and target_symbol.type is SymbolType.IMPORTED
                        and target_symbol.resolve_entity() is None
                    ):
                        with context.wrap_errors(PyImportError):
                            symbol.set_entity(
                                self.importer.import_module(f"{base_name}.{name}")
                            )

                elif self.import_and_define_module(imported_module, name, context):
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
        self, module: PyModule, name: str, error_context: PythonContext
    ) -> Optional[PyModule]:
        """
        Imports and defines a submodule in the global scope of a module.

        Args:
            module: The module to import the submodule into.
            name: The name of the submodule to import.
            error_context: The context to report errors to.

        Returns:
            submodule: The imported submodule, or None if the import failed.
        """
        try:
            submodule = self.importer.import_module(f"{module.name}.{name}")
        except PyImportError as e:
            if error_context is module.context:
                # This error originates from an import-name statement.
                error_context.errors.append(e)
            else:
                # This error originates from an import-from statement.
                error_context.errors.append(
                    PyImportError(
                        module.name, f"Cannot import name {name!r} from {module.name!r}"
                    )
                )
            return None

        symbol = Symbol(SymbolType.IMPORTED, name, entity=submodule)

        with error_context.wrap_errors(PyDuplicateSymbolError):
            module.context.global_scope.define(symbol)

        return submodule

    def load_typeshed(self):
        """
        Loads the built-in symbols from the typeshed stubs.
        """
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
            module_cls = get_stub_class("types.ModuleType")
            assert module_cls is not None, "types.ModuleType not found"
            for symbol in module_cls.scope.iter_symbols():
                self.builtin_scope.define(
                    symbol.copy(node=None, public=None, target=symbol)
                )

        self.typeshed_loaded = True

        with self.set_type_context():
            for module in self.pending_second_pass:
                visitor = PythonVisitor(module.context)
                visitor.second_pass(module.source.tree)
                self.report_module_errors(module)

        self.pending_second_pass.clear()

    def set_type_context(self):
        """
        Sets the context for the type analyzer.
        """
        return set_type_context(self.type_stubs)
