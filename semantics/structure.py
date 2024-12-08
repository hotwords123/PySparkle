import dataclasses
from contextlib import contextmanager
from typing import Iterator, NamedTuple, Optional, Unpack

from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Token import CommonToken
from antlr4.tree.Tree import TerminalNode

from .entity import PyEntity, PyFunction, PyVariable
from .scope import PySymbolNotFoundError, ScopeType, SymbolTable
from .symbol import Symbol, SymbolType
from .token import TokenInfo, TokenKind
from .types import PyClassType, PyFunctionType, PySelfType, PyType


class PythonContext:
    """
    Represents the context of a Python module being analyzed.
    """

    def __init__(self, global_scope: SymbolTable):
        self.global_scope: SymbolTable = global_scope
        self.current_scope: SymbolTable = global_scope
        self.scopes: dict[ParserRuleContext, SymbolTable] = {}

        # Used when analyzing parameter specifications.
        self.parent_function: Optional[PyFunction] = None
        # Used when analyzing function calls.
        self.called_function: Optional[PyFunction] = None

        self.imports: list[PyImport] = []

        self.entities: dict[ParserRuleContext, PyEntity] = {}

        self.token_info: dict[CommonToken, TokenInfo] = {}

        self.errors: list[Exception] = []

    @contextmanager
    def scope_guard(self, scope: SymbolTable) -> Iterator[SymbolTable]:
        old_scope = self.current_scope
        self.current_scope = scope
        try:
            yield scope
        finally:
            self.current_scope = old_scope

    def new_scope(
        self, ctx: ParserRuleContext, name: str, scope_type: ScopeType
    ) -> SymbolTable:
        parent_scope = self.current_scope
        if parent_scope.scope_type is ScopeType.CLASS:
            parent_scope = parent_scope.parent

        scope = SymbolTable(name, scope_type, parent_scope)
        self.scopes[ctx] = scope
        return scope

    def scope_of(self, ctx: ParserRuleContext) -> SymbolTable:
        assert (scope := self.scopes[ctx]) is not None
        return scope

    @property
    def parent_scope(self) -> SymbolTable:
        assert (scope := self.current_scope.parent) is not None
        return scope

    def set_node_info(self, node: TerminalNode, /, **kwargs: Unpack[TokenInfo]):
        token = node.getSymbol()
        self.token_info.setdefault(token, TokenInfo()).update(**kwargs)

    @contextmanager
    def wrap_errors(self, error_cls: type[Exception]) -> Iterator[None]:
        try:
            yield
        except error_cls as e:
            self.errors.append(e)

    def define_variable(
        self,
        name: str,
        node: TerminalNode,
        *,
        type: Optional[PyType] = None,
        scope: Optional[SymbolTable] = None,
    ) -> Symbol:
        """
        Defines a variable in the current scope.

        Args:
            name: The name of the variable.
            node: The terminal node where the variable is defined.
            type: The type of the variable (optional).
            scope: The scope to define the variable in (defaults to the current scope).

        Returns:
            symbol: The symbol representing the variable.
        """
        if scope is None:
            scope = self.current_scope

        if name not in scope:
            entity = PyVariable(name, type=type)
            symbol = Symbol(SymbolType.VARIABLE, name, node, entity=entity)
            scope.define(symbol)
        else:
            symbol = scope[name]

        self.set_node_info(node, symbol=symbol)
        return symbol

    def access_variable(self, name: str, node: TerminalNode) -> PyType:
        """
        Accesses a variable in the current scope.

        Args:
            name: The name of the variable.
            node: The terminal node where the variable is accessed.

        Returns:
            type: The type of the variable.
        """
        try:
            symbol = self.current_scope.lookup(name, raise_from=node.getSymbol())
            self.set_node_info(node, symbol=symbol)
            return symbol.get_type()

        except PySymbolNotFoundError as e:
            self.errors.append(e)
            return PyType.ANY

    def set_variable_type(
        self, symbol: Symbol, type: PyType, override: bool = False
    ) -> PyType:
        """
        Sets the type of the variable represented by a symbol.

        Args:
            symbol: The symbol representing the variable.
            type: The type of the variable.
            override: Whether to override the existing type.

        Returns:
            type: The type of the variable.
        """
        if isinstance(entity := symbol.resolve_entity(), PyVariable):
            if override or entity.type is None:
                entity.type = type
            else:
                type = entity.type

        return type

    def define_attribute(
        self,
        on_type: PyType,
        name: str,
        node: TerminalNode,
        *,
        value_type: Optional[PyType] = None,
        override_type: bool = False,
    ):
        """
        Defines an attribute on a type.

        Args:
            on_type: The type to define the attribute on.
            name: The name of the attribute.
            node: The terminal node where the attribute is defined.
            value_type: The type of the attribute value (optional).
            override_type: Whether to override the existing attribute type.
        """
        if symbol := on_type.get_attr(name):
            # If the attribute exists, the target is an attribute.
            self.set_node_info(node, symbol=symbol)
            if value_type is not None:
                self.set_variable_type(symbol, value_type, override=override_type)

        elif isinstance(on_type, PySelfType):
            # If the attribute does not exist, but the target is `self`, the attribute
            # is defined on the instance scope of the class.
            self.define_variable(
                name, node, type=value_type, scope=on_type.cls.instance_scope
            )

        else:
            # The attribute cannot be defined on the type.
            self.set_node_info(node, kind=TokenKind.FIELD)

    def access_attribute(
        self, on_type: PyType, name: str, node: TerminalNode
    ) -> PyType:
        """
        Accesses an attribute on a type.

        Args:
            on_type: The type to access the attribute on.
            name: The name of the attribute.
            node: The terminal node where the attribute is accessed.

        Returns:
            type: The type of the attribute.
        """
        if symbol := on_type.get_attr(name):
            self.set_node_info(node, symbol=symbol)
            return symbol.get_type()

        else:
            self.set_node_info(node, kind=TokenKind.FIELD)
            return PyType.ANY

    @contextmanager
    def set_parent_function(self, func: PyFunction) -> Iterator[None]:
        old_function = self.parent_function
        self.parent_function = func
        try:
            yield
        finally:
            self.parent_function = old_function

    @contextmanager
    def set_called_function(self, type: PyType) -> Iterator[None]:
        # TODO: This logic should be implemented in the `types` module.
        if isinstance(type, PyFunctionType):
            func = type.func
        elif isinstance(type, PyClassType):
            func = type.cls.lookup_method("__init__")
        else:
            func = None

        old_function = self.called_function
        self.called_function = func
        try:
            yield
        finally:
            self.called_function = old_function

    def access_func_kwarg(self, name: str, node: TerminalNode) -> PyType:
        """
        Accesses a keyword parameter in the called function.

        Args:
            name: The name of the keyword parameter.
            node: The terminal node where the keyword parameter is accessed.

        Returns:
            type: The type of the keyword parameter.
        """
        if self.called_function is not None:
            kwargs_symbol: Optional[Symbol] = None

            for param in self.called_function.parameters:
                if not param.posonly and param.star is None and param.name == name:
                    # There is a keyword parameter by the name.
                    symbol = self.called_function.scope[param.name]
                    self.set_node_info(node, symbol=symbol)
                    return symbol.get_type()

                elif param.star == "**":
                    # There is a double-star parameter. Remember it in case the keyword
                    # parameter is not found.
                    kwargs_symbol = self.called_function.scope[param.name]

            if kwargs_symbol is not None:
                # The argument is passed to the double-star parameter.
                self.set_node_info(node, symbol=kwargs_symbol)
                # TODO: Check the type of the kwargs symbol.
                return PyType.ANY

        # The keyword parameter is not found.
        self.set_node_info(node, kind=TokenKind.VARIABLE)
        return PyType.ANY


@dataclasses.dataclass
class PyImport:
    """
    Represents a Python import statement.

    Attributes:
        path: The list of module names in the import statement.
    """

    path: list[str]


@dataclasses.dataclass
class PyImportName(PyImport):
    """
    Represents a simple import statement.

    Attributes:
        alias: The alias of the imported module, or `None` if not aliased.
        symbol: The symbol that the import resolves to.
    """

    alias: Optional[str]
    symbol: Symbol


@dataclasses.dataclass
class PyImportFrom(PyImport):
    """
    Represents a `from` import statement.

    Attributes:
        relative: The number of parent directories to import from.
        targets: The targets of the import statement.
    """

    relative: Optional[int]
    targets: "PyImportFromTargets"


class PyImportFromTargets(NamedTuple):
    """
    Represents the targets of a `from` import statement.

    Attributes:
        as_names: The list of names imported from the module, or `None` for all.
    """

    as_names: Optional[list["PyImportFromAsName"]] = None


class PyImportFromAsName(NamedTuple):
    """
    Represents a name pair in a `from` import statement.

    Attributes:
        name: The name of the imported entity.
        alias: The alias of the imported entity.
        symbol: The symbol that the import resolves to.
    """

    name: str
    alias: Optional[str]
    symbol: Symbol


@dataclasses.dataclass
class PyArguments:
    args: list["PyType"] = dataclasses.field(default_factory=list)
    kwargs: list["PyKeywordArgument"] = dataclasses.field(default_factory=list)
    double_stars: list["PyType"] = dataclasses.field(default_factory=list)


class PyKeywordArgument(NamedTuple):
    name: str
    type: PyType
