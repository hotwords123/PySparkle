import dataclasses
from contextlib import contextmanager
from typing import Iterator, Literal, NamedTuple, Optional, Unpack

from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Token import CommonToken
from antlr4.tree.Tree import TerminalNode

from grammar import PythonParser

from .entity import PyEntity, PyVariable
from .scope import ScopeType, SymbolTable
from .symbol import Symbol, SymbolType
from .token import TokenInfo, TokenKind
from .types import PyInstanceType, PyType


class PythonContext:
    """
    Represents the context of a Python module being analyzed.
    """

    def __init__(self, global_scope: SymbolTable):
        self.global_scope: SymbolTable = global_scope
        self.current_scope: SymbolTable = global_scope
        self.scopes: dict[ParserRuleContext, SymbolTable] = {}

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
        if symbol := self.current_scope.lookup(name):
            self.set_node_info(node, symbol=symbol)
            return symbol.get_type()

        else:
            self.set_node_info(node, kind=TokenKind.IDENTIFIER)
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

        elif isinstance(on_type, PyInstanceType):
            # If the attribute does not exist, but the primary is an instance,
            # then a new attribute is introduced.
            # TODO: Restrict this to happen only in class definitions.
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
class PyParameterSpec:
    """
    Represents a parameter specification.

    Attributes:
        posonly: Whether the parameter is positional-only.
        kwonly: Whether the parameter is keyword-only.
        star: Whether the parameter is a star or double-star parameter.
        annotation: The type annotation of the parameter.
        default: The default value of the parameter.
    """

    posonly: bool = False
    kwonly: bool = False
    star: Optional[Literal["*", "**"]] = None
    annotation: Optional[PythonParser.AnnotationContext] = None
    star_annotation: Optional[PythonParser.StarAnnotationContext] = None
    default: Optional[PythonParser.DefaultContext] = None
