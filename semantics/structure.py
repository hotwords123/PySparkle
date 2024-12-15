import dataclasses
from typing import NamedTuple, Optional, Unpack

from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Token import CommonToken
from antlr4.tree.Tree import TerminalNode

from grammar import PythonParser

from .entity import PyEntity, PyFunction, PyModule
from .scope import SymbolTable
from .symbol import Symbol, SymbolType
from .token import TOKEN_KIND_MAP, TokenInfo, TokenKind, TokenModifier
from .types import PyArguments, PyFunctionType, PyType

PyFunctionCall = tuple[PyFunctionType, PyArguments]


class PythonContext:
    """
    Represents the context of a Python module being analyzed.
    """

    def __init__(self, global_scope: SymbolTable):
        self.global_scope: SymbolTable = global_scope
        self.scopes: dict[ParserRuleContext, SymbolTable] = {}

        self.imports: list[PyImport] = []

        self.entities: dict[ParserRuleContext, PyEntity] = {}

        self.token_info: dict[CommonToken, TokenInfo] = {}
        self.node_types: dict[ParserRuleContext, PyType] = {}
        self.function_calls: dict[ParserRuleContext, PyFunctionCall] = {}

        self.errors: list[Exception] = []

    def set_node_scope(self, node: ParserRuleContext, scope: SymbolTable):
        self.scopes[node] = scope

    def get_node_scope(self, node: ParserRuleContext) -> SymbolTable:
        while node is not None:
            if scope := self.scopes.get(node):
                return scope
            node = node.parentCtx

        return self.global_scope

    def set_node_info(self, node: TerminalNode, /, **kwargs: Unpack[TokenInfo]):
        token = node.getSymbol()
        modifiers = kwargs.pop("modifiers", None)

        token_info = self.token_info.setdefault(token, TokenInfo())
        token_info.update(**kwargs)

        if modifiers is not None:
            token_info.setdefault("modifiers", TokenModifier(0))
            token_info["modifiers"] |= modifiers

    def get_token_kind(self, token: CommonToken) -> TokenKind:
        if token_info := self.token_info.get(token):
            if token_kind := token_info.get("kind"):
                return token_kind

            if symbol := token_info.get("symbol"):
                symbol = symbol.resolve()

                match symbol.type:
                    case SymbolType.VARIABLE | SymbolType.PARAMETER:
                        return TokenKind.VARIABLE
                    case SymbolType.FUNCTION:
                        pass  # May be a @property method.
                    case SymbolType.CLASS:
                        return TokenKind.CLASS
                    case SymbolType.GLOBAL | SymbolType.NONLOCAL | SymbolType.IMPORTED:
                        pass  # Unresolved symbols.

                if entity := symbol.resolve_entity():
                    if isinstance(entity, PyModule):
                        return TokenKind.MODULE

                    if isinstance(entity, PyFunction):
                        if entity.has_modifier("property"):
                            return TokenKind.VARIABLE
                        return TokenKind.FUNCTION

                    return TokenKind.VARIABLE

        return TOKEN_KIND_MAP.get(token.type, TokenKind.NONE)

    def get_token_modifiers(self, token: CommonToken) -> TokenModifier:
        if token_info := self.token_info.get(token):
            if token_modifiers := token_info.get("modifiers"):
                return token_modifiers

        return TokenModifier(0)

    def get_token_target(self, token: CommonToken) -> Optional[Symbol]:
        if token_info := self.token_info.get(token):
            if symbol := token_info.get("symbol"):
                return symbol.resolve()

        return None

    def get_token_entity_type(self, token: CommonToken) -> Optional[PyType]:
        if token_info := self.token_info.get(token):
            if type_ := token_info.get("type"):
                return type_

            if symbol := token_info.get("symbol"):
                return symbol.get_type()

        return None

    def set_node_type(self, node: ParserRuleContext, type_: PyType):
        self.node_types[node] = type_

    def get_node_type(self, node: ParserRuleContext) -> PyType:
        return self.node_types.get(node, PyType.ANY)

    def set_function_call(
        self, node: ParserRuleContext, function: PyFunctionType, arguments: PyArguments
    ):
        self.function_calls[node] = (function, arguments)

    def get_function_call(self, node: ParserRuleContext) -> Optional[PyFunctionCall]:
        return self.function_calls.get(node)


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
        ctx: The context of the import statement.
    """

    alias: Optional[str]
    symbol: Symbol
    ctx: PythonParser.ImportNameContext


@dataclasses.dataclass
class PyImportFrom(PyImport):
    """
    Represents a `from` import statement.

    Attributes:
        relative: The number of parent directories to import from.
        targets: The targets of the import statement.
        ctx: The context of the import statement.
    """

    relative: Optional[int]
    targets: "PyImportFromTargets"
    ctx: PythonParser.ImportFromContext


class PyImportFromTargets(NamedTuple):
    """
    Represents the targets of a `from` import statement.

    Attributes:
        as_names: The list of names imported from the module, or `None` for all.
    """

    as_names: Optional[list["PyImportFromAsName"]]


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
