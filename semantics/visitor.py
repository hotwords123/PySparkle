import functools
from ast import literal_eval
from contextlib import contextmanager
from typing import Iterator, Literal, NamedTuple, Optional

from antlr4.error.Errors import RecognitionException
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.tree.Tree import ErrorNode, TerminalNode

from grammar import PythonParser, PythonParserVisitor

from .base import PySyntaxError, SemanticError
from .entity import PyClass, PyFunction, PyLambda, PyParameter, PyVariable
from .scope import PyDuplicateSymbolError, PySymbolNotFoundError, ScopeType, SymbolTable
from .structure import (
    PyImportFrom,
    PyImportFromAsName,
    PyImportFromTargets,
    PyImportName,
    PythonContext,
)
from .symbol import Symbol, SymbolType
from .token import TokenKind
from .types import (
    PyArguments,
    PyClassType,
    PyDictDisplayItem,
    PyFunctionType,
    PyInstanceType,
    PyKeywordArgument,
    PyKvPair,
    PyLiteralType,
    PyPackedTuple,
    PySelfType,
    PyTupleType,
    PyType,
    PyTypeError,
    PyUnionType,
    PyUnpack,
    PyUnpackKv,
    infer_dict_display,
    infer_list_display,
    set_error_reporter,
    set_forward_ref_evaluator,
)

_VisitorAction = Literal["visit", "default", "skip"]


def _visitor_guard(
    first_pass: _VisitorAction = "visit",
    second_pass: _VisitorAction = "visit",
    error_action: _VisitorAction = "default",
):
    """
    Decorator for visitor methods that performs different actions based on the pass
    number.

    Args:
        first_pass: The action to perform in the first pass.
        second_pass: The action to perform in the second pass.
        error_action: The action to perform if the node contains an error.

    Returns:
        decorator: The decorator function.
    """
    passes: dict[int, _VisitorAction] = {1: first_pass, 2: second_pass}

    def decorator(func):
        @functools.wraps(func)
        def wrapper(visitor: "PythonVisitor", ctx: ParserRuleContext, **kwargs):
            if ctx in visitor.context.scopes:
                old_scope = visitor._current_scope
                visitor._current_scope = visitor.context.scopes[ctx]
            else:
                old_scope = None

            old_ctx, visitor._current_ctx = visitor._current_ctx, ctx

            action = passes.get(visitor.pass_num, "visit")
            if ctx.exception is not None:
                visitor._handle_parser_error(ctx)
                if action == "visit":
                    action = error_action

            result = None
            try:
                match action:
                    case "visit":
                        if error_action == "default":
                            visitor._visit_error_nodes(ctx)
                        result = func(visitor, ctx, **kwargs)
                    case "default":
                        result = super(PythonVisitor, visitor).visitChildren(ctx)
                    case "skip":
                        pass

            except Exception as e:
                visitor._report_error(e)

            finally:
                visitor._current_ctx = old_ctx
                if old_scope is not None:
                    visitor._current_scope = old_scope

            if isinstance(result, PyType):
                visitor.context.set_node_type(ctx, result)

            return result

        return wrapper

    return decorator


_both_passes = _visitor_guard()
_first_pass_only = _visitor_guard(second_pass="skip")
_type_check = _visitor_guard(first_pass="default")


class _OuterSymbol(NamedTuple):
    symbol: Symbol
    scope: SymbolTable


_COMMON_MODIFIERS = {
    # Type checking related
    "typing.no_type_check": "no_type_check",
    "typing.runtime_checkable": "runtime_checkable",
    "typing.type_check_only": "type_check_only",
}

_CLASS_MODIFIERS = {
    **_COMMON_MODIFIERS,
    "typing.final": "final",
}

_FUNCTION_MODIFIERS = {
    **_COMMON_MODIFIERS,
    "typing.overload": "overload",
    "builtins.staticmethod": "staticmethod",
    "builtins.classmethod": "classmethod",
    "builtins.property": "property",  # TODO: getter, setter, deleter
    "typing.final": "final",
    "typing.overload": "overload",
    "abc.abstractmethod": "abstractmethod",
}


def _modifier_from_decorator(
    decorator: PyType, modifiers: dict[str, str]
) -> Optional[str]:
    if entity := decorator.entity:
        if entity.full_name in modifiers:
            return modifiers[entity.full_name]

    return None


class PythonVisitor(PythonParserVisitor):
    """
    Visitor class for the Python grammar.

    This class performs two passes over the parse tree:
    - The first pass builds symbol tables.
    - The second pass resolves symbols and their types.
    """

    def __init__(self, context: PythonContext):
        self.context = context
        self.pass_num: int

        self._current_scope: SymbolTable = context.global_scope

        self._outer_symbols: list[_OuterSymbol] = []

        # Used for error reporting.
        self._current_ctx: Optional[ParserRuleContext] = None

        # Used when analyzing method definitions.
        self._parent_class: Optional[PyClass] = None
        # Used when analyzing parameter specifications.
        self._parent_function: Optional[PyFunction] = None
        # Used when analyzing function calls.
        self._called_function: Optional[PyFunctionType] = None

    # Public Interface
    # ================

    def first_pass(self, tree):
        """
        Visits the entire parse tree for the first pass.
        """
        self.pass_num = 1
        self.visit(tree)
        self._resolve_outer_symbols()
        self._check_public_symbols()

    def second_pass(self, tree):
        """
        Visits the entire parse tree for the second pass.
        """
        self.pass_num = 2
        with (
            set_forward_ref_evaluator(self._evaluate_forward_ref),
            set_error_reporter(self._report_error),
        ):
            self.visit(tree)

    # Helper Methods
    # ==============

    def _resolve_outer_symbols(self):
        """
        Resolves the global and nonlocal declarations in the module.
        """
        for symbol, scope in self._outer_symbols:
            if symbol.type is SymbolType.GLOBAL:
                # Resolve the global variable in the global scope.
                resolved = self.context.global_scope.get(symbol.name)

            elif symbol.type is SymbolType.NONLOCAL:
                # Resolve the nonlocal variable in an outer scope.
                resolved = scope.parent.lookup(symbol.name, globals=False)

            if resolved is not None and not resolved.is_outer():
                symbol.target = resolved

    def _check_public_symbols(self):
        """
        Checks whether global symbols are public or private.
        """
        # TODO: Respect the __all__ list.
        for symbol in self.context.global_scope.iter_symbols():
            if symbol.name.startswith("_"):
                symbol.public = False
            elif symbol.public is None:
                symbol.public = symbol.type is not SymbolType.IMPORTED

    def _evaluate_forward_ref(self, value: str) -> PyType:
        """
        Evaluates a forward reference string literal in type annotations.
        """
        # TODO: Parse the string literal as tree and resolve the type.
        if symbol := self._current_scope.lookup(value):
            return symbol.get_type()
        return PyType.ANY

    def _report_error(self, error: Exception):
        """
        Reports an error during semantic analysis and type checking.

        Args:
            error: The error to report.
        """
        if isinstance(error, SemanticError):
            if error.range is None and self._current_ctx is not None:
                error.set_context(self._current_ctx)

        else:
            import traceback

            traceback.print_exception(type(error), error, error.__traceback__)

        self.context.errors.append(error)

    @contextmanager
    def _wrap_errors(self, error_cls: type[Exception]) -> Iterator[None]:
        """
        Catches errors of a specific type and reports them.

        Args:
            error_cls: The error type to catch.
        """
        try:
            yield
        except error_cls as e:
            self._report_error(e)

    def _new_scope(
        self, name: str, scope_type: ScopeType, full_name: Optional[str] = None
    ) -> SymbolTable:
        parent_scope = self._current_scope
        if full_name is None:
            full_name = f"{parent_scope.full_name}.{name}"
        if parent_scope.scope_type is ScopeType.CLASS:
            parent_scope = parent_scope.parent

        return SymbolTable(name, scope_type, parent_scope, full_name=full_name)

    # Context Management
    # ==================

    @contextmanager
    def _set_parent_class(self, cls: PyClass) -> Iterator[None]:
        old_class = self._parent_class
        self._parent_class = cls
        try:
            yield
        finally:
            self._parent_class = old_class

    @contextmanager
    def _set_parent_function(self, func: PyFunction) -> Iterator[None]:
        old_function = self._parent_function
        self._parent_function = func
        try:
            yield
        finally:
            self._parent_function = old_function

    @contextmanager
    def _set_called_function(self, type: PyType) -> Iterator[Optional[PyFunctionType]]:
        func_type = type.get_callable_type()

        old_function = self._called_function
        self._called_function = func_type
        try:
            yield func_type
        finally:
            self._called_function = old_function

    # Symbol Definition and Access
    # ============================

    def _define_variable(
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
            scope = self._current_scope

        if name not in scope:
            entity = PyVariable(name, type=type)
            symbol = Symbol(SymbolType.VARIABLE, name, node, entity=entity)
            scope.define(symbol)
        else:
            symbol = scope[name]

        self.context.set_node_info(node, symbol=symbol)
        return symbol

    def _access_variable(self, name: str, node: TerminalNode) -> PyType:
        """
        Accesses a variable in the current scope.

        Args:
            name: The name of the variable.
            node: The terminal node where the variable is accessed.

        Returns:
            type: The type of the variable.
        """
        try:
            symbol = self._current_scope.lookup(name, raise_error=True)
            self.context.set_node_info(node, symbol=symbol)
            return symbol.get_type()

        except PySymbolNotFoundError as e:
            self._report_error(e.with_context(node))
            return PyType.ANY

    def _set_variable_type(
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

    def _define_attribute(
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
        if attr := on_type.get_attr(name):
            # If the attribute exists, the target is an attribute.
            symbol, type_ = attr
            if value_type is not None:
                type_ = self._set_variable_type(
                    symbol, value_type, override=override_type
                )
            self.context.set_node_info(node, symbol=symbol, type=type_)

        elif isinstance(on_type, PySelfType):
            # If the attribute does not exist, but the target is `self`, the attribute
            # is defined on the instance scope of the class.
            self._define_variable(
                name, node, type=value_type, scope=on_type.cls.instance_scope
            )

        else:
            # The attribute cannot be defined on the type.
            self.context.set_node_info(node, kind=TokenKind.FIELD)

    def _access_attribute(
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
        if attr := on_type.get_attr(name):
            symbol, type_ = attr
            self.context.set_node_info(node, symbol=symbol, type=type_)
            return type_

        else:
            self.context.set_node_info(node, kind=TokenKind.FIELD)
            return PyType.ANY

    def _access_func_kwarg(self, name: str, node: TerminalNode) -> PyType:
        """
        Accesses a keyword parameter in the called function.

        Args:
            name: The name of the keyword parameter.
            node: The terminal node where the keyword parameter is accessed.

        Returns:
            type: The type of the keyword parameter.
        """
        if self._called_function is not None:
            parameters = self._called_function.get_parameters()
            param = parameters.get_keyword(name)

            if param is not None:
                symbol = self._called_function.func.scope[param.name]
                type_ = param.get_type()

                if param.star == "**":
                    type_ = param.signature_type or PyType.ANY

                self.context.set_node_info(node, symbol=symbol, type=type_)
                return type_

        # The keyword parameter is not found.
        self.context.set_node_info(node, kind=TokenKind.VARIABLE)
        return PyType.ANY

    # Visitor Methods
    # ===============

    def aggregateResult(self, aggregate, nextResult):
        return aggregate or nextResult

    def visitTerminal(self, node: TerminalNode) -> Optional[str | PyType]:
        match node.getSymbol().type:
            case PythonParser.NAME:
                return self.visitName(node)

            case PythonParser.NONE:
                return PyType.NONE

            case PythonParser.ELLIPSIS:
                return PyType.ELLIPSIS

            case (
                PythonParser.FALSE
                | PythonParser.TRUE
                | PythonParser.STRING_LITERAL
                | PythonParser.BYTES_LITERAL
                | PythonParser.INTEGER
                | PythonParser.FLOAT_NUMBER
                | PythonParser.IMAG_NUMBER
            ):
                if self.pass_num == 2:
                    return self.visitLiteral(node)
                else:
                    return None

            case _:
                return super().visitTerminal(node)

    def visitName(self, node: TerminalNode) -> str:
        """
        Returns:
            name: The name of the node.
        """
        return node.getText()

    def visitLiteral(self, node: TerminalNode) -> PyType:
        """
        Returns:
            type: The type of the literal.
        """
        try:
            return PyLiteralType(literal_eval(node.getText()))
        except ValueError as e:
            self._report_error(PyTypeError(str(e)).with_context(node))
            return PyType.ANY

    def visitErrorNode(self, node: ErrorNode):
        if self.pass_num == 1:
            token = node.getSymbol()

            if (
                isinstance(parent_ctx := node.parentCtx, ParserRuleContext)
                and isinstance(error := parent_ctx.exception, RecognitionException)
                and error.offendingToken is not None
                and token.tokenIndex >= error.offendingToken.tokenIndex
            ):
                # The error has already been reported.
                return

            if token.tokenIndex == -1 and token.text.startswith("<missing "):
                self._report_error(PySyntaxError(token.text[1:-1], token))
            else:
                self.context.set_node_info(node, kind=TokenKind.ERROR)
                self._report_error(
                    PySyntaxError(f"unexpected token {token.text!r}", token)
                )

    @_visitor_guard(first_pass="default", second_pass="default")
    def visitChildren(self, ctx: ParserRuleContext):
        return super().visitChildren(ctx)

    def _visit_error_nodes(self, ctx: ParserRuleContext):
        if self.pass_num == 1:
            for child in ctx.children:
                if isinstance(child, ErrorNode):
                    self.visitErrorNode(child)

    def _handle_parser_error(self, ctx: ParserRuleContext):
        if self.pass_num == 1 and (error := ctx.exception):
            if isinstance(error, RecognitionException):
                if error.message and error.offendingToken is not None:
                    self._report_error(
                        PySyntaxError(error.message, error.offendingToken, ctx.stop)
                    )
            else:
                self._report_error(PySyntaxError(str(error)).with_context(ctx))

    # invalidBlock: INDENT statements DEDENT;
    @_visitor_guard(error_action="visit")
    def visitInvalidBlock(self, ctx: PythonParser.InvalidBlockContext):
        if self.pass_num == 1:
            self._report_error(PySyntaxError("unexpected indent").with_context(ctx))

        return self.visitChildren(ctx)

    @_visitor_guard(second_pass="skip", error_action="visit")
    def visitInvalidToken(self, ctx: PythonParser.InvalidTokenContext):
        token = ctx.start
        self._report_error(
            PySyntaxError(f"unexpected token {token.text!r}", token).with_context(ctx)
        )

    # singleTarget ':' expression ('=' assignmentRhs)?
    @_both_passes
    def visitAnnotatedAssignment(self, ctx: PythonParser.AnnotatedAssignmentContext):
        annotation = self.visitExpression(ctx.expression())

        if assignment_rhs := ctx.assignmentRhs():
            self.visitAssignmentRhs(assignment_rhs)

        if self.pass_num == 1:
            self.visitSingleTarget(ctx.singleTarget(), define=True)

        elif self.pass_num == 2:
            annotation: PyType
            self.visitSingleTarget(
                ctx.singleTarget(),
                value_type=annotation.get_annotated_type(),
            )

    # (starTargets '=')+ assignmentRhs
    @_type_check
    def visitStarredAssignment(self, ctx: PythonParser.StarredAssignmentContext):
        type_ = self.visitAssignmentRhs(ctx.assignmentRhs())

        for star_targets in ctx.starTargets():
            self.visitStarTargets(star_targets, value_type=type_)

    # returnStmt: 'return' starExpressions?;
    @_type_check
    def visitReturnStmt(self, ctx: PythonParser.ReturnStmtContext):
        type_ = self.visitStarExpressions(ctx.starExpressions())

        if self._current_scope.scope_type is ScopeType.LOCAL:
            self._parent_function.returned_types.append(type_)

        else:
            self._report_error(PySyntaxError("'return' outside function"))

    # globalStmt: 'global' NAME (',' NAME)*;
    @_first_pass_only
    def visitGlobalStmt(self, ctx: PythonParser.GlobalStmtContext):
        for name_node in ctx.NAME():
            name = self.visitName(name_node)

            symbol = Symbol(SymbolType.GLOBAL, name, name_node)
            self.context.set_node_info(name_node, symbol=symbol)

            with self._wrap_errors(PyDuplicateSymbolError):
                self._current_scope.define(symbol)

            self._outer_symbols.append(_OuterSymbol(symbol, self._current_scope))

    # nonlocalStmt: 'nonlocal' NAME (',' NAME)*;
    @_first_pass_only
    def visitNonlocalStmt(self, ctx: PythonParser.NonlocalStmtContext):
        for name_node in ctx.NAME():
            name = self.visitName(name_node)

            symbol = Symbol(SymbolType.NONLOCAL, name, name_node)
            self.context.set_node_info(name_node, symbol=symbol)

            with self._wrap_errors(PyDuplicateSymbolError):
                self._current_scope.define(symbol)

            self._outer_symbols.append(_OuterSymbol(symbol, self._current_scope))

    # importFrom
    #   : 'from' ('.' | '...')* dottedName 'import' importFromTargets
    #   | 'from' ('.' | '...')+ 'import' importFromTargets;
    @_first_pass_only
    def visitImportFrom(self, ctx: PythonParser.ImportFromContext):
        num_dots = len(ctx.DOT()) + 3 * len(ctx.ELLIPSIS())
        relative = num_dots - 1 if num_dots > 0 else None

        if dotted_name := ctx.dottedName():
            path, _ = self.visitDottedName(dotted_name)
        else:
            path = []

        targets = self.visitImportFromTargets(ctx.importFromTargets())

        self.context.imports.append(PyImportFrom(path, relative, targets, ctx))

    # importFromTargets
    #   : '(' importFromAsNames ','? ')'
    #   | importFromAsNames
    #   | '*';
    @_first_pass_only
    def visitImportFromTargets(
        self, ctx: PythonParser.ImportFromTargetsContext
    ) -> PyImportFromTargets:
        """
        Returns:
            targets: The import-from targets.
        """
        if as_names := ctx.importFromAsNames():
            return PyImportFromTargets(self.visitImportFromAsNames(as_names))
        else:
            return PyImportFromTargets(None)

    # importFromAsNames: importFromAsName (',' importFromAsName)*;
    @_first_pass_only
    def visitImportFromAsNames(
        self, ctx: PythonParser.ImportFromAsNamesContext
    ) -> list[PyImportFromAsName]:
        """
        Returns:
            as_names: The list of import names and their aliases.
        """
        return [
            self.visitImportFromAsName(as_name) for as_name in ctx.importFromAsName()
        ]

    # importFromAsName: NAME ('as' NAME)?;
    @_first_pass_only
    def visitImportFromAsName(
        self, ctx: PythonParser.ImportFromAsNameContext
    ) -> PyImportFromAsName:
        """
        Returns:
            as_name: The name of the import and its alias.
        """
        name_node = ctx.NAME(0)
        name = self.visitName(name_node)

        if alias_node := ctx.NAME(1):
            alias = self.visitName(alias_node)

            # The alias is defined in the current scope.
            # The import form `from Y import X as X` (a redundant symbol alias)
            # re-exports symbol `X`.
            # https://typing.readthedocs.io/en/latest/spec/distributing.html#import-conventions
            symbol = Symbol(
                SymbolType.IMPORTED, alias, alias_node, public=name == alias
            )
            self.context.set_node_info(name_node, symbol=symbol)
            self.context.set_node_info(alias_node, symbol=symbol)

        else:
            alias = None

            # The name is defined in the current scope.
            symbol = Symbol(SymbolType.IMPORTED, name, name_node)
            self.context.set_node_info(name_node, symbol=symbol)

        with self._wrap_errors(PyDuplicateSymbolError):
            self._current_scope.define(symbol)

        return PyImportFromAsName(name, alias, symbol)

    # dottedAsName: dottedName ('as' NAME)?;
    @_first_pass_only
    def visitDottedAsName(self, ctx: PythonParser.DottedAsNameContext):
        if alias_node := ctx.NAME():
            path, _ = self.visitDottedName(ctx.dottedName())
            alias = self.visitName(alias_node)

            # The as-name is defined in the current scope.
            # The import form `import X as X` (a redundant module alias) re-exports
            # symbol `X`.
            # https://typing.readthedocs.io/en/latest/spec/distributing.html#import-conventions
            symbol = Symbol(
                SymbolType.IMPORTED, alias, alias_node, public=path == [alias]
            )
            # The name always refers to a module.
            self.context.set_node_info(alias_node, kind=TokenKind.MODULE, symbol=symbol)

            with self._wrap_errors(PyDuplicateSymbolError):
                self._current_scope.define(symbol)

        else:
            path, symbol = self.visitDottedName(ctx.dottedName(), define=True)
            alias = None

        self.context.imports.append(PyImportName(path, alias, symbol, ctx.parentCtx))

    # dottedName: dottedName '.' NAME | NAME;
    @_first_pass_only
    def visitDottedName(
        self, ctx: PythonParser.DottedNameContext, *, define: bool = False
    ) -> tuple[list[str], Optional[Symbol]]:
        """
        Args:
            define: Whether to define the name in the current scope.
                True if the dotted name is part of an import-name statement, and no
                as-name is provided.

        Returns:
            path: The list of module names in the dotted name.
            symbol: The top-level module symbol, if defined.
        """
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        # The name always refers to a module.
        self.context.set_node_info(name_node, kind=TokenKind.MODULE)

        if dotted_name := ctx.dottedName():
            path, symbol = self.visitDottedName(dotted_name, define=define)
            path.append(name)
            return path, symbol

        else:
            symbol = None
            if define:
                # The top-level module is defined in the current scope.
                symbol = Symbol(SymbolType.IMPORTED, name, name_node)
                self.context.set_node_info(name_node, symbol=symbol)

                with self._wrap_errors(PyDuplicateSymbolError):
                    self._current_scope.define(symbol)

            return [name], symbol

    # block
    #   : NEWLINE INDENT statements DEDENT
    #   | simpleStmts;
    @_both_passes
    def visitBlock(self, ctx: PythonParser.BlockContext):
        return self.visitChildren(ctx)

    # decorators: ('@' namedExpression NEWLINE)+;
    @_type_check
    def visitDecorators(self, ctx: PythonParser.DecoratorsContext) -> list[PyType]:
        return [self.visitNamedExpression(expr) for expr in ctx.namedExpression()]

    # classDef
    #   : decorators? 'class' NAME typeParams? ('(' arguments? ')')?
    #       ':' block;
    @_both_passes
    def visitClassDef(self, ctx: PythonParser.ClassDefContext):
        if decorators := ctx.decorators():
            decorators = self.visitDecorators(decorators)
        else:
            decorators = []

        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if arguments_node := ctx.arguments():
            arguments = self.visitArguments(arguments_node)

        if self.pass_num == 1:
            scope = self._new_scope(name, ScopeType.CLASS)
            self.context.set_node_scope(ctx.block(), scope)

            entity = PyClass(name, scope)
            self.context.entities[ctx] = entity

            symbol = Symbol(SymbolType.CLASS, name, name_node, entity=entity)
            self.context.set_node_info(name_node, kind=TokenKind.CLASS, symbol=symbol)

            with self._wrap_errors(PyDuplicateSymbolError):
                self._current_scope.define(symbol)

        elif self.pass_num == 2:
            entity: PyClass = self.context.entities[ctx]

            # Handle class decorators.
            entity.decorators = decorators
            for decorator in decorators:
                if modifier := _modifier_from_decorator(decorator, _CLASS_MODIFIERS):
                    entity.set_modifier(modifier)

            # Handle class arguments.
            if arguments_node:
                entity.arguments = arguments
                entity.parse_arguments(arguments_node)
            else:
                entity.arguments = PyArguments()

        with self._set_parent_class(entity):
            if type_params := ctx.typeParams():
                self.visitTypeParams(type_params)

            self.visitBlock(ctx.block())

    # functionDef
    #   : decorators? 'async'? 'def' NAME typeParams? '(' parameters? ')'
    #       ('->' expression)? ':' block;
    @_both_passes
    def visitFunctionDef(self, ctx: PythonParser.FunctionDefContext):
        if decorators := ctx.decorators():
            decorators = self.visitDecorators(decorators)
        else:
            decorators = []

        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if self.pass_num == 1:
            if self._current_scope.scope_type is ScopeType.CLASS:
                parent_cls = self._parent_class
            else:
                parent_cls = None

            scope = self._new_scope(f"{name}.<locals>", ScopeType.LOCAL)
            self.context.set_node_scope(ctx.block(), scope)

            entity = PyFunction(name, scope, cls=parent_cls)
            self.context.entities[ctx] = entity

            if (symbol := self._current_scope.get(name)) and isinstance(
                prev_func := symbol.entity, PyFunction
            ):
                # Handle function overloading.
                prev_func.overloads.append(entity)
            else:
                symbol = Symbol(SymbolType.FUNCTION, name, name_node, entity=entity)
                with self._wrap_errors(PyDuplicateSymbolError):
                    self._current_scope.define(symbol)

            self.context.set_node_info(
                name_node, kind=TokenKind.FUNCTION, symbol=symbol
            )

        elif self.pass_num == 2:
            entity: PyFunction = self.context.entities[ctx]
            entity.decorators = decorators

            for decorator in decorators:
                if modifier := _modifier_from_decorator(decorator, _FUNCTION_MODIFIERS):
                    entity.set_modifier(modifier)

        if expression := ctx.expression():
            annotation = self.visitExpression(expression)

            if self.pass_num == 2:
                annotation: PyType
                entity.return_type = annotation.get_annotated_type()

        with self._set_parent_function(entity):
            if type_params := ctx.typeParams():
                self.visitTypeParams(type_params)

            if parameters_node := ctx.parameters():
                parameters = self.visitParameters(parameters_node)

                if self.pass_num == 1:
                    entity.parameters.extend(parameters)

                elif self.pass_num == 2:
                    if entity.is_method and (
                        bound_param := entity.parameters.get_bound_param()
                    ):
                        if entity.has_modifier("classmethod"):
                            # Class methods have a `cls` parameter.
                            if bound_param.type is None:
                                bound_param.type = entity.cls.get_type()

                        elif not entity.has_modifier("staticmethod"):
                            # Instance methods have a `self` parameter.
                            if bound_param.type is None:
                                bound_param.type = entity.cls.get_self_type()

            self.visitBlock(ctx.block())

            if self.pass_num == 2 and entity.return_type is None:
                # Infer the return type of the function.
                returned_types = [t.get_inferred_type() for t in entity.returned_types]
                returned_types.append(PyType.NONE)
                entity.return_type = PyUnionType.from_items(returned_types)

    # parameters
    #   : slashNoDefault (',' paramNoDefault)* (',' paramWithDefault)*
    #       (',' starEtc?)?
    #   | slashWithDefault (',' paramWithDefault)*
    #       (',' starEtc?)?
    #   | paramNoDefault (',' paramNoDefault)* (',' paramWithDefault)*
    #       (',' starEtc?)?
    #   | paramWithDefault (',' paramWithDefault)*
    #       (',' starEtc?)?
    #   | starEtc;
    @_both_passes
    def visitParameters(self, ctx: PythonParser.ParametersContext) -> list[PyParameter]:
        parameters: list[PyParameter] = []

        if node := ctx.slashNoDefault():
            parameters.extend(self.visitSlashNoDefault(node))

        elif node := ctx.slashWithDefault():
            parameters.extend(self.visitSlashWithDefault(node))

        for node in ctx.paramNoDefault():
            parameters.append(self.visitParamNoDefault(node))

        for node in ctx.paramWithDefault():
            parameters.append(self.visitParamWithDefault(node))

        if node := ctx.starEtc():
            parameters.extend(self.visitStarEtc(node))

        return parameters

    # slashNoDefault: paramNoDefault (',' paramNoDefault)* ',' '/';
    @_both_passes
    def visitSlashNoDefault(
        self, ctx: PythonParser.SlashNoDefaultContext
    ) -> list[PyParameter]:
        parameters: list[PyParameter] = []

        for node in ctx.paramNoDefault():
            parameters.append(self.visitParamNoDefault(node))

        if self.pass_num == 1:
            for param in parameters:
                param.posonly = True

        return parameters

    # slashWithDefault
    #   : paramNoDefault (',' paramNoDefault)* (',' paramWithDefault)+ ',' '/'
    #   | paramWithDefault (',' paramWithDefault)* ',' '/';
    @_both_passes
    def visitSlashWithDefault(
        self, ctx: PythonParser.SlashWithDefaultContext
    ) -> list[PyParameter]:
        parameters: list[PyParameter] = []

        for node in ctx.paramNoDefault():
            parameters.append(self.visitParamNoDefault(node))

        for node in ctx.paramWithDefault():
            parameters.append(self.visitParamWithDefault(node))

        if self.pass_num == 1:
            for param in parameters:
                param.posonly = True

        return parameters

    # starEtc
    #   : '*' (paramNoDefault | paramNoDefaultStarAnnotation)
    #       (',' paramMaybeDefault)* (',' kwds?)?
    #   | '*' (',' paramMaybeDefault)+ (',' kwds?)?
    #   | kwds;
    @_both_passes
    def visitStarEtc(self, ctx: PythonParser.StarEtcContext) -> list[PyParameter]:
        parameters: list[PyParameter] = []

        if node := ctx.paramNoDefault():
            parameters.append(param := self.visitParamNoDefault(node))

            if self.pass_num == 1:
                param.star = "*"

            elif self.pass_num == 2:
                # *args: T translates to args: tuple[T, ...]
                type_args = param.signature_type and (param.signature_type,)
                param.type = PyInstanceType.from_stub("builtins.tuple", type_args)

        elif node := ctx.paramNoDefaultStarAnnotation():
            parameters.append(param := self.visitParamNoDefaultStarAnnotation(node))

        for node in ctx.paramMaybeDefault():
            parameters.append(param := self.visitParamMaybeDefault(node))

            if self.pass_num == 1:
                param.kwonly = True

        if node := ctx.kwds():
            parameters.append(self.visitKwds(node))

        return parameters

    # kwds: '**' paramNoDefault ','?;
    @_both_passes
    def visitKwds(self, ctx: PythonParser.KwdsContext) -> PyParameter:
        param = self.visitParamNoDefault(ctx.paramNoDefault())

        if self.pass_num == 1:
            param.star = "**"

        elif self.pass_num == 2:
            # **kwargs: V translates to kwargs: dict[str, V]
            type_args = (
                PyInstanceType.from_stub("builtins.str"),
                param.signature_type or PyType.ANY,
            )
            param.type = PyInstanceType.from_stub("builtins.dict", type_args)

        return param

    # paramNoDefault: param;
    @_both_passes
    def visitParamNoDefault(
        self, ctx: PythonParser.ParamNoDefaultContext
    ) -> PyParameter:
        return self.visitParam(ctx.param())

    # paramNoDefaultStarAnnotation: paramStarAnnotation;
    @_both_passes
    def visitParamNoDefaultStarAnnotation(
        self, ctx: PythonParser.ParamNoDefaultStarAnnotationContext
    ) -> PyParameter:
        return self.visitParamStarAnnotation(ctx.paramStarAnnotation())

    # paramWithDefault: param default;
    @_both_passes
    def visitParamWithDefault(
        self, ctx: PythonParser.ParamWithDefaultContext
    ) -> PyParameter:
        param = self.visitParam(ctx.param())

        default = self.visitDefault(default_node := ctx.default())

        if self.pass_num == 1:
            param.default = default_node

        elif self.pass_num == 2:
            if param.type is None:
                param.type = default.get_inferred_type()

        return param

    # paramMaybeDefault: param default?;
    @_both_passes
    def visitParamMaybeDefault(
        self, ctx: PythonParser.ParamMaybeDefaultContext
    ) -> PyParameter:
        param = self.visitParam(ctx.param())

        if default_node := ctx.default():
            default = self.visitDefault(default_node)

            if self.pass_num == 1:
                param.default = default_node

            elif self.pass_num == 2:
                if param.type is None:
                    param.type = default.get_inferred_type()

        return param

    # param: NAME annotation?;
    @_both_passes
    def visitParam(self, ctx: PythonParser.ParamContext) -> PyParameter:
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        annotation_node = ctx.annotation()

        if self.pass_num == 1:
            param = PyParameter(name, annotation=annotation_node)
            self.context.entities[ctx] = param

            symbol = Symbol(SymbolType.PARAMETER, name, name_node, entity=param)
            self.context.set_node_info(
                name_node, kind=TokenKind.VARIABLE, symbol=symbol
            )

            with self._wrap_errors(PyDuplicateSymbolError):
                self._parent_function.scope.define(symbol)

        else:
            param: PyParameter = self.context.entities[ctx]

        if annotation_node:
            annotation = self.visitAnnotation(annotation_node)

            if self.pass_num == 2:
                if param.star is not None:
                    param.signature_type = annotation
                else:
                    param.type = annotation

        return param

    # paramStarAnnotation: NAME starAnnotation;
    @_both_passes
    def visitParamStarAnnotation(
        self, ctx: PythonParser.ParamStarAnnotationContext
    ) -> PyParameter:
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        annotation_node = ctx.starAnnotation()

        if self.pass_num == 1:
            param = PyParameter(name, star="*", star_annotation=annotation_node)

            symbol = Symbol(SymbolType.PARAMETER, name, name_node, entity=param)
            self.context.set_node_info(
                name_node, kind=TokenKind.VARIABLE, symbol=symbol
            )

            with self._wrap_errors(PyDuplicateSymbolError):
                self._parent_function.scope.define(symbol)

        else:
            param = self.context.entities[ctx]

        annotation = self.visitStarAnnotation(annotation_node)

        if self.pass_num == 2:
            param.signature_type = annotation

        return param

    # annotation: ':' expression;
    @_type_check
    def visitAnnotation(self, ctx: PythonParser.AnnotationContext) -> PyType:
        annotation: PyType = self.visitExpression(ctx.expression())
        return annotation.get_annotated_type()

    # starAnnotation: ':' starredExpression;
    @_type_check
    def visitStarAnnotation(self, ctx: PythonParser.StarAnnotationContext) -> PyType:
        annotation: PyType = self.visitStarredExpression(ctx.starredExpression())
        return PyType.ANY  # TODO: star-annotated parameters

    # default: '=' expression;
    @_type_check
    def visitDefault(self, ctx: PythonParser.DefaultContext) -> PyType:
        return self.visitExpression(ctx.expression())

    # forStmt
    #   : 'async'? 'for' starTargets 'in' starExpressions ':' block elseBlock?;
    @_type_check
    def visitForStmt(self, ctx: PythonParser.ForStmtContext):
        iter_type = self.visitStarExpressions(ctx.starExpressions())

        self.visitStarTargets(
            ctx.starTargets(),
            value_type=iter_type.get_iterated_type(is_async=ctx.ASYNC() is not None),
        )

        self.visitBlock(ctx.block())

        if node := ctx.elseBlock():
            self.visitElseBlock(node)

    # exceptBlock
    #   : 'except' (expression ('as' NAME)?)? ':' block;
    @_both_passes
    def visitExceptBlock(self, ctx: PythonParser.ExceptBlockContext):
        if expression := ctx.expression():
            type_ = self.visitExpression(expression)
        else:
            type_ = PyType.ANY

        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 1:
                self._define_variable(name, name_node)

            elif self.pass_num == 2:
                symbol = self._current_scope[name]
                if isinstance(type_, PyClassType):
                    self._set_variable_type(symbol, type_.cls.get_instance_type())

        self.visitBlock(ctx.block())

    # exceptStarBlock
    #   : 'except' '*' expression ('as' NAME)? ':' block;
    @_both_passes
    def visitExceptStarBlock(self, ctx: PythonParser.ExceptStarBlockContext):
        self.visitExpression(ctx.expression())

        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 1:
                self._define_variable(name, name_node)

        self.visitBlock(ctx.block())

    # TODO: expressions

    # expression
    #   : logical 'if' logical 'else' expression
    #   | logical
    #   | lambdef;
    @_type_check
    def visitExpression(self, ctx: PythonParser.ExpressionContext) -> PyType:
        if ctx.IF():
            condition = self.visitLogical(ctx.logical(1))
            if_true = self.visitLogical(ctx.logical(0))
            if_false = self.visitExpression(ctx.expression())

            match condition.can_be_truthy(), condition.can_be_falsy():
                case True, True:
                    return PyUnionType.from_items((if_true, if_false))
                case True, False:
                    return if_true
                case False, True:
                    return if_false
                case False, False:
                    return PyType.NEVER

        elif node := ctx.lambdef():
            return self.visitLambdef(node)

        else:
            return self.visitLogical(ctx.logical(0))

    # starExpressions
    #   : starExpression (',' starExpression)* ','?;
    @_type_check
    def visitStarExpressions(self, ctx: PythonParser.StarExpressionsContext) -> PyType:
        if ctx.COMMA():
            # If at least one comma is present, the result is a tuple.
            return PyTupleType.from_starred(
                [self.visitStarExpression(node) for node in ctx.starExpression()]
            )

        else:
            # Otherwise, the result is the type of the single expression. Note that the
            # expression must not be a starred expression in this case.
            type_ = self.visitStarExpression(node := ctx.starExpression(0))
            if isinstance(type_, PyUnpack):
                self._report_error(
                    PySyntaxError(
                        "Starred expressions are not allowed here",
                    ).with_context(node)
                )
                return PyType.ANY

            return type_

    # starExpression
    #   : '*' bitwise
    #   | expression;
    @_type_check
    def visitStarExpression(self, ctx: PythonParser.StarExpressionContext) -> PyType:
        if ctx.STAR():
            return PyUnpack(self.visitBitwise(ctx.bitwise()))
        else:
            return self.visitExpression(ctx.expression())

    # starNamedExpressions
    #   : starNamedExpression (',' starNamedExpression)* ','?;
    @_type_check
    def visitStarNamedExpressions(
        self, ctx: PythonParser.StarNamedExpressionsContext
    ) -> list[PyType]:
        """
        Note:
            We do not return a tuple type here in contrast to `visitStarExpressions`,
            because the result is not necessarily a tuple. This constuct is also used
            in list and set displays.
        """
        return [
            self.visitStarNamedExpression(expr) for expr in ctx.starNamedExpression()
        ]

    # starNamedExpression
    #   : '*' bitwise
    #   | namedExpression;
    @_type_check
    def visitStarNamedExpression(
        self, ctx: PythonParser.StarNamedExpressionContext
    ) -> PyType:
        if ctx.STAR():
            return PyUnpack(self.visitBitwise(ctx.bitwise()))
        else:
            return self.visitNamedExpression(ctx.namedExpression())

    # assignmentExpression: NAME ':=' expression;
    @_both_passes
    def visitAssignmentExpression(
        self, ctx: PythonParser.AssignmentExpressionContext
    ) -> Optional[PyType]:
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        type_ = self.visitExpression(ctx.expression())

        # According to PEP 572, an assignment expression occurring in comprehensions
        # binds the target in the containing scope, honoring a global or nonlocal
        # declaration if present.
        scope = self._current_scope
        while scope.scope_type is ScopeType.COMPREHENSION:
            assert (scope := scope.parent) is not None

        if self.pass_num == 1:
            self._define_variable(name, name_node, scope=scope)

        elif self.pass_num == 2:
            symbol = scope[name]
            return self._set_variable_type(symbol, type_.get_inferred_type())

    # namedExpression: assignmentExpression | expression;
    @_type_check
    def visitNamedExpression(self, ctx: PythonParser.NamedExpressionContext) -> PyType:
        if node := ctx.assignmentExpression():
            return self.visitAssignmentExpression(node)
        else:
            return self.visitExpression(ctx.expression())

    # logical
    #   : 'not' logical
    #   | logical 'and' logical
    #   | logical 'or' logical
    #   | comparison;
    @_type_check
    def visitLogical(self, ctx: PythonParser.LogicalContext) -> PyType:
        if ctx.NOT():
            type_ = self.visitLogical(ctx.logical(0))

            return type_.get_inversion_type()

        elif ctx.AND():
            left_type = self.visitLogical(ctx.logical(0))
            right_type = self.visitLogical(ctx.logical(1))

            return left_type.get_conjunction_type(right_type)

        elif ctx.OR():
            left_type = self.visitLogical(ctx.logical(0))
            right_type = self.visitLogical(ctx.logical(1))

            return left_type.get_disjunction_type(right_type)

        else:
            return self.visitComparison(ctx.comparison())

    # comparison: bitwise compareOpBitwisePair*;
    @_type_check
    def visitComparison(self, ctx: PythonParser.ComparisonContext) -> PyType:
        type_ = self.visitBitwise(ctx.bitwise())

        pairs = ctx.compareOpBitwisePair()
        for pair in pairs:
            self.visitCompareOpBitwisePair(pair)

        if pairs:
            # TODO: The result of a comparison is not necessarily a boolean for
            # custom comparison methods.
            return PyInstanceType.from_stub("builtins.bool")
        else:
            return type_

    # TODO: bitwise, arithmetic

    # awaitPrimary
    #   : 'await' primary
    #   | primary
    #   | 'await' invalidPrimary
    #   | invalidPrimary;
    @_type_check
    def visitAwaitPrimary(self, ctx: PythonParser.AwaitPrimaryContext) -> PyType:
        if ctx.invalidPrimary():
            self.visitInvalidPrimary(ctx.invalidPrimary())
            return PyType.ANY

        type_ = self.visitPrimary(ctx.primary())

        if ctx.AWAIT():
            return type_.get_awaited_type()
        else:
            return type_

    # primary
    #   : primary '.' NAME
    #   | primary genexp
    #   | primary '(' arguments? ')'
    #   | primary '[' slices ']'
    #   | atom;
    @_type_check
    def visitPrimary(self, ctx: PythonParser.PrimaryContext) -> PyType:
        if atom := ctx.atom():
            return self.visitAtom(atom)

        type_ = self.visitPrimary(ctx.primary())

        if name_node := ctx.NAME():
            # Attribute access
            name = self.visitName(name_node)

            return self._access_attribute(type_, name, name_node)

        elif slices := ctx.slices():
            # Subscription
            key_type = self.visitSlices(slices)

            return type_.get_subscripted_type(key_type)

        else:
            # Function call
            with self._set_called_function(type_) as func:
                if genexp := ctx.genexp():
                    arg_type = self.visitGenexp(genexp)
                    args = PyArguments([arg_type])
                elif arguments := ctx.arguments():
                    args = self.visitArguments(arguments)
                else:
                    args = PyArguments()

                if func is not None:
                    self.context.set_function_call(ctx, func, args)

            return type_.get_return_type(args)

    # invalidPrimary
    #   : primary '.'
    #   | primary '[' ']';
    @_both_passes
    def visitInvalidPrimary(self, ctx: PythonParser.InvalidPrimaryContext):
        if self.pass_num == 1:
            if node := ctx.DOT():
                self._report_error(
                    PySyntaxError("expected attribute name after '.'", node.getSymbol())
                )
            else:
                self._report_error(
                    PySyntaxError(
                        "expected slices inside '[]'",
                        ctx.LSQB().getSymbol(),
                        ctx.RSQB().getSymbol(),
                    )
                )

        self.visitPrimary(ctx.primary())

    # slices: slice (',' slice)* ','?;
    @_type_check
    def visitSlices(self, ctx: PythonParser.SlicesContext) -> PyType:
        if ctx.COMMA():
            # If at least one comma is present, the result is a tuple.
            return PyTupleType.from_starred(
                [self.visitSlice(node) for node in ctx.slice_()]
            )

        else:
            # Otherwise, the result may be a tuple or a single slice, depending on
            # whether the slice is a starred expression.
            type_ = self.visitSlice(ctx.slice_(0))
            if isinstance(type_, PyUnpack):
                return PyTupleType.from_starred((type_,))
            return type_

    # slice
    #   : startExpr=expression? ':' stopExpr=expression? (':' stepExpr=expression)?
    #   | namedExpression
    #   | starredExpression;
    @_type_check
    def visitSlice(self, ctx: PythonParser.SliceContext) -> PyType:
        if node := ctx.namedExpression():
            return self.visitNamedExpression(node)

        elif node := ctx.starredExpression():
            return self.visitStarredExpression(node)

        else:
            type_args = tuple(
                self.visitExpression(node) if node else PyType.NONE
                for node in (ctx.startExpr, ctx.stopExpr, ctx.stepExpr)
            )

            return PyInstanceType.from_stub("builtins.slice", type_args)

    # atom
    #   : NAME
    #   | 'True'
    #   | 'False'
    #   | 'None'
    #   | strings
    #   | number
    #   | (tuple | group | genexp)
    #   | (list | listcomp)
    #   | (dict | set | dictcomp | setcomp)
    #   | '...';
    @_both_passes
    def visitAtom(self, ctx: PythonParser.AtomContext) -> Optional[PyType]:
        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 2:
                return self._access_variable(name, name_node)

        else:
            return self.visitChildren(ctx)

    # lambdef
    #   : 'lambda' lambdaParameters? ':' expression;
    @_both_passes
    def visitLambdef(self, ctx: PythonParser.LambdefContext) -> Optional[PyType]:
        if self.pass_num == 1:
            scope = self._new_scope("<lambda>.<locals>", ScopeType.LOCAL)
            self.context.set_node_scope(ctx.expression(), scope)

            entity = PyLambda(scope)
            self.context.entities[ctx] = entity
        else:
            entity: PyLambda = self.context.entities[ctx]

        with self._set_parent_function(entity):
            if node := ctx.lambdaParameters():
                parameters = self.visitLambdaParameters(node)

                if self.pass_num == 1:
                    entity.parameters.extend(parameters)

        type_ = self.visitExpression(ctx.expression())

        if self.pass_num == 2:
            type_: PyType
            entity.return_type = type_.get_inferred_type()
            return entity.get_type()

    # lambdaParameters
    #   : lambdaSlashNoDefault (',' lambdaParamNoDefault)* (',' lambdaParamWithDefault)*
    #       (',' lambdaStarEtc?)?
    #   | lambdaSlashWithDefault (',' lambdaParamWithDefault)*
    #       (',' lambdaStarEtc?)?
    #   | lambdaParamNoDefault (',' lambdaParamNoDefault)* (',' lambdaParamWithDefault)*
    #       (',' lambdaStarEtc?)?
    #   | lambdaParamWithDefault (',' lambdaParamWithDefault)*
    #       (',' lambdaStarEtc?)?
    #   | lambdaStarEtc;
    @_both_passes
    def visitLambdaParameters(
        self, ctx: PythonParser.LambdaParametersContext
    ) -> list[PyParameter]:
        parameters: list[PyParameter] = []

        if node := ctx.lambdaSlashNoDefault():
            parameters.extend(self.visitLambdaSlashNoDefault(node))

        elif node := ctx.lambdaSlashWithDefault():
            parameters.extend(self.visitLambdaSlashWithDefault(node))

        for node in ctx.lambdaParamNoDefault():
            parameters.append(self.visitLambdaParamNoDefault(node))

        for node in ctx.lambdaParamWithDefault():
            parameters.append(self.visitLambdaParamWithDefault(node))

        if node := ctx.lambdaStarEtc():
            parameters.extend(self.visitLambdaStarEtc(node))

        return parameters

    # lambdaSlashNoDefault
    #   : lambdaParamNoDefault (',' lambdaParamNoDefault)* ',' '/';
    @_both_passes
    def visitLambdaSlashNoDefault(
        self, ctx: PythonParser.LambdaSlashNoDefaultContext
    ) -> list[PyParameter]:
        parameters: list[PyParameter] = []

        for node in ctx.lambdaParamNoDefault():
            parameters.append(self.visitLambdaParamNoDefault(node))

        if self.pass_num == 1:
            for param in parameters:
                param.posonly = True

        return parameters

    # lambdaSlashWithDefault
    #   : lambdaParamNoDefault (',' lambdaParamNoDefault)* (',' lambdaParamWithDefault)+ ',' '/'
    #   | lambdaParamWithDefault (',' lambdaParamWithDefault)* ',' '/';
    @_both_passes
    def visitLambdaSlashWithDefault(
        self, ctx: PythonParser.LambdaSlashWithDefaultContext
    ) -> list[PyParameter]:
        parameters: list[PyParameter] = []

        for node in ctx.lambdaParamNoDefault():
            parameters.append(self.visitLambdaParamNoDefault(node))

        for node in ctx.lambdaParamWithDefault():
            parameters.append(self.visitLambdaParamWithDefault(node))

        if self.pass_num == 1:
            for param in parameters:
                param.posonly = True

        return parameters

    # lambdaStarEtc
    #   : '*' lambdaParamNoDefault (',' lambdaParamMaybeDefault)* (',' lambdaKwds?)?
    #   | '*' (',' lambdaParamMaybeDefault)+ (',' lambdaKwds?)?
    #   | lambdaKwds;
    @_both_passes
    def visitLambdaStarEtc(
        self, ctx: PythonParser.LambdaStarEtcContext
    ) -> list[PyParameter]:
        parameters: list[PyParameter] = []

        if node := ctx.lambdaParamNoDefault():
            parameters.append(param := self.visitLambdaParamNoDefault(node))

            if self.pass_num == 1:
                param.star = "*"

            elif self.pass_num == 2:
                param.type = PyInstanceType.from_stub("builtins.tuple")

        for node in ctx.lambdaParamMaybeDefault():
            parameters.append(param := self.visitLambdaParamMaybeDefault(node))

            if self.pass_num == 1:
                param.kwonly = True

        if node := ctx.lambdaKwds():
            parameters.append(self.visitLambdaKwds(node))

        return parameters

    # lambdaKwds: '**' lambdaParamNoDefault ','?;
    @_both_passes
    def visitLambdaKwds(self, ctx: PythonParser.LambdaKwdsContext) -> PyParameter:
        param = self.visitLambdaParamNoDefault(ctx.lambdaParamNoDefault())

        if self.pass_num == 1:
            param.star = "**"

        elif self.pass_num == 2:
            type_args = (PyInstanceType.from_stub("builtins.str"), PyType.ANY)
            param.type = PyInstanceType.from_stub("builtins.dict", type_args)

        return param

    # lambdaParamNoDefault: lambdaParam;
    @_both_passes
    def visitLambdaParamNoDefault(
        self, ctx: PythonParser.LambdaParamNoDefaultContext
    ) -> PyParameter:
        return self.visitLambdaParam(ctx.lambdaParam())

    # lambdaParamWithDefault: lambdaParam default;
    @_both_passes
    def visitLambdaParamWithDefault(
        self, ctx: PythonParser.LambdaParamWithDefaultContext
    ) -> PyParameter:
        param = self.visitLambdaParam(ctx.lambdaParam())

        default = self.visitDefault(ctx.default())

        if self.pass_num == 1:
            param.default = ctx.default()

        elif self.pass_num == 2:
            param.type = default.get_inferred_type()

        return param

    # lambdaParamMaybeDefault: lambdaParam default?;
    @_both_passes
    def visitLambdaParamMaybeDefault(
        self, ctx: PythonParser.LambdaParamMaybeDefaultContext
    ) -> PyParameter:
        param = self.visitLambdaParam(ctx.lambdaParam())

        if default_node := ctx.default():
            default = self.visitDefault(default_node)

            if self.pass_num == 1:
                param.default = default_node

            elif self.pass_num == 2:
                param.type = default.get_inferred_type()

        return param

    # lambdaParam: NAME;
    @_both_passes
    def visitLambdaParam(self, ctx: PythonParser.LambdaParamContext) -> PyParameter:
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if self.pass_num == 1:
            param = PyParameter(name)
            self.context.entities[ctx] = param

            symbol = Symbol(SymbolType.PARAMETER, name, name_node, entity=param)
            self.context.set_node_info(
                name_node, kind=TokenKind.VARIABLE, symbol=symbol
            )

            with self._wrap_errors(PyDuplicateSymbolError):
                self._parent_function.scope.define(symbol)

        else:
            param = self.context.entities[ctx]

        return param

    # string: STRING_LITERAL | BYTES_LITERAL;
    # strings: string+;
    @_type_check
    def visitStrings(self, ctx: PythonParser.StringsContext) -> PyType:
        result: Optional[str | bytes] = None

        for string in ctx.string():
            type_ = self.visitString(string)
            if not isinstance(type_, PyLiteralType):
                return PyType.ANY

            if result is None:
                result = type_.value
            elif type(result) is type(type_.value):
                result += type_.value
            else:
                self._report_error(
                    PyTypeError("cannot mix bytes and nonbytes literals").with_context(
                        string
                    )
                )
                return PyType.ANY

        if result is None:
            return PyType.ANY

        return PyLiteralType(result)

    # list: '[' starNamedExpressions? ']';
    @_type_check
    def visitList(self, ctx: PythonParser.ListContext) -> PyType:
        if expressions := ctx.starNamedExpressions():
            item_types = self.visitStarNamedExpressions(expressions)
        else:
            item_types = []

        type_args = infer_list_display(item_types)
        return PyInstanceType.from_stub("builtins.list", type_args)

    # tuple: '(' (starNamedExpression ',' starNamedExpressions?)? ')';
    @_type_check
    def visitTuple(self, ctx: PythonParser.TupleContext) -> PyType:
        item_types: list[PyType] = []

        if node := ctx.starNamedExpression():
            item_types.append(self.visitStarNamedExpression(node))

        if node := ctx.starNamedExpressions():
            item_types.extend(self.visitStarNamedExpressions(node))

        return PyTupleType.from_starred(item_types)

    # set: '{' starNamedExpressions '}';
    @_type_check
    def visitSet(self, ctx: PythonParser.SetContext) -> PyType:
        item_types = self.visitStarNamedExpressions(ctx.starNamedExpressions())

        type_args = infer_list_display(item_types)
        return PyInstanceType.from_stub("builtins.set", type_args)

    # dict: '{' doubleStarredKvpairs? '}';
    @_type_check
    def visitDict(self, ctx: PythonParser.DictContext) -> PyType:
        if kvpairs := ctx.doubleStarredKvpairs():
            item_types = self.visitDoubleStarredKvpairs(kvpairs)
        else:
            item_types = []

        type_args = infer_dict_display(item_types)
        return PyInstanceType.from_stub("builtins.dict", type_args)

    # doubleStarredKvpairs
    #   : doubleStarredKvpair (',' doubleStarredKvpair)* ','?;
    @_both_passes
    def visitDoubleStarredKvpairs(
        self, ctx: PythonParser.DoubleStarredKvpairsContext
    ) -> list[PyDictDisplayItem]:
        return [
            self.visitDoubleStarredKvpair(node) for node in ctx.doubleStarredKvpair()
        ]

    # doubleStarredKvpair
    #   : '**' bitwise
    #   | kvpair;
    @_type_check
    def visitDoubleStarredKvpair(
        self, ctx: PythonParser.DoubleStarredKvpairContext
    ) -> PyDictDisplayItem:
        if ctx.DOUBLESTAR():
            return PyUnpackKv(self.visitBitwise(ctx.bitwise()))
        else:
            return self.visitKvpair(ctx.kvpair())

    # kvpair: expression ':' expression;
    @_type_check
    def visitKvpair(self, ctx: PythonParser.KvpairContext) -> PyKvPair:
        key_type = self.visitExpression(ctx.expression(0))
        value_type = self.visitExpression(ctx.expression(1))

        return PyKvPair(key_type, value_type)

    # forIfClauses: forIfClause+;
    @_both_passes
    def visitForIfClauses(self, ctx: PythonParser.ForIfClausesContext):
        for i, clause in enumerate(ctx.forIfClause()):
            self.visitForIfClause(clause, level=i)

    # forIfClause
    #   : 'async'? 'for' starTargets 'in' logical ('if' logical)*;
    @_both_passes
    def visitForIfClause(self, ctx: PythonParser.ForIfClauseContext, *, level: int = 0):
        """
        Args:
            level: The level of the for-if clause, starting from 0.
        """
        for_clause, *if_clauses = ctx.logical()

        # The first iterable is evaluated in the enclosing scope.
        # The remaining iterables are evaluated in the current scope.
        if level == 0:
            self.context.set_node_scope(for_clause, self._current_scope.parent)

        iter_type = self.visitLogical(for_clause)

        visit_kwargs = {}
        if self.pass_num == 2:
            visit_kwargs["value_type"] = iter_type.get_iterated_type(
                is_async=ctx.ASYNC() is not None
            )

        self.visitStarTargets(ctx.starTargets(), **visit_kwargs)

        # The if-clauses are evaluated in the current scope.
        for logical in if_clauses:
            self.visitLogical(logical)

    # listcomp: '[' namedExpression forIfClauses ']';
    @_both_passes
    def visitListcomp(self, ctx: PythonParser.ListcompContext) -> Optional[PyType]:
        if self.pass_num == 1:
            scope = self._new_scope("<listcomp>", ScopeType.COMPREHENSION)
            self.context.set_node_scope(ctx.namedExpression(), scope)
            self.context.set_node_scope(ctx.forIfClauses(), scope)

        self.visitForIfClauses(ctx.forIfClauses())
        type_ = self.visitNamedExpression(ctx.namedExpression())

        if self.pass_num == 2:
            return PyInstanceType.from_stub(
                "builtins.list", (type_.get_inferred_type(),)
            )

    # setcomp: '{' namedExpression forIfClauses '}';
    @_both_passes
    def visitSetcomp(self, ctx: PythonParser.SetcompContext) -> Optional[PyType]:
        if self.pass_num == 1:
            scope = self._new_scope("<setcomp>", ScopeType.COMPREHENSION)
            self.context.set_node_scope(ctx.namedExpression(), scope)
            self.context.set_node_scope(ctx.forIfClauses(), scope)

        self.visitForIfClauses(ctx.forIfClauses())
        type_ = self.visitNamedExpression(ctx.namedExpression())

        if self.pass_num == 2:
            return PyInstanceType.from_stub(
                "builtins.set", (type_.get_inferred_type(),)
            )

    # genexp: '(' namedExpression forIfClauses ')';
    @_both_passes
    def visitGenexp(self, ctx: PythonParser.GenexpContext) -> Optional[PyType]:
        if self.pass_num == 1:
            scope = self._new_scope("<genexpr>", ScopeType.COMPREHENSION)
            self.context.set_node_scope(ctx.namedExpression(), scope)
            self.context.set_node_scope(ctx.forIfClauses(), scope)

        self.visitForIfClauses(ctx.forIfClauses())
        type_ = self.visitNamedExpression(ctx.namedExpression())

        if self.pass_num == 2:
            return PyInstanceType.from_stub(
                "typing.Generator",
                (type_.get_inferred_type(), PyType.NONE, PyType.NONE),
            )

    # dictcomp: '{' kvpair forIfClauses '}';
    @_both_passes
    def visitDictcomp(self, ctx: PythonParser.DictcompContext) -> Optional[PyType]:
        if self.pass_num == 1:
            scope = self._new_scope("<dictcomp>", ScopeType.COMPREHENSION)
            self.context.set_node_scope(ctx.kvpair(), scope)
            self.context.set_node_scope(ctx.forIfClauses(), scope)

        self.visitForIfClauses(ctx.forIfClauses())
        type_ = self.visitKvpair(ctx.kvpair())

        if self.pass_num == 2:
            key_type, value_type = type_
            return PyInstanceType.from_stub(
                "builtins.dict",
                (key_type.get_inferred_type(), value_type.get_inferred_type()),
            )

    # arguments: args ','?;
    @_type_check
    def visitArguments(self, ctx: PythonParser.ArgumentsContext) -> PyArguments:
        return self.visitArgs(ctx.args())

    # args
    #   : arg (',' arg)* (',' kwargs)?
    #   | kwargs;
    @_type_check
    def visitArgs(self, ctx: PythonParser.ArgsContext) -> PyArguments:
        arguments = PyArguments()

        for node in ctx.arg():
            arguments.append(self.visitArg(node))

        if node := ctx.kwargs():
            arguments = self.visitKwargs(node, arguments=arguments)

        return arguments

    # arg
    #   : starredExpression
    #   | assignmentExpression
    #   | expression;
    @_type_check
    def visitArg(self, ctx: PythonParser.ArgContext) -> PyType:
        if node := ctx.starredExpression():
            return self.visitStarredExpression(node)

        elif node := ctx.assignmentExpression():
            return self.visitAssignmentExpression(node)

        elif node := ctx.expression():
            return self.visitExpression(node)

    # kwargs
    #   : kwargOrStarred (',' kwargOrStarred)* (',' kwargOrDoubleStarred)?
    #   | kwargOrDoubleStarred (',' kwargOrDoubleStarred)*;
    @_type_check
    def visitKwargs(
        self,
        ctx: PythonParser.KwargsContext,
        *,
        arguments: Optional[PyArguments] = None,
    ) -> PyArguments:
        if arguments is None:
            arguments = PyArguments()

        for node in ctx.kwargOrStarred():
            arguments.append(self.visitKwargOrStarred(node))

        for node in ctx.kwargOrDoubleStarred():
            arguments.append(self.visitKwargOrDoubleStarred(node))

        return arguments

    # starredExpression
    #   : '*' expression;
    @_type_check
    def visitStarredExpression(
        self, ctx: PythonParser.StarredExpressionContext
    ) -> PyType:
        return PyUnpack(self.visitExpression(ctx.expression()))

    # kwargOrStarred
    #   : NAME '=' expression
    #   | starredExpression
    #   | invalidKwarg;
    @_type_check
    def visitKwargOrStarred(
        self, ctx: PythonParser.KwargOrStarredContext
    ) -> PyKeywordArgument | PyType:
        if node := ctx.invalidKwarg():
            return self.visitInvalidKwarg(node)

        if node := ctx.NAME():
            name = self.visitName(node)
            self._access_func_kwarg(name, node)

            type_ = self.visitExpression(ctx.expression())
            return PyKeywordArgument(name, type_)

        else:
            return self.visitStarredExpression(ctx.starredExpression())

    # kwargOrDoubleStarred
    #   : NAME '=' expression
    #   | '**' expression
    #   | invalidKwarg;
    @_type_check
    def visitKwargOrDoubleStarred(
        self, ctx: PythonParser.KwargOrDoubleStarredContext
    ) -> PyKeywordArgument | PyType:
        if node := ctx.invalidKwarg():
            return self.visitInvalidKwarg(node)

        if node := ctx.NAME():
            name = self.visitName(node)
            self._access_func_kwarg(name, node)

            type_ = self.visitExpression(ctx.expression())
            return PyKeywordArgument(name, type_)

        else:
            return PyUnpackKv(self.visitExpression(ctx.expression()))

    # invalidKwarg: NAME '='? | '*' | '**';
    @_both_passes
    def visitInvalidKwarg(
        self, ctx: PythonParser.InvalidKwargContext
    ) -> Optional[PyKeywordArgument | PyType]:
        if self.pass_num == 1:
            self._report_error(
                PySyntaxError("invalid keyword argument").with_context(ctx)
            )

        elif self.pass_num == 2:
            if name_node := ctx.NAME():
                name = self.visitName(name_node)
                self._access_func_kwarg(name, name_node)
                return PyKeywordArgument(name, PyType.ANY)

            elif ctx.STAR():
                return PyUnpack(PyType.ANY)

            elif ctx.DOUBLESTAR():
                return PyUnpackKv(PyType.ANY)

    # starTargets: starTarget (',' starTarget)* ','?;
    @_type_check
    def visitStarTargets(
        self,
        ctx: PythonParser.StarTargetsContext,
        *,
        value_type: PyType = PyType.ANY,
        always_unpack: bool = False,
    ):
        """
        Args:
            value_type: The type of the value assigned to the targets.
            always_unpack: Whether the value is always unpacked before assigned to the
                targets. Set to `True` if the targets are wrapped in square brackets.
        """
        if always_unpack or ctx.COMMA():
            # If the targets are wrapped in square brackets, or if at least one comma
            # is present, the value is unpacked before assigned to the targets.

            # Check if there is a starred expression in the targets.
            star_index: Optional[int] = None
            for i, star_target in enumerate(ctx.starTarget()):
                if star_target.STAR():
                    if star_index is not None:
                        self._report_error(
                            PySyntaxError(
                                "multiple starred expressions in assignment",
                            ).with_context(star_target)
                        )
                    else:
                        star_index = i

            # Try to determine the types of the unpacked values.
            unpack = PyUnpack(value_type)
            if (unpacked_types := unpack.get_unpacked_types()) is not None:
                self._handle_star_targets_unpacked(ctx, star_index, unpacked_types)

            else:
                unpacked_type = unpack.get_unpacked_type()
                self._handle_star_targets_homogenous(ctx, star_index, unpacked_type)

        else:
            # Otherwise, the value is directly assigned to the single target.
            star_target = ctx.starTarget(0)
            self._check_invalid_star(star_target)
            self.visitStarTarget(star_target, value_type=value_type)

    def _check_invalid_star(self, ctx: PythonParser.StarTargetContext):
        """
        Helper method to check for invalid starred expressions in assignment targets.
        """
        if ctx.STAR():
            self._report_error(PySyntaxError("cannot use starred expression here"))

    def _handle_star_targets_unpacked(
        self,
        ctx: PythonParser.StarTargetsContext,
        star_index: Optional[int],
        unpacked_types: tuple[PyType, ...],
    ):
        """
        Helper method to handle the assignment of unpacked values to starred targets.
        """
        num_targets, num_unpacked = len(ctx.starTarget()), len(unpacked_types)

        # Check if the number of unpacked values matches the number of targets.
        if star_index is not None:
            num_targets -= 1  # Exclude the starred target
            if (num_starred := num_unpacked - num_targets) < 0:
                self._report_error(
                    PyTypeError(
                        f"not enough values to unpack (expected at least "
                        f"{num_targets}, got {num_unpacked})",
                    )
                )
                num_starred = 0
        else:
            if num_unpacked < num_targets:
                self._report_error(
                    PyTypeError(
                        f"not enough values to unpack (expected {num_targets}, "
                        f"got {num_unpacked})",
                    )
                )
            elif num_unpacked > num_targets:
                self._report_error(
                    PyTypeError(
                        f"too many values to unpack (expected {num_targets}, "
                        f"got {num_unpacked})",
                    )
                )

        # Which of the unpacked items is assigned to the target?
        src_index = 0

        # Assign the unpacked types to the targets.
        for i, star_target in enumerate(ctx.starTarget()):
            if star_index is not None and i == star_index:
                value_types = unpacked_types[src_index : src_index + num_starred]
                src_index += num_starred

                # The starred target receives a list of the remaining values. However,
                # we pass the types as a "packed tuple" to handle nested starred
                # expressions. The tuple is finally converted to a list in the target.
                self.visitStarTarget(star_target, value_type=PyPackedTuple(value_types))

            else:
                try:
                    value_type = unpacked_types[src_index]
                except IndexError:
                    value_type = PyType.ANY
                src_index += 1

                self.visitStarTarget(star_target, value_type=value_type)

    def _handle_star_targets_homogenous(
        self,
        ctx: PythonParser.StarTargetsContext,
        star_index: Optional[int],
        value_type: PyType,
    ):
        """
        Helper method to handle the assignment of homogenous values to starred targets.
        """
        for i, star_target in enumerate(ctx.starTarget()):
            if star_index is not None and i == star_index:
                # The starred target receives a list of the remaining values.
                self.visitStarTarget(
                    star_target,
                    value_type=PyInstanceType.from_stub("builtins.list", (value_type,)),
                )
            else:
                # Otherwise, the target receives the value directly.
                self.visitStarTarget(star_target, value_type=value_type)

    @staticmethod
    def _convert_packed_tuple(value_type: PyType) -> PyType:
        """
        Helper method to convert a packed tuple type to a list type. If the type is not
        a packed tuple, it is returned as is.
        """
        if isinstance(value_type, PyPackedTuple):
            return value_type.to_list_type()
        else:
            return value_type

    # starTarget: '*'? targetWithStarAtom;
    @_type_check
    def visitStarTarget(
        self,
        ctx: PythonParser.StarTargetContext,
        *,
        value_type: PyType = PyType.ANY,
    ):
        # NOTE: The star is handled in parent context (`starTargets`, `starAtom` or
        # `withItem`).
        self.visitTargetWithStarAtom(ctx.targetWithStarAtom(), value_type=value_type)

    # targetWithStarAtom
    #   : primary '.' NAME
    #   | primary '[' slices ']'
    #   | starAtom
    #   | invalidPrimary;
    @_type_check
    def visitTargetWithStarAtom(
        self,
        ctx: PythonParser.TargetWithStarAtomContext,
        *,
        value_type: PyType = PyType.ANY,
    ):
        if node := ctx.invalidPrimary():
            return self.visitInvalidPrimary(node)

        if star_atom := ctx.starAtom():
            return self.visitStarAtom(star_atom, value_type=value_type)

        type_ = self.visitPrimary(ctx.primary())
        value_type = self._convert_packed_tuple(value_type)

        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            self._define_attribute(
                type_, name, name_node, value_type=value_type.get_inferred_type()
            )

        elif slices := ctx.slices():
            self.visitSlices(slices)

    # starAtom
    #   : NAME
    #   | '(' starTarget ')'
    #   | '(' starTargets? ')'
    #   | '[' starTargets? ']';
    @_both_passes
    def visitStarAtom(
        self, ctx: PythonParser.StarAtomContext, *, value_type: PyType = PyType.ANY
    ):
        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 1:
                self._define_variable(name, name_node)

            elif self.pass_num == 2:
                value_type = self._convert_packed_tuple(value_type)

                symbol = self._current_scope[name]
                self._set_variable_type(symbol, value_type.get_inferred_type())

        elif star_target := ctx.starTarget():
            if self.pass_num == 2:
                self._check_invalid_star(star_target)

            return self.visitStarTarget(star_target, value_type=value_type)

        elif star_targets := ctx.starTargets():
            return self.visitStarTargets(
                star_targets,
                value_type=value_type,
                always_unpack=ctx.LSQB() is not None,
            )

    # singleTarget
    #   : singleSubscriptAttributeTarget
    #   | NAME
    #   | '(' singleTarget ')';
    @_both_passes
    def visitSingleTarget(
        self,
        ctx: PythonParser.SingleTargetContext,
        *,
        define: bool = False,
        value_type: Optional[PyType] = None,
    ):
        """
        Args:
            define: Whether to define the target in the current scope.
            value_type: The type of the value assigned to the target.
        """
        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 1:
                if define:
                    self._define_variable(name, name_node)

            elif self.pass_num == 2:
                if value_type is not None:
                    # The type always derives from a type annotation. No further
                    # inference is performed.
                    symbol = self._current_scope[name]
                    self._set_variable_type(symbol, value_type)

                else:
                    self._access_variable(name, name_node)

        elif single_target := ctx.singleTarget():
            return self.visitSingleTarget(single_target, value_type=value_type)

        elif target := ctx.singleSubscriptAttributeTarget():
            return self.visitSingleSubscriptAttributeTarget(
                target, value_type=value_type
            )

    # singleSubscriptAttributeTarget
    #   : primary '.' NAME
    #   | primary '[' slices ']'
    #   | invalidPrimary;
    @_type_check
    def visitSingleSubscriptAttributeTarget(
        self,
        ctx: PythonParser.SingleSubscriptAttributeTargetContext,
        *,
        value_type: Optional[PyType] = None,
    ):
        if node := ctx.invalidPrimary():
            return self.visitInvalidPrimary(node)

        type_ = self.visitPrimary(ctx.primary())

        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if value_type is not None:
                # Normal assignment
                self._define_attribute(type_, name, name_node, value_type=value_type)

            else:
                # Augmented assignment
                self._access_attribute(type_, name, name_node)

        elif slices := ctx.slices():
            self.visitSlices(slices)

    # delTarget
    #   : primary '.' NAME
    #   | primary '[' slices ']'
    #   | delTargetAtom
    #   | invalidPrimary;
    @_both_passes
    def visitDelTarget(self, ctx: PythonParser.DelTargetContext):
        if node := ctx.invalidPrimary():
            return self.visitInvalidPrimary(node)

        if atom := ctx.delTargetAtom():
            return self.visitDelTargetAtom(atom)

        type_ = self.visitPrimary(ctx.primary())

        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 2:
                # TODO: handle the deletion
                self._access_attribute(type_, name, name_node)

        elif slices := ctx.slices():
            self.visitSlices(slices)

    # delTargetAtom
    #   : NAME
    #   | '(' delTarget ')'
    #   | '(' delTargets? ')'
    #   | '[' delTargets? ']';
    @_both_passes
    def visitDelTargetAtom(self, ctx: PythonParser.DelTargetAtomContext):
        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 2:
                # TODO: handle the deletion
                self._access_variable(name, name_node)

        elif del_target := ctx.delTarget():
            return self.visitDelTarget(del_target)

        elif del_targets := ctx.delTargets():
            return self.visitDelTargets(del_targets)
