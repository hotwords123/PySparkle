from ast import literal_eval
from typing import NamedTuple, Optional

from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.tree.Tree import ErrorNode, ParseTree, TerminalNode

from grammar import PythonParserVisitor
from grammar.PythonParser import PythonParser

from .base import SemanticError
from .entity import PyClass, PyFunction, PyLambda, PyParameter, PyTypeError
from .scope import PyDuplicateSymbolError, ScopeType, SymbolTable
from .structure import (
    PyArguments,
    PyDoubleStarArgument,
    PyImportFrom,
    PyImportFromAsName,
    PyImportFromTargets,
    PyImportName,
    PyKeywordArgument,
    PyParameterSpec,
    PyPositionalArgument,
    PythonContext,
)
from .symbol import Symbol, SymbolType
from .token import TokenKind
from .types import (
    PyClassType,
    PyInstanceType,
    PyLiteralType,
    PyType,
    get_context_entity,
)


def _visitor_guard(func):
    def wrapper(self: "PythonVisitor", ctx: ParserRuleContext, **kwargs):
        if ctx.exception is not None:
            return self.visitChildren(ctx)
        return func(self, ctx, **kwargs)

    return wrapper


def _first_pass_only(func):
    def wrapper(self: "PythonVisitor", node: ParseTree, **kwargs):
        if self.pass_num == 1:
            return func(self, node, **kwargs)

    return wrapper


def _type_check(func):
    def wrapper(self: "PythonVisitor", node: ParseTree, **kwargs):
        if self.pass_num == 2:
            return func(self, node, **kwargs)
        return self.visitChildren(node)

    return wrapper


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
    "staticmethod": "staticmethod",
    "classmethod": "classmethod",
    "property": "property",  # TODO: getter, setter, deleter
    "typing.final": "final",
    "typing.overload": "overload",
    "abc.abstractmethod": "abstractmethod",
}


def _modifier_from_decorator(
    decorator: PyType, modifiers: dict[str, str]
) -> Optional[str]:
    if entity := decorator.entity:
        for name, modifier in modifiers.items():
            if entity is get_context_entity(name):
                return modifier

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

        self._outer_symbols: list[_OuterSymbol] = []

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
        self.visit(tree)

    def _resolve_outer_symbols(self):
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
        # TODO: Respect the __all__ list.
        for symbol in self.context.global_scope.symbols():
            if symbol.name.startswith("_"):
                symbol.public = False
            elif symbol.public is None:
                symbol.public = symbol.type is not SymbolType.IMPORTED

    def aggregateResult(self, aggregate, nextResult):
        return aggregate or nextResult

    def visitTerminal(self, node: TerminalNode):
        match node.getSymbol().type:
            case PythonParser.NAME:
                return self.visitName(node)

            case (
                PythonParser.NONE
                | PythonParser.FALSE
                | PythonParser.TRUE
                | PythonParser.ELLIPSIS
                | PythonParser.STRING_LITERAL
                | PythonParser.BYTES_LITERAL
                | PythonParser.INTEGER
                | PythonParser.FLOAT_NUMBER
                | PythonParser.IMAG_NUMBER
            ):
                return self.visitLiteral(node)

            case _:
                return super().visitTerminal(node)

    def visitName(self, node: TerminalNode) -> str:
        """
        Returns:
            name: The name of the node.
        """
        return node.getText()

    @_type_check
    def visitLiteral(self, node: TerminalNode) -> PyType:
        try:
            return PyLiteralType(literal_eval(node.getText()))
        except ValueError:
            return PyType.ANY

    def visitErrorNode(self, node: ErrorNode):
        if self.pass_num == 1:
            self.context.set_node_info(node, kind=TokenKind.ERROR)

    # singleTarget ':' expression ('=' assignmentRhs)?
    @_visitor_guard
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
                value_type=annotation.get_annotated_type(self.context),
            )

    # (starTargets '=')+ assignmentRhs
    @_type_check
    @_visitor_guard
    def visitStarredAssignment(self, ctx: PythonParser.StarredAssignmentContext):
        type_ = self.visitAssignmentRhs(ctx.assignmentRhs())

        for star_targets in ctx.starTargets():
            self.visitStarTargets(star_targets, value_type=type_)

    # globalStmt: 'global' NAME (',' NAME)*;
    @_first_pass_only
    @_visitor_guard
    def visitGlobalStmt(self, ctx: PythonParser.GlobalStmtContext):
        for name_node in ctx.NAME():
            name = self.visitName(name_node)

            symbol = Symbol(SymbolType.GLOBAL, name, name_node)
            self.context.set_node_info(name_node, symbol=symbol)

            with self.context.wrap_errors(PyDuplicateSymbolError):
                self.context.current_scope.define(symbol)

            self._outer_symbols.append(_OuterSymbol(symbol, self.context.current_scope))

    # nonlocalStmt: 'nonlocal' NAME (',' NAME)*;
    @_first_pass_only
    @_visitor_guard
    def visitNonlocalStmt(self, ctx: PythonParser.NonlocalStmtContext):
        for name_node in ctx.NAME():
            name = self.visitName(name_node)

            symbol = Symbol(SymbolType.NONLOCAL, name, name_node)
            self.context.set_node_info(name_node, symbol=symbol)

            with self.context.wrap_errors(PyDuplicateSymbolError):
                self.context.current_scope.define(symbol)

            self._outer_symbols.append(_OuterSymbol(symbol, self.context.current_scope))

    # importFrom
    #   : 'from' ('.' | '...')* dottedName 'import' importFromTargets
    #   | 'from' ('.' | '...')+ 'import' importFromTargets;
    @_first_pass_only
    @_visitor_guard
    def visitImportFrom(self, ctx: PythonParser.ImportFromContext):
        num_dots = len(ctx.DOT()) + 3 * len(ctx.ELLIPSIS())
        relative = num_dots - 1 if num_dots > 0 else None

        if dotted_name := ctx.dottedName():
            path, _ = self.visitDottedName(dotted_name)
        else:
            path = []

        targets = self.visitImportFromTargets(ctx.importFromTargets())

        self.context.imports.append(PyImportFrom(path, relative, targets))

    # importFromTargets
    #   : '(' importFromAsNames ','? ')'
    #   | importFromAsNames
    #   | '*';
    @_first_pass_only
    @_visitor_guard
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
            return PyImportFromTargets()

    # importFromAsNames: importFromAsName (',' importFromAsName)*;
    @_first_pass_only
    @_visitor_guard
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
    @_visitor_guard
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

        with self.context.wrap_errors(PyDuplicateSymbolError):
            self.context.current_scope.define(symbol)

        return PyImportFromAsName(name, alias, symbol)

    # dottedAsName: dottedName ('as' NAME)?;
    @_first_pass_only
    @_visitor_guard
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

            with self.context.wrap_errors(PyDuplicateSymbolError):
                self.context.current_scope.define(symbol)

        else:
            path, symbol = self.visitDottedName(ctx.dottedName(), define=True)
            alias = None

        self.context.imports.append(PyImportName(path, alias, symbol))

    # dottedName: dottedName '.' NAME | NAME;
    @_first_pass_only
    @_visitor_guard
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

                with self.context.wrap_errors(PyDuplicateSymbolError):
                    self.context.current_scope.define(symbol)

            return [name], symbol

    # decorators: ('@' namedExpression NEWLINE)+;
    @_type_check
    @_visitor_guard
    def visitDecorators(self, ctx: PythonParser.DecoratorsContext) -> list[PyType]:
        return [self.visitNamedExpression(expr) for expr in ctx.namedExpression()]

    # classDef
    #   : decorators? 'class' NAME typeParams? ('(' arguments? ')')?
    #       ':' block;
    @_visitor_guard
    def visitClassDef(self, ctx: PythonParser.ClassDefContext):
        if decorators := ctx.decorators():
            decorators = self.visitDecorators(decorators)
        else:
            decorators = []

        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if node := ctx.arguments():
            arguments = self.visitArguments(node)
        else:
            arguments = PyArguments()

        if self.pass_num == 1:
            scope = self.context.new_scope(ctx, f"<class '{name}'>", ScopeType.CLASS)
            entity = PyClass(name, scope)
            self.context.entities[ctx] = entity

            symbol = Symbol(SymbolType.CLASS, name, name_node, entity=entity)
            self.context.set_node_info(name_node, kind=TokenKind.CLASS, symbol=symbol)

            with self.context.wrap_errors(PyDuplicateSymbolError):
                self.context.current_scope.define(symbol)

        elif self.pass_num == 2:
            scope = self.context.scope_of(ctx)

            entity: PyClass = self.context.entities[ctx]

            # Handle class decorators.
            entity.decorators = decorators
            for decorator in decorators:
                if modifier := _modifier_from_decorator(decorator, _CLASS_MODIFIERS):
                    entity.modifiers.add(modifier)

            # Handle class arguments.
            entity.arguments = arguments
            entity.bases.extend(
                arg.type.cls
                for arg in arguments.args
                if not arg.starred and isinstance(arg.type, PyClassType)
            )

            with self.context.wrap_errors(PyTypeError):
                entity.compute_mro()

        with self.context.scope_guard(scope):
            if type_params := ctx.typeParams():
                self.visitTypeParams(type_params)

            self.visitBlock(ctx.block())

    # functionDef
    #   : decorators? 'async'? 'def' NAME typeParams? '(' parameters? ')'
    #       ('->' expression)? ':' block;
    @_visitor_guard
    def visitFunctionDef(self, ctx: PythonParser.FunctionDefContext):
        if decorators := ctx.decorators():
            decorators = self.visitDecorators(decorators)
        else:
            decorators = []

        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if self.pass_num == 1:
            scope = self.context.new_scope(ctx, f"<function '{name}'>", ScopeType.LOCAL)

            # Find the class that the function is defined in.
            if self.context.current_scope.scope_type is ScopeType.CLASS:
                node = ctx.parentCtx
                while not isinstance(node, PythonParser.ClassDefContext):
                    node = node.parentCtx
                cls = self.context.entities[node]
            else:
                cls = None

            entity = PyFunction(name, scope, cls=cls)
            self.context.entities[ctx] = entity

            symbol = Symbol(SymbolType.FUNCTION, name, name_node, entity=entity)
            self.context.set_node_info(
                name_node, kind=TokenKind.FUNCTION, symbol=symbol
            )

            with self.context.wrap_errors(PyDuplicateSymbolError):
                self.context.current_scope.define(symbol)

        else:
            scope = self.context.scope_of(ctx)

            entity: PyFunction = self.context.entities[ctx]
            entity.decorators = decorators

            for decorator in decorators:
                if modifier := _modifier_from_decorator(decorator, _FUNCTION_MODIFIERS):
                    entity.modifiers.add(modifier)

        if expression := ctx.expression():
            annotation = self.visitExpression(expression)

            if self.pass_num == 2:
                annotation: PyType
                entity.return_type = annotation.get_annotated_type(self.context)

        with self.context.scope_guard(scope):
            if type_params := ctx.typeParams():
                self.visitTypeParams(type_params)

            if parameters := ctx.parameters():
                parameters = self.visitParameters(parameters)

                if self.pass_num == 1:
                    entity.parameters.extend(parameters)

                elif self.pass_num == 2:
                    if (
                        entity.is_method
                        and parameters
                        and (first_param := parameters[0]).type is None
                        and not first_param.spec.kwonly
                        and first_param.spec.star is None
                    ):
                        if entity.has_modifier("classmethod"):
                            # Class methods have a `cls` parameter.
                            first_param.type = entity.cls.get_type()

                        elif not entity.has_modifier("staticmethod"):
                            # Instance methods have a `self` parameter.
                            first_param.type = entity.cls.get_instance_type()

            self.visitBlock(ctx.block())

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
    @_visitor_guard
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
    @_visitor_guard
    def visitSlashNoDefault(
        self, ctx: PythonParser.SlashNoDefaultContext
    ) -> list[PyParameter]:
        parameters: list[PyParameter] = []

        for node in ctx.paramNoDefault():
            parameters.append(self.visitParamNoDefault(node))

        if self.pass_num == 1:
            for param in parameters:
                param.spec.posonly = True

        return parameters

    # slashWithDefault
    #   : paramNoDefault (',' paramNoDefault)* (',' paramWithDefault)+ ',' '/'
    #   | paramWithDefault (',' paramWithDefault)* ',' '/';
    @_visitor_guard
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
                param.spec.posonly = True

        return parameters

    # starEtc
    #   : '*' (paramNoDefault | paramNoDefaultStarAnnotation)
    #       (',' paramMaybeDefault)* (',' kwds?)?
    #   | '*' (',' paramMaybeDefault)+ (',' kwds?)?
    #   | kwds;
    @_visitor_guard
    def visitStarEtc(self, ctx: PythonParser.StarEtcContext) -> list[PyParameter]:
        parameters: list[PyParameter] = []

        if node := ctx.paramNoDefault():
            parameters.append(param := self.visitParamNoDefault(node))

            if self.pass_num == 1:
                param.spec.star = "*"

            elif self.pass_num == 2:
                # TODO: respect original annotations.
                param.type = PyInstanceType.from_builtin("tuple")

        elif node := ctx.paramNoDefaultStarAnnotation():
            parameters.append(param := self.visitParamNoDefaultStarAnnotation(node))

        for node in ctx.paramMaybeDefault():
            parameters.append(param := self.visitParamMaybeDefault(node))

            if self.pass_num == 1:
                param.spec.kwonly = True

        if node := ctx.kwds():
            parameters.append(self.visitKwds(node))

        return parameters

    # kwds: '**' paramNoDefault ','?;
    def visitKwds(self, ctx: PythonParser.KwdsContext) -> PyParameter:
        param = self.visitParamNoDefault(ctx.paramNoDefault())

        if self.pass_num == 1:
            param.spec.star = "**"

        elif self.pass_num == 2:
            # TODO: respect original annotations.
            param.type = PyInstanceType.from_builtin("dict")

        return param

    # paramNoDefault: param;
    def visitParamNoDefault(
        self, ctx: PythonParser.ParamNoDefaultContext
    ) -> PyParameter:
        return self.visitParam(ctx.param())

    # paramNoDefaultStarAnnotation: paramStarAnnotation;
    def visitParamNoDefaultStarAnnotation(
        self, ctx: PythonParser.ParamNoDefaultStarAnnotationContext
    ) -> PyParameter:
        return self.visitParamStarAnnotation(ctx.paramStarAnnotation())

    # paramWithDefault: param default;
    def visitParamWithDefault(
        self, ctx: PythonParser.ParamWithDefaultContext
    ) -> PyParameter:
        param = self.visitParam(ctx.param())

        default = self.visitDefault(default_node := ctx.default())

        if self.pass_num == 1:
            param.spec.default = default_node

        elif self.pass_num == 2:
            if param.type is None:
                param.type = default.get_inferred_type()

        return param

    # paramMaybeDefault: param default?;
    def visitParamMaybeDefault(
        self, ctx: PythonParser.ParamMaybeDefaultContext
    ) -> PyParameter:
        param = self.visitParam(ctx.param())

        if default_node := ctx.default():
            default = self.visitDefault(default_node)

            if self.pass_num == 1:
                param.spec.default = default_node

            elif self.pass_num == 2:
                if param.type is None:
                    param.type = default.get_inferred_type()

        return param

    # param: NAME annotation?;
    @_visitor_guard
    def visitParam(self, ctx: PythonParser.ParamContext) -> PyParameter:
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        annotation_node = ctx.annotation()

        if self.pass_num == 1:
            param_spec = PyParameterSpec(annotation=annotation_node)
            param = PyParameter(name, param_spec)
            self.context.entities[ctx] = param

            symbol = Symbol(SymbolType.PARAMETER, name, name_node, entity=param)
            self.context.set_node_info(
                name_node, kind=TokenKind.VARIABLE, symbol=symbol
            )

            with self.context.wrap_errors(PyDuplicateSymbolError):
                self.context.current_scope.define(symbol)

        else:
            param: PyParameter = self.context.entities[ctx]

        if annotation_node:
            with self.context.scope_guard(self.context.parent_scope):
                annotation = self.visitAnnotation(annotation_node)

            if self.pass_num == 2:
                param.type = annotation

        return param

    # paramStarAnnotation: NAME starAnnotation;
    @_visitor_guard
    def visitParamStarAnnotation(
        self, ctx: PythonParser.ParamStarAnnotationContext
    ) -> PyParameter:
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        annotation_node = ctx.starAnnotation()

        if self.pass_num == 1:
            param_spec = PyParameterSpec(star=True, star_annotation=annotation_node)
            param = PyParameter(name, param_spec)

            symbol = Symbol(SymbolType.PARAMETER, name, name_node, entity=param)
            self.context.set_node_info(
                name_node, kind=TokenKind.VARIABLE, symbol=symbol
            )

            with self.context.wrap_errors(PyDuplicateSymbolError):
                self.context.current_scope.define(symbol)

        else:
            param = self.context.entities[ctx]

        with self.context.scope_guard(self.context.parent_scope):
            annotation = self.visitStarAnnotation(annotation_node)

        if self.pass_num == 2:
            param.type = annotation

        return param

    # annotation: ':' expression;
    @_type_check
    @_visitor_guard
    def visitAnnotation(self, ctx: PythonParser.AnnotationContext) -> PyType:
        annotation: PyType = self.visitExpression(ctx.expression())
        return annotation.get_annotated_type(self.context)

    # starAnnotation: ':' starExpression;
    @_type_check
    @_visitor_guard
    def visitStarAnnotation(self, ctx: PythonParser.StarAnnotationContext) -> PyType:
        annotation: PyType = self.visitStarExpression(ctx.starExpression())
        return PyType.ANY  # TODO: star-annotated parameters

    # default: '=' expression;
    @_visitor_guard
    def visitDefault(self, ctx: PythonParser.DefaultContext) -> Optional[PyType]:
        with self.context.scope_guard(self.context.parent_scope):
            return self.visitExpression(ctx.expression())

    # exceptBlock
    #   : 'except' (expression ('as' NAME)?)? ':' block;
    @_visitor_guard
    def visitExceptBlock(self, ctx: PythonParser.ExceptBlockContext):
        if expression := ctx.expression():
            type_ = self.visitExpression(expression)
        else:
            type_ = PyType.ANY

        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 1:
                self.context.define_variable(name, name_node)

            elif self.pass_num == 2:
                symbol = self.context.current_scope[name]
                if isinstance(type_, PyClassType):
                    self.context.set_variable_type(
                        symbol, type_.cls.get_instance_type()
                    )

        self.visitBlock(ctx.block())

    # exceptStarBlock
    #   : 'except' '*' expression ('as' NAME)? ':' block;
    @_visitor_guard
    def visitExceptStarBlock(self, ctx: PythonParser.ExceptStarBlockContext):
        self.visitExpression(ctx.expression())

        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 1:
                self.context.define_variable(name, name_node)

        self.visitBlock(ctx.block())

    # TODO: expressions

    # assignmentExpression: NAME ':=' expression;
    @_visitor_guard
    def visitAssignmentExpression(
        self, ctx: PythonParser.AssignmentExpressionContext
    ) -> Optional[PyType]:
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        type_ = self.visitExpression(ctx.expression())

        # According to PEP 572, an assignment expression occurring in comprehensions
        # binds the target in the containing scope, honoring a global or nonlocal
        # declaration if present.
        scope = self.context.current_scope
        while scope.scope_type is ScopeType.COMPREHENSION:
            assert (scope := scope.parent) is not None

        if self.pass_num == 1:
            self.context.define_variable(name, name_node, scope=scope)

        elif self.pass_num == 2:
            symbol = scope[name]
            return self.context.set_variable_type(symbol, type_.get_inferred_type())

    # logical
    #   : 'not' logical
    #   | logical 'and' logical
    #   | logical 'or' logical
    #   | comparison;
    @_type_check
    @_visitor_guard
    def visitLogical(self, ctx: PythonParser.LogicalContext) -> PyType:
        if ctx.NOT():
            self.visitLogical(ctx.logical(0))

            return PyInstanceType.from_builtin("bool")

        elif ctx.AND() or ctx.OR():
            left_type = self.visitLogical(ctx.logical(0))
            right_type = self.visitLogical(ctx.logical(1))

            return PyType.ANY  # TODO

        else:
            return self.visitComparison(ctx.comparison())

    # comparison: bitwise compareOpBitwisePair*;
    @_type_check
    @_visitor_guard
    def visitComparison(self, ctx: PythonParser.ComparisonContext) -> PyType:
        type_ = self.visitBitwise(ctx.bitwise())

        pairs = ctx.compareOpBitwisePair()
        for pair in pairs:
            self.visitCompareOpBitwisePair(pair)

        if pairs:
            # TODO: The result of a comparison is not necessarily a boolean for
            # custom comparison methods.
            return PyInstanceType.from_builtin("bool")
        else:
            return type_

    # TODO: bitwise, arithmetic

    # awaitPrimary: 'await' primary | primary;
    @_type_check
    @_visitor_guard
    def visitAwaitPrimary(self, ctx: PythonParser.AwaitPrimaryContext) -> PyType:
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
    @_visitor_guard
    def visitPrimary(self, ctx: PythonParser.PrimaryContext) -> PyType:
        if atom := ctx.atom():
            return self.visitAtom(atom)

        type_ = self.visitPrimary(ctx.primary())

        if name_node := ctx.NAME():
            # Attribute access
            name = self.visitName(name_node)

            return self.context.access_attribute(type_, name, name_node)

        elif slices := ctx.slices():
            # Subscription
            self.visitSlices(slices)

            return type_.get_subscripted_type()

        elif genexp := ctx.genexp():
            # Function call with generator expression
            self.visitGenexp(genexp)

            return type_.get_return_type()

        else:
            # Function call
            if arguments := ctx.arguments():
                self.visitArguments(arguments)

            return type_.get_return_type()

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
    @_visitor_guard
    def visitAtom(self, ctx: PythonParser.AtomContext) -> Optional[PyType]:
        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 2:
                return self.context.access_variable(name, name_node)

        else:
            return super().visitAtom(ctx)

    # lambdef
    #   : 'lambda' lambdaParameters? ':' expression;
    @_visitor_guard
    def visitLambdef(self, ctx: PythonParser.LambdefContext) -> Optional[PyType]:
        if self.pass_num == 1:
            scope = self.context.new_scope(ctx, "<lambda>", ScopeType.LAMBDA)
            entity = PyLambda(scope)
            self.context.entities[ctx] = entity
        else:
            scope = self.context.scope_of(ctx)
            entity: PyLambda = self.context.entities[ctx]

        with self.context.scope_guard(scope):
            if node := ctx.lambdaParameters():
                parameters = self.visitLambdaParameters(node)

            type_ = self.visitExpression(ctx.expression())

            if self.pass_num == 1:
                entity.parameters.extend(parameters)

            elif self.pass_num == 2:
                type_: PyType
                entity.return_type = type_.get_inferred_type()

        if self.pass_num == 2:
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
    @_visitor_guard
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
    @_visitor_guard
    def visitLambdaSlashNoDefault(
        self, ctx: PythonParser.LambdaSlashNoDefaultContext
    ) -> list[PyParameter]:
        parameters: list[PyParameter] = []

        for node in ctx.lambdaParamNoDefault():
            parameters.append(self.visitLambdaParamNoDefault(node))

        if self.pass_num == 1:
            for param in parameters:
                param.spec.posonly = True

        return parameters

    # lambdaSlashWithDefault
    #   : lambdaParamNoDefault (',' lambdaParamNoDefault)* (',' lambdaParamWithDefault)+ ',' '/'
    #   | lambdaParamWithDefault (',' lambdaParamWithDefault)* ',' '/';
    @_visitor_guard
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
                param.spec.posonly = True

        return parameters

    # lambdaStarEtc
    #   : '*' lambdaParamNoDefault (',' lambdaParamMaybeDefault)* (',' lambdaKwds?)?
    #   | '*' (',' lambdaParamMaybeDefault)+ (',' lambdaKwds?)?
    #   | lambdaKwds;
    @_visitor_guard
    def visitLambdaStarEtc(
        self, ctx: PythonParser.LambdaStarEtcContext
    ) -> list[PyParameter]:
        parameters: list[PyParameter] = []

        if node := ctx.lambdaParamNoDefault():
            parameters.append(param := self.visitLambdaParamNoDefault(node))

            if self.pass_num == 1:
                param.spec.star = "*"

            elif self.pass_num == 2:
                param.type = PyInstanceType.from_builtin("tuple")

        for node in ctx.lambdaParamMaybeDefault():
            parameters.append(param := self.visitLambdaParamMaybeDefault(node))

            if self.pass_num == 1:
                param.spec.kwonly = True

        if node := ctx.lambdaKwds():
            parameters.append(self.visitLambdaKwds(node))

        return parameters

    # lambdaKwds: '**' lambdaParamNoDefault ','?;
    def visitLambdaKwds(self, ctx: PythonParser.LambdaKwdsContext) -> PyParameter:
        param = self.visitLambdaParamNoDefault(ctx.lambdaParamNoDefault())

        if self.pass_num == 1:
            param.spec.star = "**"

        elif self.pass_num == 2:
            param.type = PyInstanceType.from_builtin("dict")

        return param

    # lambdaParamNoDefault: lambdaParam;
    def visitLambdaParamNoDefault(
        self, ctx: PythonParser.LambdaParamNoDefaultContext
    ) -> PyParameter:
        return self.visitLambdaParam(ctx.lambdaParam())

    # lambdaParamWithDefault: lambdaParam default;
    def visitLambdaParamWithDefault(
        self, ctx: PythonParser.LambdaParamWithDefaultContext
    ) -> PyParameter:
        param = self.visitLambdaParam(ctx.lambdaParam())

        default = self.visitDefault(ctx.default())

        if self.pass_num == 1:
            param.spec.default = ctx.default()

        elif self.pass_num == 2:
            param.type = default.get_inferred_type()

        return param

    # lambdaParamMaybeDefault: lambdaParam default?;
    def visitLambdaParamMaybeDefault(
        self, ctx: PythonParser.LambdaParamMaybeDefaultContext
    ) -> PyParameter:
        param = self.visitLambdaParam(ctx.lambdaParam())

        if default_node := ctx.default():
            default = self.visitDefault(default_node)

            if self.pass_num == 1:
                param.spec.default = default_node

            elif self.pass_num == 2:
                param.type = default.get_inferred_type()

        return param

    # lambdaParam: NAME;
    @_visitor_guard
    def visitLambdaParam(self, ctx: PythonParser.LambdaParamContext) -> PyParameter:
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if self.pass_num == 1:
            param = PyParameter(name, PyParameterSpec())
            self.context.entities[ctx] = param

            symbol = Symbol(SymbolType.PARAMETER, name, name_node, entity=param)
            self.context.set_node_info(
                name_node, kind=TokenKind.VARIABLE, symbol=symbol
            )

            with self.context.wrap_errors(PyDuplicateSymbolError):
                self.context.current_scope.define(symbol)

        else:
            param = self.context.entities[ctx]

        return param

    # string: STRING_LITERAL | BYTES_LITERAL;
    # strings: string+;
    @_type_check
    @_visitor_guard
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
                token = string.STRING_LITERAL() or string.BYTES_LITERAL()
                self.context.errors.append(
                    SemanticError("cannot mix bytes and nonbytes literals", token)
                )
                return PyType.ANY

        if result is None:
            return PyType.ANY

        return PyLiteralType(result)

    # list: '[' starNamedExpressions? ']';
    @_type_check
    @_visitor_guard
    def visitList(self, ctx: PythonParser.ListContext) -> PyType:
        if expressions := ctx.starNamedExpressions():
            self.visitStarNamedExpressions(expressions)

        # TODO: literals
        return PyInstanceType.from_builtin("list")

    # tuple: '(' (starNamedExpression ',' starNamedExpressions?)? ')';
    @_type_check
    @_visitor_guard
    def visitTuple(self, ctx: PythonParser.TupleContext) -> PyType:
        if expression := ctx.starNamedExpression():
            self.visitStarNamedExpression(expression)

        if expressions := ctx.starNamedExpressions():
            self.visitStarNamedExpressions(expressions)

        # TODO: literals
        return PyInstanceType.from_builtin("tuple")

    # set: '{' starNamedExpressions '}';
    @_type_check
    @_visitor_guard
    def visitSet(self, ctx: PythonParser.SetContext) -> PyType:
        self.visitStarNamedExpressions(ctx.starNamedExpressions())

        # TODO: literals
        return PyInstanceType.from_builtin("set")

    # dict: '{' doubleStarredKvpairs? '}';
    @_type_check
    @_visitor_guard
    def visitDict(self, ctx: PythonParser.DictContext) -> PyType:
        if kvpairs := ctx.doubleStarredKvpairs():
            self.visitDoubleStarredKvpairs(kvpairs)

        # TODO: literals
        return PyInstanceType.from_builtin("dict")

    # forIfClauses: forIfClause+;
    @_visitor_guard
    def visitForIfClauses(self, ctx: PythonParser.ForIfClausesContext):
        for i, clause in enumerate(ctx.forIfClause()):
            self.visitForIfClause(clause, level=i)

    # forIfClause
    #   : 'async'? 'for' starTargets 'in' logical ('if' logical)*;
    @_visitor_guard
    def visitForIfClause(self, ctx: PythonParser.ForIfClauseContext, *, level: int = 0):
        """
        Args:
            level: The level of the for-if clause, starting from 0.
        """
        self.visitStarTargets(ctx.starTargets())

        # The first iterable is evaluated in the enclosing scope.
        # The remaining iterables are evaluated in the current scope.
        scope = self.context.parent_scope if level == 0 else self.context.current_scope
        with self.context.scope_guard(scope):
            self.visitLogical(ctx.logical(0))

        # The if-clauses are evaluated in the current scope.
        for logical in ctx.logical()[1:]:
            self.visitLogical(logical)

    # listcomp: '[' namedExpression forIfClauses ']';
    @_visitor_guard
    def visitListcomp(self, ctx: PythonParser.ListcompContext) -> Optional[PyType]:
        if self.pass_num == 1:
            scope = self.context.new_scope(ctx, "<listcomp>", ScopeType.COMPREHENSION)
        else:
            scope = self.context.scope_of(ctx)

        with self.context.scope_guard(scope):
            self.visitNamedExpression(ctx.namedExpression())
            self.visitForIfClauses(ctx.forIfClauses())

        if self.pass_num == 2:
            return PyInstanceType.from_builtin("list")

    # setcomp: '{' namedExpression forIfClauses '}';
    @_visitor_guard
    def visitSetcomp(self, ctx: PythonParser.SetcompContext) -> Optional[PyType]:
        if self.pass_num == 1:
            scope = self.context.new_scope(ctx, "<setcomp>", ScopeType.COMPREHENSION)
        else:
            scope = self.context.scope_of(ctx)

        with self.context.scope_guard(scope):
            self.visitNamedExpression(ctx.namedExpression())
            self.visitForIfClauses(ctx.forIfClauses())

        if self.pass_num == 2:
            return PyInstanceType.from_builtin("set")

    # genexp: '(' namedExpression forIfClauses ')';
    @_visitor_guard
    def visitGenexp(self, ctx: PythonParser.GenexpContext) -> Optional[PyType]:
        if self.pass_num == 1:
            scope = self.context.new_scope(ctx, "<genexp>", ScopeType.COMPREHENSION)
        else:
            scope = self.context.scope_of(ctx)

        with self.context.scope_guard(scope):
            self.visitNamedExpression(ctx.namedExpression())
            self.visitForIfClauses(ctx.forIfClauses())

        if self.pass_num == 2:
            return PyInstanceType.from_builtin("types.GeneratorType")

    # dictcomp: '{' kvpair forIfClauses '}';
    @_visitor_guard
    def visitDictcomp(self, ctx: PythonParser.DictcompContext) -> Optional[PyType]:
        if self.pass_num == 1:
            scope = self.context.new_scope(ctx, "<dictcomp>", ScopeType.COMPREHENSION)
        else:
            scope = self.context.scope_of(ctx)

        with self.context.scope_guard(scope):
            self.visitKvpair(ctx.kvpair())
            self.visitForIfClauses(ctx.forIfClauses())

        if self.pass_num == 2:
            return PyInstanceType.from_builtin("dict")

    # arguments: args ','?;
    @_type_check
    @_visitor_guard
    def visitArguments(self, ctx: PythonParser.ArgumentsContext) -> PyArguments:
        return self.visitArgs(ctx.args())

    # args
    #   : arg (',' arg)* (',' kwargs)?
    #   | kwargs;
    @_type_check
    @_visitor_guard
    def visitArgs(self, ctx: PythonParser.ArgsContext) -> PyArguments:
        arguments = PyArguments()

        for node in ctx.arg():
            arguments.args.append(self.visitArg(node))

        if node := ctx.kwargs():
            arguments = self.visitKwargs(node, arguments=arguments)

        return arguments

    # arg
    #   : starredExpression
    #   | assignmentExpression
    #   | expression;
    @_type_check
    @_visitor_guard
    def visitArg(self, ctx: PythonParser.ArgContext) -> PyPositionalArgument:
        starred = False

        if node := ctx.starredExpression():
            type_ = self.visitStarredExpression(node)
            starred = True

        elif node := ctx.assignmentExpression():
            type_ = self.visitAssignmentExpression(node)

        elif node := ctx.expression():
            type_ = self.visitExpression(node)

        return PyPositionalArgument(type_, starred=starred)

    # kwargs
    #   : kwargOrStarred (',' kwargOrStarred)* (',' kwargOrDoubleStarred)?
    #   | kwargOrDoubleStarred (',' kwargOrDoubleStarred)*;
    @_type_check
    @_visitor_guard
    def visitKwargs(
        self,
        ctx: PythonParser.KwargsContext,
        *,
        arguments: Optional[PyArguments] = None,
    ) -> PyArguments:
        if arguments is None:
            arguments = PyArguments()

        for node in ctx.kwargOrStarred():
            arg = self.visitKwargOrStarred(node)
            if isinstance(arg, PyKeywordArgument):
                arguments.kwargs.append(arg)
            else:
                arguments.args.append(arg)

        for node in ctx.kwargOrDoubleStarred():
            arg = self.visitKwargOrDoubleStarred(node)
            if isinstance(arg, PyKeywordArgument):
                arguments.kwargs.append(arg)
            else:
                arguments.double_stars.append(arg)

        return arguments

    # starredExpression
    #   : '*' expression;
    @_type_check
    @_visitor_guard
    def visitStarredExpression(
        self, ctx: PythonParser.StarredExpressionContext
    ) -> PyType:
        return self.visitExpression(ctx.expression())

    # kwargOrStarred
    #   : NAME '=' expression
    #   | starredExpression;
    @_type_check
    @_visitor_guard
    def visitKwargOrStarred(
        self, ctx: PythonParser.KwargOrStarredContext
    ) -> PyKeywordArgument | PyPositionalArgument:
        if node := ctx.NAME():
            name = self.visitName(node)
            type_ = self.visitExpression(ctx.expression())
            return PyKeywordArgument(name, type_)

        elif node := ctx.starredExpression():
            return self.visitStarredExpression(node)

    # kwargOrDoubleStarred
    #   : NAME '=' expression
    #   | '**' expression;
    @_type_check
    @_visitor_guard
    def visitKwargOrDoubleStarred(
        self, ctx: PythonParser.KwargOrDoubleStarredContext
    ) -> PyKeywordArgument | PyDoubleStarArgument:
        type_ = self.visitExpression(ctx.expression())

        if node := ctx.NAME():
            name = self.visitName(node)
            return PyKeywordArgument(name, type_)

        else:
            return PyDoubleStarArgument(type_)

    # starTargets: starTarget (',' starTarget)* ','?;
    @_type_check
    @_visitor_guard
    def visitStarTargets(
        self,
        ctx: PythonParser.StarTargetsContext,
        *,
        value_type: Optional[PyType] = None,
    ):
        for star_target in ctx.starTarget():
            # TODO: handle multiple targets
            self.visitStarTarget(star_target, value_type=value_type)

    # starTarget: '*'? targetWithStarAtom;
    @_type_check
    @_visitor_guard
    def visitStarTarget(
        self,
        ctx: PythonParser.StarTargetContext,
        *,
        value_type: Optional[PyType] = None,
    ):
        # TODO: handle the star
        self.visitTargetWithStarAtom(ctx.targetWithStarAtom(), value_type=value_type)

    # targetWithStarAtom
    #   : primary '.' NAME
    #   | primary '[' slices ']'
    #   | starAtom;
    @_type_check
    @_visitor_guard
    def visitTargetWithStarAtom(
        self,
        ctx: PythonParser.TargetWithStarAtomContext,
        *,
        value_type: Optional[PyType] = None,
    ):
        if star_atom := ctx.starAtom():
            return self.visitStarAtom(star_atom, value_type=value_type)

        type_ = self.visitPrimary(ctx.primary())

        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            self.context.define_attribute(
                type_, name, name_node, value_type=value_type.get_inferred_type()
            )

        elif slices := ctx.slices():
            self.visitSlices(slices)

    # starAtom
    #   : NAME
    #   | '(' starTarget ')'
    #   | '(' starTargets? ')'
    #   | '[' starTargets? ']';
    @_visitor_guard
    def visitStarAtom(
        self, ctx: PythonParser.StarAtomContext, *, value_type: Optional[PyType] = None
    ):
        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 1:
                self.context.define_variable(name, name_node)

            elif self.pass_num == 2:
                if value_type is not None:
                    symbol = self.context.current_scope[name]
                    self.context.set_variable_type(
                        symbol, value_type.get_inferred_type()
                    )

        elif star_target := ctx.starTarget():
            return self.visitStarTarget(star_target, value_type=value_type)

        elif star_targets := ctx.starTargets():
            return self.visitStarTargets(star_targets, value_type=value_type)

    # singleTarget
    #   : singleSubscriptAttributeTarget
    #   | NAME
    #   | '(' singleTarget ')';
    @_visitor_guard
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
                    self.context.define_variable(name, name_node)

            elif self.pass_num == 2:
                if value_type is not None:
                    # The type always derives from a type annotation. No further
                    # inference is performed.
                    symbol = self.context.current_scope[name]
                    self.context.set_variable_type(symbol, value_type)

                else:
                    self.context.access_variable(name, name_node)

        elif single_target := ctx.singleTarget():
            return self.visitSingleTarget(single_target, value_type=value_type)

        elif target := ctx.singleSubscriptAttributeTarget():
            return self.visitSingleSubscriptAttributeTarget(
                target, value_type=value_type
            )

    # singleSubscriptAttributeTarget
    #   : primary '.' NAME
    #   | primary '[' slices ']';
    @_type_check
    @_visitor_guard
    def visitSingleSubscriptAttributeTarget(
        self,
        ctx: PythonParser.SingleSubscriptAttributeTargetContext,
        *,
        value_type: Optional[PyType] = None,
    ):
        type_ = self.visitPrimary(ctx.primary())

        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if value_type is not None:
                self.context.define_attribute(
                    type_, name, name_node, value_type=value_type
                )

            else:
                self.context.access_attribute(type_, name, name_node)

        elif slices := ctx.slices():
            self.visitSlices(slices)

    # delTarget
    #   : primary '.' NAME
    #   | primary '[' slices ']'
    #   | delTargetAtom;
    @_visitor_guard
    def visitDelTarget(self, ctx: PythonParser.DelTargetContext):
        if atom := ctx.delTargetAtom():
            return self.visitDelTargetAtom(atom)

        type_ = self.visitPrimary(ctx.primary())

        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 2:
                # TODO: handle the deletion
                self.context.access_attribute(type_, name, name_node)

        elif slices := ctx.slices():
            self.visitSlices(slices)

    # delTargetAtom
    #   : NAME
    #   | '(' delTarget ')'
    #   | '(' delTargets? ')'
    #   | '[' delTargets? ']';
    @_visitor_guard
    def visitDelTargetAtom(self, ctx: PythonParser.DelTargetAtomContext):
        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 2:
                # TODO: handle the deletion
                self.context.access_variable(name, name_node)

        elif del_target := ctx.delTarget():
            return self.visitDelTarget(del_target)

        elif del_targets := ctx.delTargets():
            return self.visitDelTargets(del_targets)
