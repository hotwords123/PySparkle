from contextlib import nullcontext
from typing import NamedTuple, Optional

from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.tree.Tree import ErrorNode, TerminalNode

from grammar import PythonParserVisitor
from grammar.PythonParser import PythonParser

from .scope import PyDuplicateSymbolError, ScopeType, SymbolTable
from .structure import (
    PyImportFrom,
    PyImportFromAsName,
    PyImportFromTargets,
    PyImportName,
    PythonContext,
)
from .symbol import Symbol, SymbolType
from .token import TokenKind


def _visitor_guard(func):
    def wrapper(self: "PythonVisitor", ctx: ParserRuleContext, **kwargs):
        if ctx.exception is not None:
            return self.visitChildren(ctx)
        return func(self, ctx, **kwargs)

    return wrapper


def _first_pass_only(func):
    def wrapper(self: "PythonVisitor", ctx: ParserRuleContext, **kwargs):
        if self.pass_num == 1:
            return func(self, ctx, **kwargs)

    return wrapper


class _OuterSymbol(NamedTuple):
    symbol: Symbol
    scope: SymbolTable


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
        for symbol in self.context.global_scope.iter_symbols():
            if symbol.name.startswith("_"):
                symbol.public = False
            elif symbol.public is None:
                symbol.public = symbol.type is not SymbolType.IMPORTED

    def visitTerminal(self, node: TerminalNode):
        match node.getSymbol().type:
            case PythonParser.NAME:
                return self.visitName(node)

    def visitName(self, node: TerminalNode) -> str:
        """
        Returns:
            name: The name of the node.
        """
        return node.getText()

    def visitErrorNode(self, node: ErrorNode):
        if self.pass_num == 1:
            self.context.set_node_info(node, kind=TokenKind.ERROR)

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

    # classDef
    #   : decorators? 'class' NAME typeParams? ('(' arguments? ')')?
    #       ':' block;
    @_visitor_guard
    def visitClassDef(self, ctx: PythonParser.ClassDefContext):
        if decorators := ctx.decorators():
            self.visitDecorators(decorators)

        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if self.pass_num == 1:
            symbol = Symbol(SymbolType.CLASS, name, name_node)
            self.context.set_node_info(name_node, kind=TokenKind.CLASS, symbol=symbol)

            with self.context.wrap_errors(PyDuplicateSymbolError):
                self.context.current_scope.define(symbol)

        if arguments := ctx.arguments():
            self.visitArguments(arguments)

        with (
            self.context.new_scope(ctx, f"<class '{name}'>", ScopeType.CLASS)
            if self.pass_num == 1
            else self.context.scope_of(ctx)
        ):
            if type_params := ctx.typeParams():
                self.visitTypeParams(type_params)

            self.visitBlock(ctx.block())

    # functionDef
    #   : decorators? 'async'? 'def' NAME typeParams? '(' parameters? ')'
    #       ('->' expression)? ':' block;
    @_visitor_guard
    def visitFunctionDef(self, ctx: PythonParser.FunctionDefContext):
        if decorators := ctx.decorators():
            self.visitDecorators(decorators)

        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if self.pass_num == 1:
            symbol = Symbol(SymbolType.FUNCTION, name, name_node)
            self.context.set_node_info(
                name_node, kind=TokenKind.FUNCTION, symbol=symbol
            )

            with self.context.wrap_errors(PyDuplicateSymbolError):
                self.context.current_scope.define(symbol)

        if expression := ctx.expression():
            self.visitExpression(expression)

        with (
            self.context.new_scope(ctx, f"<function '{name}'>", ScopeType.LOCAL)
            if self.pass_num == 1
            else self.context.scope_of(ctx)
        ):
            if type_params := ctx.typeParams():
                self.visitTypeParams(type_params)

            if parameters := ctx.parameters():
                self.visitParameters(parameters)

            self.visitBlock(ctx.block())

    # param: NAME annotation?;
    @_visitor_guard
    def visitParam(self, ctx: PythonParser.ParamContext):
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if self.pass_num == 1:
            symbol = Symbol(SymbolType.PARAMETER, name, name_node)
            self.context.set_node_info(
                name_node, kind=TokenKind.VARIABLE, symbol=symbol
            )

            with self.context.wrap_errors(PyDuplicateSymbolError):
                self.context.current_scope.define(symbol)

        if annotation := ctx.annotation():
            with self.context.parent_scope():
                self.visitAnnotation(annotation)

    # paramStarAnnotation: NAME starAnnotation;
    @_visitor_guard
    def visitParamStarAnnotation(self, ctx: PythonParser.ParamStarAnnotationContext):
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if self.pass_num == 1:
            symbol = Symbol(SymbolType.PARAMETER, name, name_node)
            self.context.set_node_info(
                name_node, kind=TokenKind.VARIABLE, symbol=symbol
            )

            with self.context.wrap_errors(PyDuplicateSymbolError):
                self.context.current_scope.define(symbol)

        with self.context.parent_scope():
            self.visitStarAnnotation(ctx.starAnnotation())

    # default: '=' expression;
    @_visitor_guard
    def visitDefault(self, ctx: PythonParser.DefaultContext):
        with self.context.parent_scope():
            self.visitExpression(ctx.expression())

    # exceptBlock
    #   : 'except' (expression ('as' NAME)?)? ':' block;
    @_visitor_guard
    def visitExceptBlock(self, ctx: PythonParser.ExceptBlockContext):
        if expression := ctx.expression():
            self.visitExpression(expression)

        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 1:
                if name not in self.context.current_scope:
                    symbol = Symbol(SymbolType.VARIABLE, name, name_node)
                    self.context.current_scope.define(symbol)
                else:
                    symbol = self.context.current_scope[name]

                self.context.set_node_info(
                    name_node, kind=TokenKind.VARIABLE, symbol=symbol
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
                if name not in self.context.current_scope:
                    symbol = Symbol(SymbolType.VARIABLE, name, name_node)
                    self.context.current_scope.define(symbol)
                else:
                    symbol = self.context.current_scope[name]

                self.context.set_node_info(
                    name_node, kind=TokenKind.VARIABLE, symbol=symbol
                )

        self.visitBlock(ctx.block())

    # assignmentExpression: NAME ':=' expression;
    @_visitor_guard
    def visitAssignmentExpression(self, ctx: PythonParser.AssignmentExpressionContext):
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        self.visitExpression(ctx.expression())

        # According to PEP 572, an assignment expression occurring in comprehensions
        # binds the target in the containing scope, honoring a global or nonlocal
        # declaration if present.
        scope = self.context.current_scope
        while scope.scope_type is ScopeType.COMPREHENSION:
            assert (scope := scope.parent) is not None

        if self.pass_num == 1:
            if name not in scope:
                symbol = Symbol(SymbolType.VARIABLE, name, name_node)
                scope.define(symbol)
            else:
                symbol = scope[name]

            self.context.set_node_info(name_node, symbol=symbol)

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
    def visitAtom(self, ctx: PythonParser.AtomContext):
        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 2:
                if symbol := self.context.current_scope.lookup(name):
                    self.context.set_node_info(name_node, symbol=symbol)
                else:
                    self.context.set_node_info(name_node, kind=TokenKind.IDENTIFIER)

        else:
            return super().visitAtom(ctx)

    # lambdef
    #   : 'lambda' lambdaParameters? ':' expression;
    @_visitor_guard
    def visitLambdef(self, ctx: PythonParser.LambdefContext):
        if parameters := ctx.lambdaParameters():
            self.visitLambdaParameters(parameters)

        with (
            self.context.new_scope(ctx, "<lambda>", ScopeType.LAMBDA)
            if self.pass_num == 1
            else self.context.scope_of(ctx)
        ):
            self.visitExpression(ctx.expression())

    # lambdaParam: NAME;
    @_visitor_guard
    def visitLambdaParam(self, ctx: PythonParser.LambdaParamContext):
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if self.pass_num == 1:
            symbol = Symbol(SymbolType.PARAMETER, name, name_node)
            self.context.set_node_info(
                name_node, kind=TokenKind.VARIABLE, symbol=symbol
            )

            with self.context.wrap_errors(PyDuplicateSymbolError):
                self.context.current_scope.define(symbol)

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
        with self.context.parent_scope() if level == 0 else nullcontext():
            self.visitLogical(ctx.logical(0))

        # The if-clauses are evaluated in the current scope.
        for logical in ctx.logical()[1:]:
            self.visitLogical(logical)

    # listcomp: '[' namedExpression forIfClauses ']';
    @_visitor_guard
    def visitListcomp(self, ctx: PythonParser.ListcompContext):
        with (
            self.context.new_scope(ctx, "<listcomp>", ScopeType.COMPREHENSION)
            if self.pass_num == 1
            else self.context.scope_of(ctx)
        ):
            self.visitNamedExpression(ctx.namedExpression())
            self.visitForIfClauses(ctx.forIfClauses())

    # setcomp: '{' namedExpression forIfClauses '}';
    @_visitor_guard
    def visitSetcomp(self, ctx: PythonParser.SetcompContext):
        with (
            self.context.new_scope(ctx, "<setcomp>", ScopeType.COMPREHENSION)
            if self.pass_num == 1
            else self.context.scope_of(ctx)
        ):
            self.visitNamedExpression(ctx.namedExpression())
            self.visitForIfClauses(ctx.forIfClauses())

    # genexp: '(' namedExpression forIfClauses ')';
    @_visitor_guard
    def visitGenexp(self, ctx: PythonParser.GenexpContext):
        with (
            self.context.new_scope(ctx, "<genexp>", ScopeType.COMPREHENSION)
            if self.pass_num == 1
            else self.context.scope_of(ctx)
        ):
            self.visitNamedExpression(ctx.namedExpression())
            self.visitForIfClauses(ctx.forIfClauses())

    # dictcomp: '{' kvpair forIfClauses '}';
    @_visitor_guard
    def visitDictcomp(self, ctx: PythonParser.DictcompContext):
        with (
            self.context.new_scope(ctx, "<dictcomp>", ScopeType.COMPREHENSION)
            if self.pass_num == 1
            else self.context.scope_of(ctx)
        ):
            self.visitKvpair(ctx.kvpair())
            self.visitForIfClauses(ctx.forIfClauses())

    # starAtom
    #   : NAME
    #   | '(' starTarget ')'
    #   | '(' starTargets? ')'
    #   | '[' starTargets? ']';
    @_visitor_guard
    def visitStarAtom(self, ctx: PythonParser.StarAtomContext):
        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 1:
                if name not in self.context.current_scope:
                    symbol = Symbol(SymbolType.VARIABLE, name, name_node)
                    self.context.current_scope.define(symbol)
                else:
                    symbol = self.context.current_scope[name]

                self.context.set_node_info(name_node, symbol=symbol)

        else:
            return super().visitStarAtom(ctx)

    # singleTarget
    #   : singleSubscriptAttributeTarget
    #   | NAME
    #   | '(' singleTarget ')';
    @_visitor_guard
    def visitSingleTarget(self, ctx: PythonParser.SingleTargetContext):
        if name_node := ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 1:
                if name not in self.context.current_scope:
                    symbol = Symbol(SymbolType.VARIABLE, name, name_node)
                    self.context.current_scope.define(symbol)
                else:
                    symbol = self.context.current_scope[name]

                self.context.set_node_info(name_node, symbol=symbol)

        else:
            return super().visitSingleTarget(ctx)
