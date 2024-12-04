from contextlib import nullcontext
from typing import Literal

from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.tree.Tree import ErrorNode, TerminalNode

from grammar import PythonParserVisitor
from grammar.PythonParser import PythonParser

from .scope import PyDuplicateSymbolError, ScopeType
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
    def wrapper(self: PythonParserVisitor, ctx: ParserRuleContext, **kwargs):
        if ctx.exception is not None:
            return self.visitChildren(ctx)
        return func(self, ctx, **kwargs)

    return wrapper


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

    def fullVisit(self, tree):
        """
        Visits the entire parse tree for both passes.
        """
        self.pass_num = 1
        self.visit(tree)

        self.pass_num = 2
        self.visit(tree)

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
    @_visitor_guard
    def visitGlobalStmt(self, ctx: PythonParser.GlobalStmtContext):
        for name_node in ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 1:
                symbol = Symbol(SymbolType.GLOBAL, name, name_node)
                self.context.set_node_info(
                    name_node, kind=TokenKind.VARIABLE, symbol=symbol
                )

                with self.context.wrap_errors(PyDuplicateSymbolError):
                    self.context.current_scope.define(symbol)

            elif self.pass_num == 2:
                symbol = self.context.current_scope[name]

                if symbol.type is SymbolType.GLOBAL:
                    # Resolve the global variable in the global scope.
                    resolved = self.context.global_scope.get(name)
                    if resolved is not None and not resolved.is_outer():
                        symbol.target = resolved

    # nonlocalStmt: 'nonlocal' NAME (',' NAME)*;
    @_visitor_guard
    def visitNonlocalStmt(self, ctx: PythonParser.NonlocalStmtContext):
        for name_node in ctx.NAME():
            name = self.visitName(name_node)

            if self.pass_num == 1:
                symbol = Symbol(SymbolType.NONLOCAL, name, name_node)
                self.context.set_node_info(
                    name_node, kind=TokenKind.VARIABLE, symbol=symbol
                )

                with self.context.wrap_errors(PyDuplicateSymbolError):
                    self.context.current_scope.define(symbol)

            elif self.pass_num == 2:
                symbol = self.context.current_scope[name]

                if symbol.type is SymbolType.NONLOCAL:
                    # Resolve the nonlocal variable in an outer scope.
                    resolved = self.context.current_scope.parent.lookup(
                        name, globals=False
                    )
                    if resolved is not None and not resolved.is_outer():
                        symbol.target = resolved

    # importFrom
    #   : 'from' ('.' | '...')* dottedName 'import' importFromTargets
    #   | 'from' ('.' | '...')+ 'import' importFromTargets;
    @_visitor_guard
    def visitImportFrom(self, ctx: PythonParser.ImportFromContext):
        num_dots = len(ctx.DOT()) + 3 * len(ctx.ELLIPSIS())
        relative = num_dots - 1 if num_dots > 0 else None

        if dotted_name := ctx.dottedName():
            path = self.visitDottedName(dotted_name)
        else:
            path = []

        targets = self.visitImportFromTargets(ctx.importFromTargets())

        if self.pass_num == 1:
            self.context.imports.append(PyImportFrom(path, relative, targets))

        elif self.pass_num == 2:
            "TODO: Resolve the import."

    # importFromTargets
    #   : '(' importFromAsNames ','? ')'
    #   | importFromAsNames
    #   | '*';
    @_visitor_guard
    def visitImportFromTargets(
        self, ctx: PythonParser.ImportFromTargetsContext
    ) -> PyImportFromTargets:
        """
        Returns:
            targets: The import-from targets.
        """
        if as_names := ctx.importFromAsNames():
            return self.visitImportFromAsNames(as_names)
        else:
            return ...

    # importFromAsNames: importFromAsName (',' importFromAsName)*;
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
            aliased = True
            alias = self.visitName(alias_node)

            if self.pass_num == 1:
                # TODO: The name may refer to a module or an attribute.
                self.context.set_node_info(name_node, kind=TokenKind.VARIABLE)
        else:
            aliased = False
            alias_node, alias = name_node, name

        if self.pass_num == 1:
            # The as-name is defined in the current scope.
            symbol = Symbol(SymbolType.IMPORTED, alias, alias_node)
            # TODO: The name may refer to a module or an attribute.
            self.context.set_node_info(
                alias_node, kind=TokenKind.VARIABLE, symbol=symbol
            )

            with self.context.wrap_errors(PyDuplicateSymbolError):
                self.context.current_scope.define(symbol)

        return PyImportFromAsName(name, alias if aliased else None)

    # dottedAsName: dottedName ('as' NAME)?;
    @_visitor_guard
    def visitDottedAsName(self, ctx: PythonParser.DottedAsNameContext):
        if alias_node := ctx.NAME():
            path = self.visitDottedName(ctx.dottedName())
            alias = self.visitName(alias_node)

            if self.pass_num == 1:
                # The as-name is defined in the current scope.
                symbol = Symbol(SymbolType.IMPORTED, alias, alias_node)
                # The name always refers to a module.
                self.context.set_node_info(
                    alias_node, kind=TokenKind.MODULE, symbol=symbol
                )

                with self.context.wrap_errors(PyDuplicateSymbolError):
                    self.context.current_scope.define(symbol)

        else:
            path = self.visitDottedName(ctx.dottedName(), define=True)
            alias = None

        if self.pass_num == 1:
            self.context.imports.append(PyImportName(path, alias))

        elif self.pass_num == 2:
            "TODO: Resolve the import."

    # dottedName: dottedName '.' NAME | NAME;
    @_visitor_guard
    def visitDottedName(
        self, ctx: PythonParser.DottedNameContext, *, define: bool = False
    ) -> list[str]:
        """
        Args:
            define: Whether to define the name in the current scope.
                True if the dotted name is part of an import-name statement, and no
                as-name is provided.

        Returns:
            path: The list of module names in the dotted name.
        """
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if self.pass_num == 1:
            # The name always refers to a module.
            self.context.set_node_info(name_node, kind=TokenKind.MODULE)

        if dotted_name := ctx.dottedName():
            path = self.visitDottedName(dotted_name, define=define)
            path.append(name)
            return path

        else:
            if define and self.pass_num == 1:
                # The top-level module is defined in the current scope.
                symbol = Symbol(SymbolType.IMPORTED, name, name_node)
                self.context.set_node_info(name_node, symbol=symbol)

                with self.context.wrap_errors(PyDuplicateSymbolError):
                    self.context.current_scope.define(symbol)

            return [name]

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

            self.context.set_node_info(
                name_node, kind=TokenKind.VARIABLE, symbol=symbol
            )

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
                    self.context.set_node_info(
                        name_node,
                        kind=self.token_kind_for_symbol(symbol),
                        symbol=symbol,
                    )
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

                self.context.set_node_info(
                    name_node,
                    kind=self.token_kind_for_symbol(symbol),
                    symbol=symbol,
                )

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

                self.context.set_node_info(
                    name_node,
                    kind=self.token_kind_for_symbol(symbol),
                    symbol=symbol,
                )

        else:
            return super().visitSingleTarget(ctx)

    @staticmethod
    def token_kind_for_symbol(symbol: Symbol) -> TokenKind:
        if symbol.is_outer() and symbol.target is not None:
            symbol = symbol.target

        match symbol.type:
            case (
                SymbolType.VARIABLE
                | SymbolType.PARAMETER
                | SymbolType.IMPORTED
                | SymbolType.GLOBAL
                | SymbolType.NONLOCAL
            ):
                return TokenKind.VARIABLE
            case SymbolType.FUNCTION:
                return TokenKind.FUNCTION
            case SymbolType.CLASS:
                return TokenKind.CLASS
            case _:
                return TokenKind.NONE
