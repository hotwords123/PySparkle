from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.tree.Tree import ErrorNode, TerminalNode

from grammar import PythonParserVisitor
from grammar.PythonParser import PythonParser

from .scope import DuplicateSymbolError, ScopeType
from .structure import PythonContext
from .symbol import Symbol, SymbolType
from .token import TokenInfo, TokenKind


def _visitor_guard(func):
    def wrapper(self: PythonParserVisitor, ctx: ParserRuleContext, **kwargs):
        if ctx.exception is not None:
            return self.visitChildren(ctx)
        return func(self, ctx, **kwargs)

    return wrapper


class PythonVisitorFirstPass(PythonParserVisitor):
    """
    First pass visitor to build the symbol table.
    """

    def __init__(self, context: PythonContext):
        self.context = context

    def visitTerminal(self, node: TerminalNode):
        match node.getSymbol().type:
            case PythonParser.NAME:
                return self.visitName(node)

    def visitName(self, node: TerminalNode) -> str:
        return node.getText()

    def visitErrorNode(self, node: ErrorNode):
        self.context.set_node_info(node, TokenInfo(kind=TokenKind.ERROR))

    # globalStmt: 'global' NAME (',' NAME)*;
    @_visitor_guard
    def visitGlobalStmt(self, ctx: PythonParser.GlobalStmtContext):
        for name_node in ctx.NAME():
            name = self.visitName(name_node)
            try:
                symbol = Symbol(SymbolType.GLOBAL, name, name_node)
                self.context.current_scope.define(symbol)
            except DuplicateSymbolError as e:
                self.context.errors.append(e)

    # nonlocalStmt: 'nonlocal' NAME (',' NAME)*;
    @_visitor_guard
    def visitNonlocalStmt(self, ctx: PythonParser.NonlocalStmtContext):
        for name_node in ctx.NAME():
            name = self.visitName(name_node)
            try:
                symbol = Symbol(SymbolType.NONLOCAL, name, name_node)
                self.context.current_scope.define(symbol)
            except DuplicateSymbolError as e:
                self.context.errors.append(e)

    # importFromAsName: NAME ('as' NAME)?;
    @_visitor_guard
    def visitImportFromAsName(self, ctx: PythonParser.ImportFromAsNameContext):
        name_node = ctx.NAME(0)
        name = self.visitName(name_node)

        if as_name_node := ctx.NAME(1):
            as_name = self.visitName(as_name_node)
        else:
            as_name_node, as_name = name_node, name

        try:
            symbol = Symbol(SymbolType.IMPORTED, as_name, as_name_node)
            self.context.current_scope.define(symbol)
        except DuplicateSymbolError as e:
            self.context.errors.append(e)

    # dottedAsName: dottedName ('as' NAME)?;
    @_visitor_guard
    def visitDottedAsName(self, ctx: PythonParser.DottedAsNameContext):
        if as_name_node := ctx.NAME():
            self.visitDottedName(ctx.dottedName(), renamed=True)

            as_name = self.visitName(as_name_node)
            try:
                symbol = Symbol(SymbolType.MODULE, as_name, as_name_node)
                self.context.current_scope.define(symbol)
            except DuplicateSymbolError as e:
                self.context.errors.append(e)

        else:
            self.visitDottedName(ctx.dottedName())

    # dottedName: dottedName '.' NAME | NAME;
    @_visitor_guard
    def visitDottedName(
        self, ctx: PythonParser.DottedNameContext, renamed: bool = False
    ):
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if dotted_name := ctx.dottedName():
            self.visitDottedName(dotted_name)

        elif not renamed:
            try:
                symbol = Symbol(SymbolType.MODULE, name, name_node)
                self.context.current_scope.define(symbol)
            except DuplicateSymbolError as e:
                self.context.errors.append(e)

    # classDef
    #   : decorators? 'class' NAME typeParams? ('(' arguments? ')')?
    #       ':' block;
    @_visitor_guard
    def visitClassDef(self, ctx: PythonParser.ClassDefContext):
        if decorators := ctx.decorators():
            self.visitDecorators(decorators)

        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if arguments := ctx.arguments():
            self.visitArguments(arguments)

        with self.context.new_scope(ctx, f"<class '{name}'>", ScopeType.CLASS):
            if type_params := ctx.typeParams():
                self.visitTypeParams(type_params)

            self.visitBlock(ctx.block())

        try:
            symbol = Symbol(SymbolType.CLASS, name, name_node)
            self.context.current_scope.define(symbol)
        except DuplicateSymbolError as e:
            self.context.errors.append(e)

    # functionDef
    #   : decorators? 'async'? 'def' NAME typeParams? '(' parameters? ')'
    #       ('->' expression)? ':' block;
    @_visitor_guard
    def visitFunctionDef(self, ctx: PythonParser.FunctionDefContext):
        if decorators := ctx.decorators():
            self.visitDecorators(decorators)

        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if expression := ctx.expression():
            self.visitExpression(expression)

        with self.context.new_scope(ctx, f"<function '{name}'>", ScopeType.LOCAL):
            if type_params := ctx.typeParams():
                self.visitTypeParams(type_params)

            if parameters := ctx.parameters():
                self.visitParameters(parameters)

            self.visitBlock(ctx.block())

        try:
            symbol = Symbol(SymbolType.FUNCTION, name, name_node)
            self.context.current_scope.define(symbol)
        except DuplicateSymbolError as e:
            self.context.errors.append(e)

    # param: NAME annotation?;
    @_visitor_guard
    def visitParam(self, ctx: PythonParser.ParamContext):
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        if annotation := ctx.annotation():
            with self.context.parent_scope():
                self.visitAnnotation(annotation)

        try:
            symbol = Symbol(SymbolType.PARAMETER, name, name_node)
            self.context.current_scope.define(symbol)
        except DuplicateSymbolError as e:
            self.context.errors.append(e)

    # paramStarAnnotation: NAME starAnnotation;
    @_visitor_guard
    def visitParamStarAnnotation(self, ctx: PythonParser.ParamStarAnnotationContext):
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        with self.context.parent_scope():
            self.visitStarAnnotation(ctx.starAnnotation())

        try:
            symbol = Symbol(SymbolType.PARAMETER, name, name_node)
            self.context.current_scope.define(symbol)
        except DuplicateSymbolError as e:
            self.context.errors.append(e)

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
            if name not in self.context.current_scope:
                symbol = Symbol(SymbolType.VARIABLE, name, name_node)
                self.context.current_scope.define(symbol)

        self.visitBlock(ctx.block())

    # exceptStarBlock
    #   : 'except' '*' expression ('as' NAME)? ':' block;
    @_visitor_guard
    def visitExceptStarBlock(self, ctx: PythonParser.ExceptStarBlockContext):
        self.visitExpression(ctx.expression())

        if name_node := ctx.NAME():
            name = self.visitName(name_node)
            if name not in self.context.current_scope:
                symbol = Symbol(SymbolType.VARIABLE, name, name_node)
                self.context.current_scope.define(symbol)

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
            scope = scope.parent

        if name not in scope:
            symbol = Symbol(SymbolType.VARIABLE, name, name_node)
            scope.define(symbol)

    # lambdef
    #   : 'lambda' lambdaParameters? ':' expression;
    @_visitor_guard
    def visitLambdef(self, ctx: PythonParser.LambdefContext):
        if parameters := ctx.lambdaParameters():
            self.visitLambdaParameters(parameters)

        with self.context.new_scope(ctx, "<lambda>", ScopeType.LAMBDA):
            self.visitExpression(ctx.expression())

    # lambdaParam: NAME;
    @_visitor_guard
    def visitLambdaParam(self, ctx: PythonParser.LambdaParamContext):
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        try:
            symbol = Symbol(SymbolType.VARIABLE, name, name_node)
            self.context.current_scope.define(symbol)
        except DuplicateSymbolError as e:
            self.context.errors.append(e)

    # forIfClauses: forIfClause+;
    @_visitor_guard
    def visitForIfClauses(self, ctx: PythonParser.ForIfClausesContext):
        for i, clause in enumerate(ctx.forIfClause()):
            self.visitForIfClause(clause, level=i)

    # listcomp: '[' namedExpression forIfClauses ']';
    @_visitor_guard
    def visitListcomp(self, ctx: PythonParser.ListcompContext):
        self.visitNamedExpression(ctx.namedExpression())
        self.visitForIfClauses(ctx.forIfClauses())

    # forIfClause
    #   : 'async'? 'for' starTargets 'in' logical ('if' logical)*;
    @_visitor_guard
    def visitForIfClause(self, ctx: PythonParser.ForIfClauseContext, level: int = 0):
        self.visitStarTargets(ctx.starTargets())

        if level == 0:
            # The first iterable is evaluated in the enclosing scope.
            with self.context.parent_scope():
                self.visitLogical(ctx.logical(0))
        else:
            # The remaining iterables are evaluated in the current scope.
            self.visitLogical(ctx.logical(0))

        # The if-clauses are evaluated in the current scope.
        for logical in ctx.logical()[1:]:
            self.visitLogical(logical)

    # listcomp: '[' namedExpression forIfClauses ']';
    @_visitor_guard
    def visitListcomp(self, ctx: PythonParser.ListcompContext):
        with self.context.new_scope(ctx, "<listcomp>", ScopeType.COMPREHENSION):
            self.visitNamedExpression(ctx.namedExpression())
            self.visitForIfClauses(ctx.forIfClauses())

    # setcomp: '{' namedExpression forIfClauses '}';
    @_visitor_guard
    def visitSetcomp(self, ctx: PythonParser.SetcompContext):
        with self.context.new_scope(ctx, "<setcomp>", ScopeType.COMPREHENSION):
            self.visitNamedExpression(ctx.namedExpression())
            self.visitForIfClauses(ctx.forIfClauses())

    # genexp: '(' namedExpression forIfClauses ')';
    @_visitor_guard
    def visitGenexp(self, ctx: PythonParser.GenexpContext):
        with self.context.new_scope(ctx, "<genexp>", ScopeType.COMPREHENSION):
            self.visitNamedExpression(ctx.namedExpression())
            self.visitForIfClauses(ctx.forIfClauses())

    # dictcomp: '{' kvpair forIfClauses '}';
    @_visitor_guard
    def visitDictcomp(self, ctx: PythonParser.DictcompContext):
        with self.context.new_scope(ctx, "<dictcomp>", ScopeType.COMPREHENSION):
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
            if name not in self.context.current_scope:
                symbol = Symbol(SymbolType.VARIABLE, name, name_node)
                self.context.current_scope.define(symbol)

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
            if name not in self.context.current_scope:
                symbol = Symbol(SymbolType.VARIABLE, name, name_node)
                self.context.current_scope.define(symbol)

        else:
            return super().visitSingleTarget(ctx)


class PythonVisitorSecondPass(PythonParserVisitor):
    """
    Second pass visitor to resolve names and types.
    """

    def __init__(self, context: PythonContext):
        self.context = context

    def visitTerminal(self, node: TerminalNode):
        match node.getSymbol().type:
            case PythonParser.NAME:
                return self.visitName(node)

    def visitName(self, node: TerminalNode) -> str:
        return node.getText()

    # globalStmt: 'global' NAME (',' NAME)*;
    @_visitor_guard
    def visitGlobalStmt(self, ctx: PythonParser.GlobalStmtContext):
        for name_node in ctx.NAME():
            name = self.visitName(name_node)
            symbol = self.context.current_scope[name]

            self.context.set_node_info(
                name_node, TokenInfo(kind=TokenKind.VARIABLE, symbol=symbol)
            )

            if symbol.type is SymbolType.GLOBAL:
                resolved = self.context.global_scope.get(name)
                if resolved is not None and not resolved.is_outer():
                    symbol.target = resolved

    # nonlocalStmt: 'nonlocal' NAME (',' NAME)*;
    @_visitor_guard
    def visitNonlocalStmt(self, ctx: PythonParser.NonlocalStmtContext):
        for name_node in ctx.NAME():
            name = self.visitName(name_node)
            symbol = self.context.current_scope[name]

            self.context.set_node_info(
                name_node, TokenInfo(kind=TokenKind.VARIABLE, symbol=symbol)
            )

            if symbol.type is SymbolType.NONLOCAL:
                resolved = self.context.current_scope.parent.lookup(name, globals=False)
                if resolved is not None and not resolved.is_outer():
                    symbol.target = resolved

    # importFromAsName: NAME ('as' NAME)?;
    @_visitor_guard
    def visitImportFromAsName(self, ctx: PythonParser.ImportFromAsNameContext):
        name_node = ctx.NAME(0)
        name = self.visitName(name_node)

        if as_name_node := ctx.NAME(1):
            as_name = self.visitName(as_name_node)
        else:
            as_name_node, as_name = name_node, name

            self.context.set_node_info(name_node, TokenInfo(kind=TokenKind.VARIABLE))

        symbol = self.context.current_scope[as_name]
        self.context.set_node_info(
            as_name_node, TokenInfo(kind=TokenKind.VARIABLE, symbol=symbol)
        )

    # dottedAsName: dottedName ('as' NAME)?;
    @_visitor_guard
    def visitDottedAsName(self, ctx: PythonParser.DottedAsNameContext):
        if as_name_node := ctx.NAME():
            self.visitDottedName(ctx.dottedName(), renamed=True)

            as_name = self.visitName(as_name_node)
            symbol = self.context.current_scope[as_name]

            self.context.set_node_info(
                as_name_node, TokenInfo(kind=TokenKind.MODULE, symbol=symbol)
            )

        else:
            self.visitDottedName(ctx.dottedName())

    # dottedName: dottedName '.' NAME | NAME;
    @_visitor_guard
    def visitDottedName(
        self, ctx: PythonParser.DottedNameContext, renamed: bool = False
    ):
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        info = TokenInfo(kind=TokenKind.MODULE)

        if dotted_name := ctx.dottedName():
            self.visitDottedName(dotted_name)

        elif not renamed:
            symbol = self.context.current_scope[name]
            info.symbol = symbol

        self.context.set_node_info(name_node, info)

    # classDef
    #   : decorators? 'class' NAME typeParams? ('(' arguments? ')')?
    #       ':' block;
    @_visitor_guard
    def visitClassDef(self, ctx: PythonParser.ClassDefContext):
        if decorators := ctx.decorators():
            self.visitDecorators(decorators)

        name_node = ctx.NAME()
        name = self.visitName(name_node)

        symbol = self.context.current_scope[name]

        self.context.set_node_info(
            name_node, TokenInfo(kind=TokenKind.CLASS, symbol=symbol)
        )

        if arguments := ctx.arguments():
            self.visitArguments(arguments)

        with self.context.scope_of(ctx):
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

        symbol = self.context.current_scope[name]

        self.context.set_node_info(
            name_node, TokenInfo(kind=TokenKind.FUNCTION, symbol=symbol)
        )

        if expression := ctx.expression():
            self.visitExpression(expression)

        with self.context.scope_of(ctx):
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

        symbol = self.context.current_scope[name]

        self.context.set_node_info(
            name_node, TokenInfo(kind=TokenKind.VARIABLE, symbol=symbol)
        )

        if annotation := ctx.annotation():
            with self.context.parent_scope():
                self.visitAnnotation(annotation)

    # paramStarAnnotation: NAME starAnnotation;
    @_visitor_guard
    def visitParamStarAnnotation(self, ctx: PythonParser.ParamStarAnnotationContext):
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        symbol = self.context.current_scope[name]

        self.context.set_node_info(
            name_node, TokenInfo(kind=TokenKind.VARIABLE, symbol=symbol)
        )

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
            symbol = self.context.current_scope[name]

            self.context.set_node_info(
                name_node, TokenInfo(kind=TokenKind.VARIABLE, symbol=symbol)
            )

        self.visitBlock(ctx.block())

    # exceptStarBlock
    #   : 'except' '*' expression ('as' NAME)? ':' block;
    @_visitor_guard
    def visitExceptStarBlock(self, ctx: PythonParser.ExceptStarBlockContext):
        self.visitExpression(ctx.expression())

        if name_node := ctx.NAME():
            name = self.visitName(name_node)
            symbol = self.context.current_scope[name]

            self.context.set_node_info(
                name_node, TokenInfo(kind=TokenKind.VARIABLE, symbol=symbol)
            )

        self.visitBlock(ctx.block())

    # assignmentExpression: NAME ':=' expression;
    @_visitor_guard
    def visitAssignmentExpression(self, ctx: PythonParser.AssignmentExpressionContext):
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        scope = self.context.current_scope
        while scope.scope_type is ScopeType.COMPREHENSION:
            scope = scope.parent

        symbol = scope[name]

        self.context.set_node_info(
            name_node, TokenInfo(kind=TokenKind.VARIABLE, symbol=symbol)
        )

        self.visitExpression(ctx.expression())

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

            if symbol := self.context.current_scope.lookup(name):
                info = TokenInfo(
                    kind=_symbol_type_to_token_kind(symbol.type), symbol=symbol
                )
            else:
                info = TokenInfo(kind=TokenKind.IDENTIFIER)

            self.context.set_node_info(name_node, info)

        else:
            return super().visitAtom(ctx)

    # lambdef
    #   : 'lambda' lambdaParameters? ':' expression;
    @_visitor_guard
    def visitLambdef(self, ctx: PythonParser.LambdefContext):
        if parameters := ctx.lambdaParameters():
            self.visitLambdaParameters(parameters)

        with self.context.scope_of(ctx):
            self.visitExpression(ctx.expression())

    # lambdaParam: NAME;
    @_visitor_guard
    def visitLambdaParam(self, ctx: PythonParser.LambdaParamContext):
        name_node = ctx.NAME()
        name = self.visitName(name_node)

        symbol = self.context.current_scope[name]

        self.context.set_node_info(
            name_node, TokenInfo(kind=TokenKind.VARIABLE, symbol=symbol)
        )

    # forIfClauses: forIfClause+;
    @_visitor_guard
    def visitForIfClauses(self, ctx: PythonParser.ForIfClausesContext):
        for i, clause in enumerate(ctx.forIfClause()):
            self.visitForIfClause(clause, level=i)

    # forIfClause
    #   : 'async'? 'for' starTargets 'in' logical ('if' logical)*;
    @_visitor_guard
    def visitForIfClause(self, ctx: PythonParser.ForIfClauseContext, level: int = 0):
        self.visitStarTargets(ctx.starTargets())

        if level == 0:
            with self.context.parent_scope():
                self.visitLogical(ctx.logical(0))
        else:
            self.visitLogical(ctx.logical(0))

        for logical in ctx.logical()[1:]:
            self.visitLogical(logical)

    # listcomp: '[' namedExpression forIfClauses ']';
    @_visitor_guard
    def visitListcomp(self, ctx: PythonParser.ListcompContext):
        with self.context.scope_of(ctx):
            self.visitNamedExpression(ctx.namedExpression())
            self.visitForIfClauses(ctx.forIfClauses())

    # setcomp: '{' namedExpression forIfClauses '}';
    @_visitor_guard
    def visitSetcomp(self, ctx: PythonParser.SetcompContext):
        with self.context.scope_of(ctx):
            self.visitNamedExpression(ctx.namedExpression())
            self.visitForIfClauses(ctx.forIfClauses())

    # genexp: '(' namedExpression forIfClauses ')';
    @_visitor_guard
    def visitGenexp(self, ctx: PythonParser.GenexpContext):
        with self.context.scope_of(ctx):
            self.visitNamedExpression(ctx.namedExpression())
            self.visitForIfClauses(ctx.forIfClauses())

    # dictcomp: '{' kvpair forIfClauses '}';
    @_visitor_guard
    def visitDictcomp(self, ctx: PythonParser.DictcompContext):
        with self.context.scope_of(ctx):
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
            symbol = self.context.current_scope.lookup(name)
            assert symbol is not None

            self.context.set_node_info(
                name_node,
                TokenInfo(kind=_symbol_type_to_token_kind(symbol.type), symbol=symbol),
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
            symbol = self.context.current_scope.lookup(name)
            assert symbol is not None

            self.context.set_node_info(
                name_node,
                TokenInfo(kind=_symbol_type_to_token_kind(symbol.type), symbol=symbol),
            )

        else:
            return super().visitSingleTarget(ctx)


def _symbol_type_to_token_kind(symbol_type: SymbolType) -> TokenKind:
    match symbol_type:
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
        case SymbolType.MODULE:
            return TokenKind.MODULE
        case _:
            return TokenKind.NONE
