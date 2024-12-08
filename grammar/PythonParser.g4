parser grammar PythonParser;

options {
    tokenVocab=PythonLexer;
}

// https://docs.python.org/3/reference/grammar.html

// STARTING RULES
// ==============

file: statements? EOF;

// GENERAL STATEMENTS
// ==================

statements: statement+;

statement
    : compoundStmt
    | simpleStmts
    | invalidBlock
    | invalidToken NEWLINE;

simpleStmts: simpleStmt (';' simpleStmt)* (';')? NEWLINE;

simpleStmt
    : assignment
    // | typeAlias
    | starExpressions
    | returnStmt
    | importStmt
    | raiseStmt
    | 'pass'
    | delStmt
    | yieldStmt
    | assertStmt
    | 'break'
    | 'continue'
    | globalStmt
    | nonlocalStmt
    ;

compoundStmt
    : functionDef
    | ifStmt
    | classDef
    | withStmt
    | forStmt
    | tryStmt
    | whileStmt
    // | matchStmt
    ;

invalidBlock: INDENT statements DEDENT;

invalidToken
    : 'else'
    | 'except'
    | 'in'
    | 'finally'
    | 'is'
    | 'and'
    | 'as'
    | 'elif'
    | 'or'
    | '/'
    | '//'
    | '%'
    | '<<'
    | '>>'
    | '&'
    | '|'
    | '^'
    | ':='
    | '<'
    | '>'
    | '<='
    | '>='
    | '=='
    | '!='
    | ')'
    | ']'
    | '}'
    | ','
    | ':'
    | '!'
    | '.'
    | ';'
    | '='
    | '->'
    | '+='
    | '-='
    | '*='
    | '/='
    | '//='
    | '%='
    | '@='
    | '&='
    | '|='
    | '^='
    | '>>='
    | '<<='
    | '**='
    | ERROR
    ;

// SIMPLE STATEMENTS
// =================

assignment
    : singleTarget ':' expression ('=' assignmentRhs)?  # annotatedAssignment
    | (starTargets '=')+ assignmentRhs                  # starredAssignment
    | singleTarget augassign assignmentRhs              # augmentedAssignment
    ;

assignmentRhs: yieldExpr | starExpressions;

augassign
    : '+='
    | '-='
    | '*='
    | '@='
    | '/='
    | '%='
    | '&='
    | '|='
    | '^='
    | '<<='
    | '>>='
    | '**='
    | '//=';

returnStmt: 'return' starExpressions?;

raiseStmt: 'raise' (expression ('from' expression)?)?;

globalStmt: 'global' NAME (',' NAME)*;

nonlocalStmt: 'nonlocal' NAME (',' NAME)*;

delStmt: 'del' delTargets;

yieldStmt: yieldExpr;

assertStmt: 'assert' expression (',' expression)?;

importStmt: importName | importFrom;

// Import statements
// -----------------

importName: 'import' dottedAsNames;
importFrom
    : 'from' ('.' | '...')* dottedName 'import' importFromTargets
    | 'from' ('.' | '...')+ 'import' importFromTargets;
importFromTargets
    : '(' importFromAsNames ','? ')'
    | importFromAsNames
    | '*';
importFromAsNames
    : importFromAsName (',' importFromAsName)*;
importFromAsName
    : NAME ('as' NAME)?;
dottedAsNames
    : dottedAsName (',' dottedAsName)*;
dottedAsName
    : dottedName ('as' NAME)?;
dottedName
    : dottedName '.' NAME
    | NAME;

// COMPOUND STATEMENTS
// ===================

// Common elements
// ---------------

block
    : NEWLINE INDENT statements DEDENT
    | simpleStmts;

decorators: ('@' namedExpression NEWLINE)+;

// Class definitions
// -----------------

classDef
    : decorators? 'class' NAME typeParams? ('(' arguments? ')')?
        ':' block;

// Function definitions
// --------------------

functionDef
    : decorators? 'async'? 'def' NAME typeParams? '(' parameters? ')'
        ('->' expression)? ':' block;

// Function parameters
// -------------------

parameters
    : slashNoDefault (',' paramNoDefault)* (',' paramWithDefault)*
        (',' starEtc?)?
    | slashWithDefault (',' paramWithDefault)*
        (',' starEtc?)?
    | paramNoDefault (',' paramNoDefault)* (',' paramWithDefault)*
        (',' starEtc?)?
    | paramWithDefault (',' paramWithDefault)*
        (',' starEtc?)?
    | starEtc;

slashNoDefault
    : paramNoDefault (',' paramNoDefault)* ',' '/';
slashWithDefault
    : paramNoDefault (',' paramNoDefault)* (',' paramWithDefault)+ ',' '/'
    | paramWithDefault (',' paramWithDefault)* ',' '/';

starEtc
    : '*' (paramNoDefault | paramNoDefaultStarAnnotation)
        (',' paramMaybeDefault)* (',' kwds?)?
    | '*' (',' paramMaybeDefault)+ (',' kwds?)?
    | kwds;

kwds: '**' paramNoDefault ','?;

paramNoDefault: param;
paramNoDefaultStarAnnotation: paramStarAnnotation;
paramWithDefault: param default;
paramMaybeDefault: param default?;

param: NAME annotation?;
paramStarAnnotation: NAME starAnnotation;
annotation: ':' expression;
starAnnotation: ':' starredExpression;
default: '=' expression;

// If statement
// ------------

ifStmt
    : 'if' namedExpression ':' block (elifStmt | elseBlock)?;
elifStmt
    : 'elif' namedExpression ':' block (elifStmt | elseBlock)?;
elseBlock
    : 'else' ':' block;

// While statement
// ---------------

whileStmt
    : 'while' namedExpression ':' block elseBlock?;

// For statement
// -------------

forStmt
    : 'async'? 'for' starTargets 'in' starExpressions ':' block elseBlock?;

// With statement
// --------------

withStmt
    : 'async'? 'with' '(' withItem (',' withItem)* ','? ')' ':' block
    | 'async'? 'with' withItem (',' withItem)* ':' block;

withItem: expression ('as' starTarget)?;

// Try statement
// -------------

tryStmt
    : 'try' ':' block (finallyBlock
        | exceptBlock+ elseBlock? finallyBlock?
        | exceptStarBlock+ elseBlock? finallyBlock?);

// Except statement
// ----------------

exceptBlock
    : 'except' (expression ('as' NAME)?)? ':' block;
exceptStarBlock
    : 'except' '*' expression ('as' NAME)? ':' block;
finallyBlock
    : 'finally' ':' block;

// Match statement
// ---------------

// TODO

// Type statement
// --------------

// typeAlias
//     : 'type' NAME '=' expression;

// Type parameter declaration
// --------------------------

typeParams
    : '[' typeParam (',' typeParam)* ','? ']';

typeParam
    : NAME typeParamBound? typeParamDefault?
    | '*' NAME typeParamStarredDefault?
    | '**' NAME typeParamDefault?;

typeParamBound: ':' expression;
typeParamDefault: '=' expression;
typeParamStarredDefault: '=' starExpression;

// EXPRESSIONS
// ===========

expressions
    : expression (',' expression)* ','?;

expression
    : logical 'if' logical 'else' expression
    | logical
    | lambdef
    ;

yieldExpr
    : 'yield' ('from' expression | starExpressions)?;

starExpressions
    : starExpression (',' starExpression)* ','?;

starExpression
    : '*' bitwise
    | expression;

starNamedExpressions
    : starNamedExpression (',' starNamedExpression)* ','?;

starNamedExpression
    : '*' bitwise
    | namedExpression;

assignmentExpression
    : NAME ':=' expression;

namedExpression
    : assignmentExpression
    | expression;

logical
    : 'not' logical
    | logical 'and' logical
    | logical 'or' logical
    | comparison;

// Comparison operators
// --------------------

comparison
    : bitwise compareOpBitwisePair*;

compareOpBitwisePair
    : compareOp bitwise;

compareOp
    : '==' | '!=' | '<=' | '<' | '>=' | '>'
    | 'in' | 'not' 'in' | 'is' | 'is' 'not';

// Bitwise operators
// -----------------

bitwise
    : bitwise ('<<' | '>>') bitwise
    | bitwise '&' bitwise
    | bitwise '^' bitwise
    | bitwise '|' bitwise
    | arithmetic;

// Arithmetic operators
// --------------------

arithmetic
    :<assoc=right> arithmetic '**' arithmetic
    | ('+' | '-' | '~') arithmetic
    | arithmetic ('*' | '/' | '//' | '%' | '@') arithmetic
    | arithmetic ('+' | '-') arithmetic
    | awaitPrimary;

// Primary elements
// ----------------

awaitPrimary
    : 'await' primary
    | primary;

primary
    : primary '.' NAME
    | primary genexp
    | primary '(' arguments? ')'
    | primary '[' slices ']'
    | atom;

slices: slice (',' slice)* ','?;

slice
    : expression? ':' expression? (':' expression)?
    | namedExpression
    | starredExpression;

atom
    : NAME
    | 'True'
    | 'False'
    | 'None'
    | strings
    | number
    | (tuple | group | genexp)
    | (list | listcomp)
    | (dict | set | dictcomp | setcomp)
    | '...';

group
    : '(' (yieldExpr | namedExpression) ')';

// Lambda functions
// ----------------

lambdef
    : 'lambda' lambdaParameters? ':' expression;

lambdaParameters
    : lambdaSlashNoDefault (',' lambdaParamNoDefault)* (',' lambdaParamWithDefault)*
        (',' lambdaStarEtc?)?
    | lambdaSlashWithDefault (',' lambdaParamWithDefault)*
        (',' lambdaStarEtc?)?
    | lambdaParamNoDefault (',' lambdaParamNoDefault)* (',' lambdaParamWithDefault)*
        (',' lambdaStarEtc?)?
    | lambdaParamWithDefault (',' lambdaParamWithDefault)*
        (',' lambdaStarEtc?)?
    | lambdaStarEtc;

lambdaSlashNoDefault
    : lambdaParamNoDefault (',' lambdaParamNoDefault)* ',' '/';

lambdaSlashWithDefault
    : lambdaParamNoDefault (',' lambdaParamNoDefault)* (',' lambdaParamWithDefault)+ ',' '/'
    | lambdaParamWithDefault (',' lambdaParamWithDefault)* ',' '/';

lambdaStarEtc
    : '*' lambdaParamNoDefault (',' lambdaParamMaybeDefault)* (',' lambdaKwds?)?
    | '*' (',' lambdaParamMaybeDefault)+ (',' lambdaKwds?)?
    | lambdaKwds;

lambdaKwds: '**' lambdaParamNoDefault ','?;

lambdaParamNoDefault: lambdaParam;
lambdaParamWithDefault: lambdaParam default;
lambdaParamMaybeDefault: lambdaParam default?;
lambdaParam: NAME;

// LITERALS
// ========

// TODO: f-strings

string: STRING_LITERAL | BYTES_LITERAL;
strings: string+;

number: INTEGER | FLOAT_NUMBER | IMAG_NUMBER;

list
    : '[' starNamedExpressions? ']';

tuple
    : '(' (starNamedExpression ',' starNamedExpressions?)? ')';

set
    : '{' starNamedExpressions '}';

// Dicts
// -----

dict
    : '{' doubleStarredKvpairs? '}';

doubleStarredKvpairs
    : doubleStarredKvpair (',' doubleStarredKvpair)* ','?;

doubleStarredKvpair
    : '**' bitwise
    | kvpair;

kvpair: expression ':' expression;

// Comprenhensions & Generators
// ----------------------------

forIfClauses: forIfClause+;

forIfClause
    : 'async'? 'for' starTargets 'in' logical ('if' logical)*;

listcomp
    : '[' namedExpression forIfClauses ']';

setcomp
    : '{' namedExpression forIfClauses '}';

genexp
    : '(' namedExpression forIfClauses ')';

dictcomp
    : '{' kvpair forIfClauses '}';

// FUNCTION CALL ARGUMENTS
// =======================

arguments
    : args ','?;

args
    : arg (',' arg)* (',' kwargs)?
    | kwargs;

arg
    : starredExpression
    | assignmentExpression
    | expression;

kwargs
    : kwargOrStarred (',' kwargOrStarred)* (',' kwargOrDoubleStarred)?
    | kwargOrDoubleStarred (',' kwargOrDoubleStarred)*;

starredExpression
    : '*' expression;

kwargOrStarred
    : NAME '=' expression
    | starredExpression;

kwargOrDoubleStarred
    : NAME '=' expression
    | '**' expression;

// ASSIGNMENT TARGETS
// ==================

// General targets
// ---------------

starTargets
    : starTarget (',' starTarget)* ','?;

starTarget
    : '*'? targetWithStarAtom;

targetWithStarAtom
    : primary '.' NAME
    | primary '[' slices ']'
    | starAtom;

starAtom
    : NAME
    | '(' starTarget ')'
    | '(' starTargets? ')'
    | '[' starTargets? ']';

singleTarget
    : singleSubscriptAttributeTarget
    | NAME
    | '(' singleTarget ')';

singleSubscriptAttributeTarget
    : primary '.' NAME
    | primary '[' slices ']';

// Targets for del statements
// --------------------------

delTargets
    : delTarget (',' delTarget)* ','?;

delTarget
    : primary '.' NAME
    | primary '[' slices ']'
    | delTargetAtom;

delTargetAtom
    : NAME
    | '(' delTarget ')'
    | '(' delTargets? ')'
    | '[' delTargets? ']';
