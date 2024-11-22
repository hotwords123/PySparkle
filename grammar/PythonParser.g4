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
    : compound_stmt
    | simple_stmts
    | invalid_block
    | invalid_token NEWLINE;

simple_stmts: simple_stmt (';' simple_stmt)* (';')? NEWLINE;

simple_stmt
    : assignment
    // | type_alias
    | star_expressions
    | return_stmt
    | import_stmt
    | raise_stmt
    | 'pass'
    | del_stmt
    | yield_stmt
    | assert_stmt
    | 'break'
    | 'continue'
    | global_stmt
    | nonlocal_stmt
    ;

compound_stmt
    : function_def
    | if_stmt
    | class_def
    | with_stmt
    | for_stmt
    | try_stmt
    | while_stmt
    // | match_stmt
    ;

invalid_block: INDENT statements DEDENT;

invalid_token
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
    : (NAME | '(' single_target ')' | single_subscript_attribute_target)
        ':' expression ('=' annotated_rhs)?
    | (star_targets '=')+ (yield_expr | star_expressions)
    | single_target augassign (yield_expr | star_expressions);

annotated_rhs: yield_expr | star_expressions;

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

return_stmt: 'return' star_expressions?;

raise_stmt: 'raise' (expression ('from' expression)?)?;

global_stmt: 'global' NAME (',' NAME)*;

nonlocal_stmt: 'nonlocal' NAME (',' NAME)*;

del_stmt: 'del' del_targets;

yield_stmt: yield_expr;

assert_stmt: 'assert' expression (',' expression)?;

import_stmt: import_name | import_from;

// Import statements
// -----------------

import_name: 'import' dotted_as_names;
import_from
    : 'from' ('.' | '...')* dotted_name 'import' import_from_targets
    | 'from' ('.' | '...')+ 'import' import_from_targets;
import_from_targets
    : '(' import_from_as_names ','? ')'
    | import_from_as_names
    | '*';
import_from_as_names
    : import_from_as_name (',' import_from_as_name)*;
import_from_as_name
    : NAME ('as' NAME)?;
dotted_as_names
    : dotted_as_name (',' dotted_as_name)*;
dotted_as_name:
    | dotted_name ('as' NAME)?;
dotted_name
    : dotted_name '.' NAME
    | NAME;

// COMPOUND STATEMENTS
// ===================

// Common elements
// ---------------

block
    : NEWLINE INDENT statements DEDENT
    | simple_stmts;

decorators: ('@' named_expression NEWLINE)+;

// Class definitions
// -----------------

class_def
    : decorators? 'class' NAME type_params? ('(' arguments? ')')?
        ':' block;

// Function definitions
// --------------------

function_def
    : decorators? ASYNC? 'def' NAME type_params? '(' parameters? ')'
        ('->' expression)? ':' block;

// Function parameters
// -------------------

parameters
    : slash_no_default (',' param_no_default)* (',' param_with_default)*
        (',' star_etc?)?
    | slash_with_default (',' param_with_default)*
        (',' star_etc?)?
    | param_no_default (',' param_no_default)* (',' param_with_default)*
        (',' star_etc?)?
    | param_with_default (',' param_with_default)*
        (',' star_etc?)?
    | star_etc;

slash_no_default
    : param_no_default (',' param_no_default)* ',' '/';
slash_with_default
    : param_no_default (',' param_no_default)* (',' param_with_default)+ ',' '/'
    | param_with_default (',' param_with_default)* ',' '/';

star_etc
    : '*' (param_no_default | param_no_default_star_annotation)
        (',' param_maybe_default)* (',' kwds?)?
    | '*' (',' param_maybe_default)+ (',' kwds?)?
    | kwds;

kwds: '**' param_no_default ','?;

param_no_default: param;
param_no_default_star_annotation: param_star_annotation;
param_with_default: param default;
param_maybe_default: param default?;

param: NAME annotation?;
param_star_annotation: NAME star_annotation;
annotation: ':' expression;
star_annotation: ':' star_expression;
default: '=' expression;

// If statement
// ------------

if_stmt
    : 'if' named_expression ':' block (elif_stmt | else_block)?;
elif_stmt
    : 'elif' named_expression ':' block (elif_stmt | else_block)?;
else_block
    : 'else' ':' block;

// While statement
// ---------------

while_stmt
    : 'while' named_expression ':' block else_block?;

// For statement
// -------------

for_stmt
    : ASYNC? 'for' star_targets 'in' star_expressions ':' block else_block?;

// With statement
// --------------

with_stmt
    : ASYNC? 'with' '(' with_item (',' with_item)* ','? ')' ':' block
    | ASYNC? 'with' with_item (',' with_item)* ':' block;

with_item: expression ('as' star_target)?;

// Try statement
// -------------

try_stmt
    : 'try' ':' block (finally_block
        | except_block+ else_block? finally_block?
        | except_star_block+ else_block? finally_block?);

// Except statement
// ----------------

except_block
    : 'except' (expression ('as' NAME)?)? ':' block;
except_star_block
    : 'except' '*' expression ('as' NAME)? ':' block;
finally_block
    : 'finally' ':' block;

// Match statement
// ---------------

// TODO

// Type statement
// --------------

// type_alias
//     : 'type' NAME '=' expression;

// Type parameter declaration
// --------------------------

type_params
    : '[' type_param (',' type_param)* ','? ']';

type_param
    : NAME type_param_bound? type_param_default?
    | '*' NAME type_param_starred_default?
    | '**' NAME type_param_default?;

type_param_bound: ':' expression;
type_param_default: '=' expression;
type_param_starred_default: '=' star_expression;

// EXPRESSIONS
// ===========

expressions
    : expression (',' expression)* ','?;

expression
    : logical 'if' logical 'else' expression
    | logical
    // | lambdef
    ;

yield_expr
    : 'yield' ('from' expression | star_expressions)?;

star_expressions
    : star_expression (',' star_expression)* ','?;

star_expression
    : '*' bitwise
    | expression;

star_named_expressions
    : star_named_expression (',' star_named_expression)* ','?;

star_named_expression
    : '*' bitwise
    | named_expression;

assignment_expression
    : NAME ':=' expression;

named_expression
    : assignment_expression
    | expression;

logical
    : 'not' logical
    | logical 'and' logical
    | logical 'or' logical
    | comparison;

// Comparison operators
// --------------------

comparison
    : bitwise compare_op_bitwise_pair*;

compare_op_bitwise_pair
    : '==' bitwise       # compare_op_eq
    | '!=' bitwise       # compare_op_noteq
    | '<=' bitwise       # compare_op_lte
    | '<' bitwise        # compare_op_lt
    | '>=' bitwise       # compare_op_gte
    | '>' bitwise        # compare_op_gt
    | 'in' bitwise       # compare_op_in
    | 'not' 'in' bitwise # compare_op_notin
    | 'is' bitwise       # compare_op_is
    | 'is' 'not' bitwise # compare_op_isnot
    ;

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
    | await_primary;

// Primary elements
// ----------------

await_primary
    : AWAIT primary
    | primary;

primary
    : primary '.' NAME
    | primary genexp
    | primary '(' arguments? ')'
    | primary '[' slices ']'
    | atom;

slices
    : slice
    | (slice | starred_expression) (',' (slice | starred_expression))* ','?;

slice
    : expression? ':' expression? (':' expression)?
    | named_expression;

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
    : '(' (yield_expr | named_expression) ')';

// Lambda functions
// ----------------

// TODO

// LITERALS
// ========

// TODO: f-strings

string: STRING_LITERAL | BYTES_LITERAL;
strings: string+;

number: INTEGER | FLOAT_NUMBER | IMAG_NUMBER;

list
    : '[' star_named_expressions? ']';

tuple
    : '(' (star_named_expression ',' star_named_expressions?)? ')';

set
    : '{' star_named_expressions '}';

// Dicts
// -----

dict
    : '{' double_starred_kvpairs? '}';

double_starred_kvpairs
    : double_starred_kvpair (',' double_starred_kvpair)* ','?;

double_starred_kvpair
    : '**' bitwise
    | kvpair;

kvpair: expression ':' expression;

// Comprenhensions & Generators
// ----------------------------

for_if_clauses: for_if_clause+;

for_if_clause
    : ASYNC? 'for' star_targets 'in' logical ('if' logical)*;

listcomp
    : '[' named_expression for_if_clauses ']';

setcomp
    : '{' named_expression for_if_clauses '}';

genexp
    : '(' named_expression for_if_clauses ')';

dictcomp
    : '{' kvpair for_if_clauses '}';

// FUNCTION CALL ARGUMENTS
// =======================

arguments
    : args ','?;

args
    : arg (',' arg)* (',' kwargs)?
    | kwargs;

arg
    : starred_expression
    | assignment_expression
    | expression;

kwargs
    : kwarg_or_starred (',' kwarg_or_starred)* (',' kwarg_or_double_starred)?
    | kwarg_or_double_starred (',' kwarg_or_double_starred)*;

starred_expression
    : '*' expression;

kwarg_or_starred
    : NAME '=' expression
    | starred_expression;

kwarg_or_double_starred
    : NAME '=' expression
    | '**' expression;

// ASSIGNMENT TARGETS
// ==================

// General targets
// ---------------

star_targets
    : star_target (',' star_target)* ','?;

star_target
    : STAR? target_with_star_atom;

target_with_star_atom
    : primary '.' NAME
    | primary '[' slices ']'
    | star_atom;

star_atom
    : NAME
    | '(' star_target ')'
    | '(' star_targets? ')'
    | '[' star_targets? ']';

single_target
    : single_subscript_attribute_target
    | NAME
    | '(' single_target ')';

single_subscript_attribute_target
    : primary '.' NAME
    | primary '[' slices ']';

// Targets for del statements
// --------------------------

del_targets
    : del_target (',' del_target)* ','?;

del_target
    : primary '.' NAME
    | primary '[' slices ']'
    | del_t_atom;

del_t_atom
    : NAME
    | '(' del_target ')'
    | '(' del_targets? ')'
    | '[' del_targets? ']';
