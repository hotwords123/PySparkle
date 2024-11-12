lexer grammar PythonLexer;

options {
    superClass=PythonLexerBase;
}

// https://docs.python.org/3/reference/lexical_analysis.html

tokens {
    INDENT, DEDENT // https://docs.python.org/3/reference/lexical_analysis.html#indentation
}

/* Keywords */
// https://docs.python.org/3/reference/lexical_analysis.html#keywords
FALSE    : 'False';
AWAIT    : 'await';
ELSE     : 'else';
IMPORT   : 'import';
PASS     : 'pass';
NONE     : 'None';
BREAK    : 'break';
EXCEPT   : 'except';
IN       : 'in';
RAISE    : 'raise';
TRUE     : 'True';
CLASS    : 'class';
FINALLY  : 'finally';
IS       : 'is';
RETURN   : 'return';
AND      : 'and';
CONTINUE : 'continue';
FOR      : 'for';
LAMBDA   : 'lambda';
TRY      : 'try';
AS       : 'as';
DEF      : 'def';
FROM     : 'from';
NONLOCAL : 'nonlocal';
WHILE    : 'while';
ASSERT   : 'assert';
DEL      : 'del';
GLOBAL   : 'global';
NOT      : 'not';
WITH     : 'with';
ASYNC    : 'async';
ELIF     : 'elif';
IF       : 'if';
OR       : 'or';
YIELD    : 'yield';

/* Operators */
// https://docs.python.org/3/reference/lexical_analysis.html#operators
PLUS         : '+';
MINUS        : '-';
STAR         : '*';
DOUBLESTAR   : '**';
SLASH        : '/';
DOUBLESLASH  : '//';
PERCENT      : '%';
AT           : '@';
LEFTSHIFT    : '<<';
RIGHTSHIFT   : '>>';
AMPERSAND    : '&';
VBAR         : '|';
CIRCUMFLEX   : '^';
TILDE        : '~';
COLONEQUAL   : ':=';
LESS         : '<';
GREATER      : '>';
LESSEQUAL    : '<=';
GREATEREQUAL : '>=';
DOUBLEEQUAL  : '==';
NOTEQUAL     : '!=';

/* Delimiters */
// https://docs.python.org/3/reference/lexical_analysis.html#delimiters
LPAREN           : '(';
RPAREN           : ')';
LSQB             : '[';
RSQB             : ']';
LBRACE           : '{';
RBRACE           : '}';
COMMA            : ',';
COLON            : ':';
EXCLAMATION      : '!';
DOT              : '.';
SEMICOLON        : ';';
EQUAL            : '=';
RARROW           : '->';
PLUSEQUAL        : '+=';
MINUSEQUAL       : '-=';
STAREQUAL        : '*=';
SLASHEQUAL       : '/=';
DOUBLESLASHEQUAL : '//=';
PERCENTEQUAL     : '%=';
ATEQUAL          : '@=';
AMPERSANDEQUAL   : '&=';
VBAREQUAL        : '|=';
CIRCUMFLEXEQUAL  : '^=';
RIGHTSHIFTEQUAL  : '>>=';
LEFTSHIFTEQUAL   : '<<=';
DOUBLESTAREQUAL  : '**=';
ELLIPSIS         : '...';

/* Line structure */
// https://docs.python.org/3/reference/lexical_analysis.html#physical-lines
NEWLINE : '\r'? '\n' | '\r';

// https://docs.python.org/3/reference/lexical_analysis.html#comments
COMMENT : '#' ~[\r\n]* -> channel(HIDDEN);

// https://docs.python.org/3/reference/lexical_analysis.html#explicit-line-joining
EXPLICIT_LINE_JOINING : '\\' NEWLINE -> channel(HIDDEN);

// https://docs.python.org/3/reference/lexical_analysis.html#whitespace-between-tokens
WS : [ \t\f]+ -> channel(HIDDEN);

/* Identifiers */
// https://docs.python.org/3/reference/lexical_analysis.html#identifiers
// Let's not worry about Unicode identifiers for now
NAME : ID_START ID_CONTINUE*;

fragment ID_START : [a-zA-Z_];
fragment ID_CONTINUE : ID_START | [0-9];

/* String and Bytes literals */
// https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals
// Let's not worry about f-strings for now
STRING_LITERAL : STRING_PREFIX? (SHORT_STRING | LONG_STRING);

fragment STRING_PREFIX
    : 'r' | 'u' | 'R' | 'U' | 'f' | 'F'
    | 'fr' | 'Fr' | 'fR' | 'FR' | 'rf' | 'rF' | 'Rf' | 'RF';

fragment SHORT_STRING
    : '\'' SHORT_STRING_ITEM_SINGLE* '\''
    | '"' SHORT_STRING_ITEM_DOUBLE* '"';

fragment LONG_STRING
    : '\'\'\'' LONG_STRING_ITEM*? '\'\'\''
    | '"""' LONG_STRING_ITEM*? '"""';

fragment SHORT_STRING_ITEM_SINGLE
    : ~[\\\r\n']
    | STRING_ESCAPE_SEQ;

fragment SHORT_STRING_ITEM_DOUBLE
    : ~[\\\r\n"]
    | STRING_ESCAPE_SEQ;

fragment LONG_STRING_ITEM
    : ~[\\]
    | STRING_ESCAPE_SEQ;

fragment STRING_ESCAPE_SEQ
    : '\\' .;

BYTES_LITERAL : BYTES_PREFIX? (SHORT_BYTES | LONG_BYTES);

fragment BYTES_PREFIX
    : 'b' | 'B'
    | 'br' | 'Br' | 'bR' | 'BR' | 'rb' | 'rB' | 'Rb' | 'RB';

fragment SHORT_BYTES
    : '\'' SHORT_BYTES_ITEM_SINGLE* '\''
    | '"' SHORT_BYTES_ITEM_DOUBLE* '"';

fragment LONG_BYTES
    : '\'\'\'' LONG_BYTES_ITEM*? '\'\'\''
    | '"""' LONG_BYTES_ITEM*? '"""';

fragment SHORT_BYTES_ITEM_SINGLE
    : '\u0000'..'\u0009'
    | '\u000B'..'\u000C'
    | '\u000E'..'\u0026'
    | '\u0028'..'\u005B'
    | '\u005D'..'\u007F'
    | BYTES_ESCAPE_SEQ;

fragment SHORT_BYTES_ITEM_DOUBLE
    : '\u0000'..'\u0009'
    | '\u000B'..'\u000C'
    | '\u000E'..'\u0021'
    | '\u0023'..'\u005B'
    | '\u005D'..'\u007F'
    | BYTES_ESCAPE_SEQ;

fragment LONG_BYTES_ITEM
    : '\u0000'..'\u005B'
    | '\u005D'..'\u007F'
    | BYTES_ESCAPE_SEQ;

fragment BYTES_ESCAPE_SEQ
    : '\\' '\u0000'..'\u007F';

/* Numeric literals */
// https://docs.python.org/3/reference/lexical_analysis.html#numeric-literals
INTEGER : DEC_INTEGER | BIN_INTEGER | OCT_INTEGER | HEX_INTEGER;

fragment DEC_INTEGER : [1-9] ('_'? [0-9])* | '0'+ ('_'? '0')*;
fragment BIN_INTEGER : '0' [bB] ('_'? [01])+;
fragment OCT_INTEGER : '0' [oO] ('_'? [0-7])+;
fragment HEX_INTEGER : '0' [xX] ('_'? [0-9a-fA-F])+;

FLOAT_NUMBER : POINT_FLOAT | EXPONENT_FLOAT;

fragment POINT_FLOAT : DIGIT_PART? FRACTION | DIGIT_PART '.';
fragment EXPONENT_FLOAT : (DIGIT_PART | POINT_FLOAT) EXPONENT;
fragment DIGIT_PART : [0-9] ('_'? [0-9])*;
fragment FRACTION : '.' DIGIT_PART;
fragment EXPONENT : [eE] [+-]? DIGIT_PART;

IMAG_NUMBER : (FLOAT_NUMBER | INTEGER) [jJ];

ERROR: .;
