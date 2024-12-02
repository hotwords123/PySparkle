#!/bin/bash

cd grammar

export ANTLR4_TOOLS_ANTLR_VERSION=4.13.2

antlr4 -Dlanguage=Python3 PythonLexer.g4
antlr4 -Dlanguage=Python3 PythonParser.g4 -visitor -no-listener
