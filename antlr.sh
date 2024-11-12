#!/bin/bash

cd grammar

antlr4 -Dlanguage=Python3 PythonLexer.g4
antlr4 -Dlanguage=Python3 PythonParser.g4 -visitor -no-listener
