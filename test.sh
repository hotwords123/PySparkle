#!/bin/bash -e

for file in code/*.py; do
    echo "Testing $file"
    python -m driver lex $file -o ${file%.py}.lex
    python -m driver parse $file -o ${file%.py}.yaml
done
