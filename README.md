# PySparkle

## Getting Started

### Install dependencies

```sh
conda create -n pysparkle python=3.12
conda activate pysparkle
pip install -r requirements.txt
```

### Generate ANTLR4 lexer and parser

```sh
bash antlr.sh
```

### Run

```sh
python driver.py input.py
```

### Test

```sh
bash test.sh
```
