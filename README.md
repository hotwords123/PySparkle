# PySparkle

## Getting Started

### Install dependencies

```sh
conda create -n pysparkle python=3.12
conda activate pysparkle
pip install -r requirements.txt
```

## Lexer and parser

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

## Language support

### Supported Python features

- [x] Common Python syntax
- [x] Module system
- [x] Type annotations
- [x] Basic type inference
- [x] Generic types
- [ ] Control flow analysis

### Supported language server features

- [x] Syntax highlighting
- [x] Semantic highlighting
- [x] Hover
- [x] Go to definition
- [x] Code completion
- [x] Signature help
- [x] Diagnostics

### Analyze code

```sh
python analyze.py input.py [--root /path/to/root] [-o output.html]
```

Then put the `style.css` and `script.js` in the same directory as `output.html` in order to make the highlighting and symbol navigation work.

### Run language server

```sh
python start_server.py [--python-executable /path/to/python] [--python-path /path/to/packages ...] [options...]
```

For detailed options, see `python start_server.py --help`.

NOTE: The `--python-executable` and `--python-path` options are used to specify the Python environment for the code that is being analyzed, NOT the language server itself.

### VSCode extension

The VSCode extension is available at [pysparkle-vscode](https://github.com/hotwords123/pysparkle-vscode). Install it by copying the `pysparkle-vscode` directory to the `.vscode/extensions` directory in your home directory or some workspace directory. You may need to manually activate the extension in the Extensions view.

The following settings are required in order to make the extension work:

- `pysparkle.server.cwd`: Set to the directory that contains the `start_server.py` script.
- `pysparkle.server.pythonPath`: Set to the Python executable that the language server should use (typically in a virtual environment). The Python environment should have the necessary packages installed.
- `pysparkle.server.launchScript`: Set to `start_server.py`.
- `pysparkle.server.launchArgs`: (Optional) Additional arguments to pass to the language server.

You may need to disable the built-in Python extension in order to avoid conflicts.
