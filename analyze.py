import functools
import logging
import sys
from pathlib import Path
from typing import Optional

import dominate
import dominate.tags as dom
from antlr4 import *
from antlr4.Token import CommonToken
from dominate.util import text as dom_text
from typeshed_client import get_search_context

from core.analysis import PythonAnalyzer
from core.source import PythonSource
from semantics.entity import PyModule
from semantics.token import TokenKind, is_synthetic_token


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s|%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    source_path = Path(args.input)

    assert "." not in source_path.stem, "source file name should not contain '.'"

    search_context = get_search_context()

    analyzer = PythonAnalyzer(search_paths=[search_context.typeshed])

    if args.typeshed:
        analyzer.load_typeshed()
    else:
        analyzer.typeshed_loaded = True

    # https://typing.readthedocs.io/en/latest/spec/distributing.html#import-resolution-ordering
    search_paths = [
        source_path.parent,
        search_context.typeshed,
        *search_context.search_path,
    ]
    analyzer.importer.search_paths = search_paths

    module = PyModule(source_path.stem, source_path)
    module.loader = functools.partial(PythonSource.parse_file, source_path)
    analyzer.load_module(module)

    with analyzer.set_type_context():
        doc = generate_html(module)

    if args.output:
        with open(args.output, "w", newline="") as f:
            print(doc, file=f)
    else:
        print(doc, file=sys.stdout)


def generate_html(module: PyModule) -> dominate.document:
    token_source = (module.source.lexer, module.source.input_stream)

    doc = dominate.document(title="Python Code")

    with doc.head:
        dom.link(rel="stylesheet", href="style.css")
        dom.script(src="script.js")

    with doc:
        with dom.div(cls="highlight"):
            with dom.pre():
                for token in module.source.stream.tokens:
                    token: CommonToken
                    if is_synthetic_token(token):
                        continue

                    token_kind = module.context.get_token_kind(token)
                    if token_kind is TokenKind.NONE:
                        dom_text(token.text)
                    else:
                        attrs = dict(
                            id=f"token-{token.tokenIndex}",
                            cls=f"token token-{token_kind.value}",
                        )

                        if entity_type := module.context.get_token_entity_type(token):
                            attrs["title"] = str(entity_type)

                        if (
                            (target := module.context.get_token_target(token))
                            and (target_token := target.token)
                            and target_token.source == token_source
                        ):
                            attrs["data-target"] = f"token-{target_token.tokenIndex}"

                        dom.span(token.text, **attrs)

    return doc


def parse_args(args: Optional[list[str]] = None):
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file")
    parser.add_argument("-o", "--output", help="output file")
    parser.add_argument("--no-typeshed", dest="typeshed", action="store_false", help="do not load typeshed")
    # fmt: on

    return parser.parse_args(args)


if __name__ == "__main__":
    main(parse_args())
