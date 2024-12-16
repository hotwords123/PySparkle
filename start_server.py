import argparse
import logging
from typing import Optional

from typeshed_client import get_search_context

from lsp.server import server


def main(args):
    logging.basicConfig(
        level=args.log_level,
        format="[%(name)s|%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("pygls.protocol.json_rpc").setLevel(logging.WARNING)

    server.search_context = get_search_context(
        search_path=args.python_path,
        python_executable=args.python_executable,
    )

    if args.tcp:
        server.start_tcp(args.host, args.port)
    elif args.ws:
        server.start_ws(args.host, args.port)
    else:
        server.start_io()


def parse_args(args: Optional[list[str]] = None):
    name = type(server).__name__
    parser = argparse.ArgumentParser(description=f"start a {name} instance")
    # fmt: off
    parser.add_argument("--tcp", action="store_true", help="start a TCP server")
    parser.add_argument("--ws", action="store_true", help="start a WebSocket server")
    parser.add_argument("--host", default="127.0.0.1", help="bind to this address")
    parser.add_argument("--port", type=int, default=8888, help="bind to this port")
    parser.add_argument("--log-level", default="INFO", help="set the logging level")
    parser.add_argument("--python-executable", help="path to the Python executable")
    parser.add_argument("--python-path", nargs="+", help="additional module search paths")
    # fmt: on

    return parser.parse_args(args)


if __name__ == "__main__":
    main(parse_args())
