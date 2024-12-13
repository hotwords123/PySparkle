import logging

from pygls.cli import start_server

from lsp.server import server

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s|%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    start_server(server)
