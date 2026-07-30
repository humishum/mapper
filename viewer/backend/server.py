"""ASGI entry point for the canonical package catalog API."""

from __future__ import annotations

import uvicorn

from .api import create_app
from .config import HOST, PORT, RELOAD

app = create_app()


def main() -> None:
    uvicorn.run(
        "viewer.backend.server:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
    )


if __name__ == "__main__":
    main()
