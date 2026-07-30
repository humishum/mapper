"""Environment settings for the canonical catalog server."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("MAPPER_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
RELOAD = os.getenv("MAPPER_RELOAD", "true").lower() in {"1", "true", "yes", "on"}


def catalog_path_from_env() -> Path:
    """Resolve the catalog lazily so tests and launchers can set the environment."""
    return Path(os.environ.get("MAPPER_CATALOG_PATH", "mapper-catalog.sqlite3")).expanduser()
