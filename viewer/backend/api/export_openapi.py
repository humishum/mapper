"""Deterministically export the canonical v1 OpenAPI document."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .app import create_app


def export_openapi(output_path: str | Path) -> Path:
    """Write a stable OpenAPI document without depending on a real catalog."""
    output = Path(output_path)
    with tempfile.TemporaryDirectory(prefix="mapper-openapi-") as directory:
        app = create_app(Path(directory) / "catalog.sqlite3")
        document = app.openapi()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("schemas/openapi.v1.json"),
        help="output path (default: schemas/openapi.v1.json)",
    )
    arguments = parser.parse_args()
    export_openapi(arguments.output)


if __name__ == "__main__":
    main()
