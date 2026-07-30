#!/usr/bin/env python3
"""Regenerate the checked-in reconstruction package JSON schemas."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viewer.backend.domain.package import Manifest, Metrics, SourcesDocument  # noqa: E402

SCHEMAS = {
    "manifest.v1.json": Manifest,
    "metrics.v1.json": Metrics,
    "sources.v1.json": SourcesDocument,
}


def main() -> None:
    schema_dir = ROOT / "schemas"
    schema_dir.mkdir(exist_ok=True)
    for filename, model in SCHEMAS.items():
        schema = model.model_json_schema(
            mode="validation",
            ref_template=f"{filename}#/$defs/{{model}}",
        )
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = f"https://mapper.local/schemas/{filename}"
        (schema_dir / filename).write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
