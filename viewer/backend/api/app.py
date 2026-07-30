"""Application factory for the canonical v1 backend."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..adapters.blob_store import FileBlobStore
from ..config import catalog_path_from_env
from ..services.catalog import Catalog
from .models import HealthResponse
from .routes import router


def create_app(catalog_path: str | Path | None = None) -> FastAPI:
    path = Path(
        catalog_path if catalog_path is not None else catalog_path_from_env()
    )
    catalog = Catalog(path)
    app = FastAPI(
        title="Mapper Catalog API",
        description="Canonical reconstruction catalog and immutable artifact server",
        version="1.0.0",
    )
    app.state.catalog = catalog
    app.state.blobs = FileBlobStore(catalog)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["Range", "If-Range", "If-None-Match", "Content-Type"],
        expose_headers=[
            "Accept-Ranges",
            "Cache-Control",
            "Content-Range",
            "Content-Length",
            "ETag",
        ],
    )
    app.include_router(router)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return {"status": "healthy", "catalog": str(path)}

    return app
