"""SQLite catalog for validated reconstruction packages."""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ..domain.package import AlignmentStatus
from .package_validator import PackageValidator, ValidatedPackage

CATALOG_SCHEMA_VERSION = 1


class CatalogConflict(ValueError):
    pass


class CatalogNotFound(LookupError):
    pass


def new_opaque_id(prefix: str) -> str:
    """Create an opaque identifier; callers must not parse meaning from it."""
    return f"{prefix}_{uuid.uuid4().hex}"


class Catalog:
    def __init__(
        self,
        database_path: str | Path,
        *,
        validator: PackageValidator | None = None,
    ):
        self.database_path = Path(database_path)
        self.validator = validator or PackageValidator()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;

                CREATE TABLE IF NOT EXISTS catalog_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS captures (
                    row_id INTEGER PRIMARY KEY,
                    capture_id TEXT NOT NULL UNIQUE,
                    started_at TEXT,
                    ended_at TEXT,
                    device TEXT,
                    lens TEXT,
                    video_name TEXT,
                    source_uri TEXT,
                    frame_count INTEGER,
                    fps REAL,
                    footprint_geojson TEXT
                );

                CREATE TABLE IF NOT EXISTS runs (
                    row_id INTEGER PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    capture_id TEXT NOT NULL REFERENCES captures(capture_id),
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT,
                    git_commit TEXT,
                    git_status TEXT NOT NULL,
                    package_root TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    manifest_sha256 TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    row_id INTEGER PRIMARY KEY,
                    artifact_id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    kind TEXT NOT NULL,
                    alignment_status TEXT NOT NULL,
                    frame_name TEXT NOT NULL,
                    units TEXT NOT NULL,
                    point_count INTEGER,
                    footprint_geojson TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS artifact_footprints USING rtree(
                    artifact_row_id,
                    min_longitude, max_longitude,
                    min_latitude, max_latitude
                );

                CREATE TABLE IF NOT EXISTS representations (
                    row_id INTEGER PRIMARY KEY,
                    representation_id TEXT NOT NULL UNIQUE,
                    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
                    kind TEXT NOT NULL,
                    format TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    UNIQUE(artifact_id, relative_path)
                );

                CREATE TABLE IF NOT EXISTS sources (
                    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
                    source_index INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    capture_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY(artifact_id, source_index)
                );

                CREATE TABLE IF NOT EXISTS layer_defaults (
                    artifact_id TEXT PRIMARY KEY REFERENCES artifacts(artifact_id),
                    definition_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_runs_capture ON runs(capture_id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_status
                    ON artifacts(alignment_status);
                CREATE INDEX IF NOT EXISTS idx_representations_artifact
                    ON representations(artifact_id);
                """
            )
            connection.execute(
                """
                INSERT INTO catalog_metadata(key, value) VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(CATALOG_SCHEMA_VERSION),),
            )
            connection.commit()

    def register_package(self, package_root: str | Path) -> dict[str, Any]:
        """Validate then register a package atomically and idempotently."""
        package = self.validator.validate(package_root)
        manifest = package.manifest
        canonical_root = str(package.root)
        canonical_manifest = str(package.manifest_path)

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._guard_identity(
                    connection,
                    "runs",
                    "run_id",
                    manifest.run_id,
                    canonical_root,
                )
                existing_run_artifact = connection.execute(
                    "SELECT artifact_id FROM artifacts WHERE run_id = ?",
                    (manifest.run_id,),
                ).fetchone()
                if (
                    existing_run_artifact is not None
                    and existing_run_artifact["artifact_id"] != manifest.artifact_id
                ):
                    raise CatalogConflict(
                        f"run_id {manifest.run_id!r} is already bound to artifact_id "
                        f"{existing_run_artifact['artifact_id']!r}"
                    )
                existing_artifact = connection.execute(
                    """
                    SELECT a.artifact_id, a.run_id, r.package_root
                    FROM artifacts a JOIN runs r ON r.run_id = a.run_id
                    WHERE a.artifact_id = ?
                    """,
                    (manifest.artifact_id,),
                ).fetchone()
                if (
                    existing_artifact is not None
                    and existing_artifact["package_root"] != canonical_root
                ):
                    raise CatalogConflict(
                        f"artifact_id {manifest.artifact_id!r} is already registered "
                        "from another package"
                    )
                if (
                    existing_artifact is not None
                    and existing_artifact["run_id"] != manifest.run_id
                ):
                    raise CatalogConflict(
                        f"artifact_id {manifest.artifact_id!r} is already bound to run_id "
                        f"{existing_artifact['run_id']!r}"
                    )
                representation_ids = [
                    artifact.representation_id for artifact in manifest.artifacts
                ]
                if representation_ids:
                    placeholders = ",".join("?" for _ in representation_ids)
                    conflicting_representation = connection.execute(
                        f"""
                        SELECT representation_id, artifact_id
                        FROM representations
                        WHERE representation_id IN ({placeholders})
                          AND artifact_id != ?
                        LIMIT 1
                        """,
                        (*representation_ids, manifest.artifact_id),
                    ).fetchone()
                    if conflicting_representation is not None:
                        raise CatalogConflict(
                            "representation_id "
                            f"{conflicting_representation['representation_id']!r} "
                            "is already bound to another artifact"
                        )

                self._upsert_capture(connection, package)
                self._upsert_run(
                    connection, package, canonical_root, canonical_manifest
                )
                artifact_row_id = self._upsert_artifact(connection, package)
                self._replace_representations(connection, package)
                self._replace_sources(connection, package)
                self._replace_layer_default(connection, package)
                self._replace_footprint(connection, package, artifact_row_id)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get_artifact(manifest.artifact_id)

    @staticmethod
    def _guard_identity(
        connection: sqlite3.Connection,
        table: str,
        id_column: str,
        identifier: str,
        package_root: str,
    ) -> None:
        row = connection.execute(
            f"SELECT {id_column}, package_root FROM {table} WHERE {id_column} = ?",
            (identifier,),
        ).fetchone()
        if row is not None and row["package_root"] != package_root:
            raise CatalogConflict(
                f"{id_column} {identifier!r} is already registered from another package"
            )

    @staticmethod
    def _upsert_capture(
        connection: sqlite3.Connection, package: ValidatedPackage
    ) -> None:
        manifest = package.manifest
        capture = manifest.capture
        footprint = (
            json.dumps(manifest.footprint_wgs84.as_geojson())
            if manifest.footprint_wgs84
            else None
        )
        connection.execute(
            """
            INSERT INTO captures(
                capture_id, started_at, ended_at, device, lens, video_name,
                source_uri, frame_count, fps, footprint_geojson
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(capture_id) DO UPDATE SET
                started_at=COALESCE(excluded.started_at, captures.started_at),
                ended_at=COALESCE(excluded.ended_at, captures.ended_at),
                device=COALESCE(excluded.device, captures.device),
                lens=COALESCE(excluded.lens, captures.lens),
                video_name=COALESCE(excluded.video_name, captures.video_name),
                source_uri=COALESCE(excluded.source_uri, captures.source_uri),
                frame_count=COALESCE(excluded.frame_count, captures.frame_count),
                fps=COALESCE(excluded.fps, captures.fps),
                footprint_geojson=COALESCE(
                    excluded.footprint_geojson, captures.footprint_geojson
                )
            """,
            (
                manifest.capture_id,
                capture.started_at.isoformat() if capture.started_at else None,
                capture.ended_at.isoformat() if capture.ended_at else None,
                capture.device,
                capture.lens,
                capture.video_name,
                capture.source_uri,
                capture.frame_count,
                capture.fps,
                footprint,
            ),
        )

    @staticmethod
    def _upsert_run(
        connection: sqlite3.Connection,
        package: ValidatedPackage,
        package_root: str,
        manifest_path: str,
    ) -> None:
        manifest = package.manifest
        producer = manifest.producer
        connection.execute(
            """
            INSERT INTO runs(
                run_id, capture_id, status, created_at, model_name, model_version,
                git_commit, git_status, package_root, manifest_path, manifest_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                capture_id=excluded.capture_id,
                status=excluded.status,
                created_at=excluded.created_at,
                model_name=excluded.model_name,
                model_version=excluded.model_version,
                git_commit=excluded.git_commit,
                git_status=excluded.git_status,
                manifest_path=excluded.manifest_path,
                manifest_sha256=excluded.manifest_sha256
            """,
            (
                manifest.run_id,
                manifest.capture_id,
                manifest.alignment.status.value,
                manifest.created_at.isoformat(),
                producer.model_name,
                producer.model_version,
                producer.git_commit,
                producer.git_status,
                package_root,
                manifest_path,
                package.manifest_sha256,
            ),
        )

    @staticmethod
    def _upsert_artifact(
        connection: sqlite3.Connection, package: ValidatedPackage
    ) -> int:
        manifest = package.manifest
        primary = next(
            (
                artifact
                for artifact in manifest.artifacts
                if artifact.kind in {"points", "mesh", "splats"}
            ),
            manifest.artifacts[0],
        )
        point_count = sum(
            artifact.point_count or 0
            for artifact in manifest.artifacts
            if artifact.kind == "points"
        )
        footprint = (
            json.dumps(manifest.footprint_wgs84.as_geojson())
            if manifest.footprint_wgs84
            else None
        )
        connection.execute(
            """
            INSERT INTO artifacts(
                artifact_id, run_id, kind, alignment_status, frame_name, units,
                point_count, footprint_geojson, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(artifact_id) DO UPDATE SET
                run_id=excluded.run_id,
                kind=excluded.kind,
                alignment_status=excluded.alignment_status,
                frame_name=excluded.frame_name,
                units=excluded.units,
                point_count=excluded.point_count,
                footprint_geojson=excluded.footprint_geojson,
                created_at=excluded.created_at
            """,
            (
                manifest.artifact_id,
                manifest.run_id,
                primary.kind,
                manifest.alignment.status.value,
                manifest.coordinate_frame.name,
                manifest.coordinate_frame.units,
                point_count or None,
                footprint,
                manifest.created_at.isoformat(),
            ),
        )
        row = connection.execute(
            "SELECT row_id FROM artifacts WHERE artifact_id = ?",
            (manifest.artifact_id,),
        ).fetchone()
        return int(row["row_id"])

    @staticmethod
    def _replace_representations(
        connection: sqlite3.Connection, package: ValidatedPackage
    ) -> None:
        artifact_id = package.manifest.artifact_id
        connection.execute(
            "DELETE FROM representations WHERE artifact_id = ?", (artifact_id,)
        )
        connection.executemany(
            """
            INSERT INTO representations(
                representation_id, artifact_id, kind, format, relative_path,
                media_type, byte_size, sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    artifact.representation_id,
                    artifact_id,
                    artifact.kind,
                    artifact.format,
                    artifact.path,
                    artifact.media_type,
                    artifact.byte_size,
                    artifact.sha256,
                )
                for artifact in package.manifest.artifacts
            ],
        )

    @staticmethod
    def _replace_sources(
        connection: sqlite3.Connection, package: ValidatedPackage
    ) -> None:
        artifact_id = package.manifest.artifact_id
        connection.execute("DELETE FROM sources WHERE artifact_id = ?", (artifact_id,))
        if package.sources is None:
            return
        connection.executemany(
            """
            INSERT INTO sources(
                artifact_id, source_index, kind, capture_id, run_id, record_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    artifact_id,
                    source.source_index,
                    source.kind,
                    source.capture_id,
                    source.run_id,
                    source.model_dump_json(),
                )
                for source in package.sources.sources
            ],
        )

    @staticmethod
    def _replace_layer_default(
        connection: sqlite3.Connection, package: ValidatedPackage
    ) -> None:
        artifact_id = package.manifest.artifact_id
        connection.execute(
            "DELETE FROM layer_defaults WHERE artifact_id = ?", (artifact_id,)
        )
        if package.manifest.layer_default is not None:
            connection.execute(
                "INSERT INTO layer_defaults(artifact_id, definition_json) VALUES (?, ?)",
                (
                    artifact_id,
                    package.manifest.layer_default.model_dump_json(),
                ),
            )

    @staticmethod
    def _replace_footprint(
        connection: sqlite3.Connection,
        package: ValidatedPackage,
        artifact_row_id: int,
    ) -> None:
        connection.execute(
            "DELETE FROM artifact_footprints WHERE artifact_row_id = ?",
            (artifact_row_id,),
        )
        manifest = package.manifest
        if (
            manifest.alignment.status == AlignmentStatus.UNALIGNED
            or manifest.footprint_wgs84 is None
        ):
            return
        min_lon, min_lat, max_lon, max_lat = manifest.footprint_wgs84.bounds
        connection.execute(
            """
            INSERT INTO artifact_footprints(
                artifact_row_id, min_longitude, max_longitude,
                min_latitude, max_latitude
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (artifact_row_id, min_lon, max_lon, min_lat, max_lat),
        )

    def get_capture(self, capture_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM captures WHERE capture_id = ?", (capture_id,)
            ).fetchone()
        if row is None:
            raise CatalogNotFound(f"capture {capture_id!r} was not found")
        return self._capture_dict(row)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise CatalogNotFound(f"run {run_id!r} was not found")
        return dict(row)

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT a.*, r.capture_id, r.package_root, r.manifest_path,
                       r.manifest_sha256, l.definition_json
                FROM artifacts a
                JOIN runs r ON r.run_id = a.run_id
                LEFT JOIN layer_defaults l ON l.artifact_id = a.artifact_id
                WHERE a.artifact_id = ?
                """,
                (artifact_id,),
            ).fetchone()
            if row is None:
                raise CatalogNotFound(f"artifact {artifact_id!r} was not found")
            representations = connection.execute(
                """
                SELECT representation_id, kind, format, relative_path, media_type,
                       byte_size, sha256
                FROM representations WHERE artifact_id = ?
                ORDER BY representation_id
                """,
                (artifact_id,),
            ).fetchall()
        value = dict(row)
        footprint = value.pop("footprint_geojson")
        definition = value.pop("definition_json")
        value["footprint"] = json.loads(footprint) if footprint else None
        value["layer_default"] = json.loads(definition) if definition else None
        value["representations"] = [dict(item) for item in representations]
        return value

    def query_artifacts(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        started_after: datetime | None = None,
        ended_before: datetime | None = None,
        alignment_status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        clauses: list[str] = []
        parameters: list[Any] = []
        footprint_join = ""
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            if not all(math.isfinite(value) for value in bbox):
                raise ValueError("bbox values must be finite")
            if not all(
                -180 <= longitude <= 180 for longitude in (min_lon, max_lon)
            ):
                raise ValueError("bbox longitudes must be between -180 and 180")
            if not all(-90 <= latitude <= 90 for latitude in (min_lat, max_lat)):
                raise ValueError("bbox latitudes must be between -90 and 90")
            if min_lat > max_lat:
                raise ValueError(
                    "bbox minimum latitude must not exceed maximum latitude"
                )
            footprint_join = (
                "JOIN artifact_footprints f ON f.artifact_row_id = a.row_id"
            )
            if min_lon <= max_lon:
                clauses.extend(
                    [
                        "f.max_longitude >= ?",
                        "f.min_longitude <= ?",
                    ]
                )
                parameters.extend([min_lon, max_lon])
            else:
                # A west bound east of the east bound denotes the union
                # [min_lon, 180] U [-180, max_lon].
                clauses.append(
                    "(f.max_longitude >= ? OR f.min_longitude <= ?)"
                )
                parameters.extend([min_lon, max_lon])
            clauses.extend(
                [
                    "f.max_latitude >= ?",
                    "f.min_latitude <= ?",
                ]
            )
            parameters.extend([min_lat, max_lat])
        if started_after is not None:
            clauses.append("c.ended_at >= ?")
            parameters.append(started_after.isoformat())
        if ended_before is not None:
            clauses.append("c.started_at <= ?")
            parameters.append(ended_before.isoformat())
        if alignment_status is not None:
            try:
                alignment_status = AlignmentStatus(alignment_status).value
            except ValueError as exc:
                allowed = ", ".join(status.value for status in AlignmentStatus)
                raise ValueError(
                    f"alignment_status must be one of: {allowed}"
                ) from exc
            clauses.append("a.alignment_status = ?")
            parameters.append(alignment_status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT a.artifact_id, a.run_id, r.capture_id, a.kind,
                       a.alignment_status, a.frame_name, a.units, a.point_count,
                       a.footprint_geojson, a.created_at
                FROM artifacts a
                {footprint_join}
                JOIN runs r ON r.run_id = a.run_id
                JOIN captures c ON c.capture_id = r.capture_id
                {where}
                ORDER BY a.created_at DESC, a.artifact_id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        results = []
        for row in rows:
            value = dict(row)
            footprint = value.pop("footprint_geojson")
            value["footprint"] = json.loads(footprint) if footprint else None
            results.append(value)
        return results

    def get_representation(self, representation_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT p.*, r.package_root
                FROM representations p
                JOIN artifacts a ON a.artifact_id = p.artifact_id
                JOIN runs r ON r.run_id = a.run_id
                WHERE p.representation_id = ?
                """,
                (representation_id,),
            ).fetchone()
        if row is None:
            raise CatalogNotFound(f"representation {representation_id!r} was not found")
        return dict(row)

    def get_sources(self, artifact_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
            if exists is None:
                raise CatalogNotFound(f"artifact {artifact_id!r} was not found")
            rows = connection.execute(
                """
                SELECT record_json FROM sources
                WHERE artifact_id = ? ORDER BY source_index
                """,
                (artifact_id,),
            ).fetchall()
        return [json.loads(row["record_json"]) for row in rows]

    @staticmethod
    def _capture_dict(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        footprint = value.pop("footprint_geojson")
        value["footprint"] = json.loads(footprint) if footprint else None
        return value
