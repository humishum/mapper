"""Dependency-free structural validation for COPC files.

This parser is deliberately independent of both ``copc_converter`` and
``laspy``.  It validates the hierarchy bytes a range-reading viewer consumes,
including the easy-to-miss requirement that every node is reachable from the
root.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

LAS_14_HEADER_SIZE = 375
VLR_HEADER_SIZE = 54
COPC_INFO_SIZE = 160
HIERARCHY_ENTRY_SIZE = 32
EVLR_HEADER_SIZE = 60


@dataclass(frozen=True)
class CopcStructure:
    point_count: int
    point_format: int
    point_length: int
    scale: tuple[float, float, float]
    offset: tuple[float, float, float]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    hierarchy_point_count: int
    node_count: int
    hierarchy_pages: int
    max_depth: int
    root_present: bool
    all_nodes_reachable: bool
    unreachable_nodes: int
    invalid_node_keys: int


def _read_exact(handle: BinaryIO, size: int, label: str) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise ValueError(f"truncated {label}: wanted {size} bytes, got {len(value)}")
    return value


def inspect_copc(path: Path) -> CopcStructure:
    """Parse a COPC header and its complete paged hierarchy."""

    path = Path(path)
    with path.open("rb") as handle:
        header = _read_exact(handle, LAS_14_HEADER_SIZE, "LAS 1.4 header")
        if header[:4] != b"LASF":
            raise ValueError("not a LAS/LAZ file (missing LASF signature)")
        if (header[24], header[25]) != (1, 4):
            raise ValueError(f"COPC requires LAS 1.4, got {header[24]}.{header[25]}")

        point_format = header[104] & 0x3F
        (point_length,) = struct.unpack_from("<H", header, 105)
        (legacy_count,) = struct.unpack_from("<I", header, 107)
        (point_count_14,) = struct.unpack_from("<Q", header, 247)
        point_count = point_count_14 or legacy_count
        scale = struct.unpack_from("<3d", header, 131)
        offset = struct.unpack_from("<3d", header, 155)
        max_x, min_x, max_y, min_y, max_z, min_z = struct.unpack_from(
            "<6d", header, 179
        )

        vlr_header = _read_exact(handle, VLR_HEADER_SIZE, "first VLR header")
        user_id = vlr_header[2:18].rstrip(b"\x00").decode("ascii", "replace")
        (record_id,) = struct.unpack_from("<H", vlr_header, 18)
        if (user_id, record_id) != ("copc", 1):
            raise ValueError(
                "first VLR is not the COPC info VLR "
                f"(user_id={user_id!r}, record_id={record_id})"
            )
        info = _read_exact(handle, COPC_INFO_SIZE, "COPC info VLR")
        root_offset, root_size = struct.unpack_from("<2Q", info, 40)

        pages = 0
        point_counts: dict[tuple[int, int, int, int], int] = {}
        pending = [(root_offset, root_size)]
        visited_pages: set[tuple[int, int]] = set()
        while pending:
            page_offset, page_size = pending.pop()
            page_ref = (page_offset, page_size)
            if page_ref in visited_pages:
                raise ValueError(f"COPC hierarchy contains a page cycle at {page_ref}")
            visited_pages.add(page_ref)
            if page_size <= 0 or page_size % HIERARCHY_ENTRY_SIZE:
                raise ValueError(f"invalid COPC hierarchy page size: {page_size}")
            handle.seek(page_offset)
            page = _read_exact(handle, page_size, "COPC hierarchy page")
            pages += 1
            for position in range(0, len(page), HIERARCHY_ENTRY_SIZE):
                level, x, y, z = struct.unpack_from("<4i", page, position)
                child_offset, byte_size, entry_points = struct.unpack_from(
                    "<Qii", page, position + 16
                )
                if entry_points == -1:
                    pending.append((child_offset, byte_size))
                    continue
                key = (level, x, y, z)
                if key in point_counts:
                    raise ValueError(f"duplicate COPC hierarchy node: {key}")
                point_counts[key] = entry_points

    root = (0, 0, 0, 0)
    reachable: set[tuple[int, int, int, int]] = set()
    invalid = 0
    for key in sorted(point_counts):
        level, x, y, z = key
        if level < 0:
            invalid += 1
            continue
        limit = 1 << level
        if not (0 <= x < limit and 0 <= y < limit and 0 <= z < limit):
            invalid += 1
            continue
        if key == root:
            reachable.add(key)
        elif (level - 1, x // 2, y // 2, z // 2) in reachable:
            reachable.add(key)

    return CopcStructure(
        point_count=point_count,
        point_format=point_format,
        point_length=point_length,
        scale=scale,
        offset=offset,
        bounds_min=(min_x, min_y, min_z),
        bounds_max=(max_x, max_y, max_z),
        hierarchy_point_count=sum(point_counts.values()),
        node_count=len(point_counts),
        hierarchy_pages=pages,
        max_depth=max((key[0] for key in point_counts), default=0),
        root_present=root in point_counts,
        all_nodes_reachable=len(reachable) == len(point_counts),
        unreachable_nodes=len(point_counts) - len(reachable),
        invalid_node_keys=invalid,
    )


def validate_copc_structure(path: Path, expected_point_count: int) -> CopcStructure:
    """Raise ``ValueError`` when a COPC is incomplete or untraversable."""

    structure = inspect_copc(path)
    failures = []
    if structure.point_count != expected_point_count:
        failures.append(
            f"header has {structure.point_count} points, expected {expected_point_count}"
        )
    if structure.hierarchy_point_count != expected_point_count:
        failures.append(
            "hierarchy has "
            f"{structure.hierarchy_point_count} points, expected {expected_point_count}"
        )
    if not structure.root_present:
        failures.append("root hierarchy node is missing")
    if not structure.all_nodes_reachable:
        failures.append(
            f"{structure.unreachable_nodes} hierarchy nodes are unreachable from the root"
        )
    if structure.invalid_node_keys:
        failures.append(
            f"{structure.invalid_node_keys} hierarchy node keys are invalid"
        )
    if failures:
        raise ValueError("invalid COPC: " + "; ".join(failures))
    return structure


def flatten_copc_hierarchy(path: Path) -> int:
    """Rewrite a paged hierarchy as one flat page and return pages removed.

    ``copc_converter`` v0.11.0 deliberately pages every three octree levels.
    That is useful for clients that load pages lazily, but Giro3D 2.0.3 eagerly
    walks the complete hierarchy and turns a modest tree into thousands of
    serial range requests. Point chunks are untouched; only the terminal COPC
    hierarchy EVLR and the root-page pointer in the COPC info VLR are changed.

    The publisher calls this while its output is still a disposable partial
    file. Temporal-index output is not currently supported and is rejected by
    requiring the hierarchy to be the file's sole EVLR.
    """

    path = Path(path)
    with path.open("r+b") as handle:
        header = _read_exact(handle, LAS_14_HEADER_SIZE, "LAS 1.4 header")
        if header[:4] != b"LASF" or (header[24], header[25]) != (1, 4):
            raise ValueError("hierarchy flattening requires a LAS 1.4 file")
        (evlr_start,) = struct.unpack_from("<Q", header, 235)
        (evlr_count,) = struct.unpack_from("<I", header, 243)
        if evlr_count != 1:
            raise ValueError(
                "hierarchy flattening requires exactly one EVLR "
                f"(found {evlr_count}); temporal-index COPC is unsupported"
            )

        handle.seek(LAS_14_HEADER_SIZE)
        vlr_header = _read_exact(handle, VLR_HEADER_SIZE, "first VLR header")
        user_id = vlr_header[2:18].rstrip(b"\x00").decode("ascii", "replace")
        (record_id,) = struct.unpack_from("<H", vlr_header, 18)
        if (user_id, record_id) != ("copc", 1):
            raise ValueError("first VLR is not the COPC info VLR")
        info = _read_exact(handle, COPC_INFO_SIZE, "COPC info VLR")
        root_offset, root_size = struct.unpack_from("<2Q", info, 40)

        entries: dict[tuple[int, int, int, int], bytes] = {}
        pending = [(root_offset, root_size)]
        visited: set[tuple[int, int]] = set()
        while pending:
            page_offset, page_size = pending.pop()
            reference = (page_offset, page_size)
            if reference in visited:
                raise ValueError(f"COPC hierarchy page cycle at {reference}")
            visited.add(reference)
            if page_size <= 0 or page_size % HIERARCHY_ENTRY_SIZE:
                raise ValueError(f"invalid hierarchy page size: {page_size}")
            handle.seek(page_offset)
            page = _read_exact(handle, page_size, "COPC hierarchy page")
            for position in range(0, len(page), HIERARCHY_ENTRY_SIZE):
                raw = page[position : position + HIERARCHY_ENTRY_SIZE]
                key = struct.unpack_from("<4i", raw, 0)
                child_offset, byte_size, point_count = struct.unpack_from(
                    "<Qii", raw, 16
                )
                if point_count == -1:
                    pending.append((child_offset, byte_size))
                    continue
                if key in entries:
                    raise ValueError(f"duplicate COPC hierarchy node: {key}")
                entries[key] = raw

        handle.seek(evlr_start)
        evlr_header = bytearray(
            _read_exact(handle, EVLR_HEADER_SIZE, "hierarchy EVLR header")
        )
        evlr_user_id = evlr_header[2:18].rstrip(b"\x00").decode("ascii", "replace")
        (evlr_record_id,) = struct.unpack_from("<H", evlr_header, 18)
        if (evlr_user_id, evlr_record_id) != ("copc", 1000):
            raise ValueError("sole EVLR is not the COPC hierarchy EVLR")

        payload = b"".join(entries[key] for key in sorted(entries))
        struct.pack_into("<Q", evlr_header, 20, len(payload))
        handle.seek(evlr_start)
        handle.write(evlr_header)
        handle.write(payload)
        handle.truncate()

        flat_root_offset = evlr_start + EVLR_HEADER_SIZE
        handle.seek(LAS_14_HEADER_SIZE + VLR_HEADER_SIZE + 40)
        handle.write(struct.pack("<2Q", flat_root_offset, len(payload)))
        handle.flush()

    return max(0, len(visited) - 1)
