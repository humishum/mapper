#!/usr/bin/env python3
"""Read COPC/LAS structure with the standard library only.

Phase 0 measurement tool. Deliberately dependency-free so hierarchy numbers can be
produced from the repo venv without the PDAL environment, and so the numbers come
from the bytes on disk rather than from the writer that produced them.

Reports the fields a viewer actually consumes when it opens a COPC file:
LAS header (scale/offset/bounds/point format), the COPC info VLR (root spacing,
octree center/halfsize), and the full octree hierarchy walked page by page
(node count, point count and byte size per level). It also verifies that every
hierarchy node is reachable from the root; matching header and hierarchy point
counts alone do not prove that a renderer can traverse the full artifact.

Usage:
    python copc_stats.py FILE.copc.laz [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

LAS_HEADER_SIZE = 375
VLR_HEADER_SIZE = 54
COPC_INFO_OFFSET = LAS_HEADER_SIZE + VLR_HEADER_SIZE  # COPC info VLR must be first
COPC_INFO_SIZE = 160
HIER_ENTRY_SIZE = 32


@dataclass
class LasHeader:
    version: str
    point_format: int
    point_length: int
    point_count: int
    scale: tuple[float, float, float]
    offset: tuple[float, float, float]
    mins: tuple[float, float, float]
    maxs: tuple[float, float, float]
    offset_to_point_data: int
    vlr_count: int
    evlr_count: int
    generating_software: str

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "point_format": self.point_format,
            "point_length_bytes": self.point_length,
            "point_count": self.point_count,
            "scale": list(self.scale),
            "offset": list(self.offset),
            "bounds_min": list(self.mins),
            "bounds_max": list(self.maxs),
            "extent_m": [self.maxs[i] - self.mins[i] for i in range(3)],
            "offset_to_point_data": self.offset_to_point_data,
            "vlr_count": self.vlr_count,
            "evlr_count": self.evlr_count,
            "generating_software": self.generating_software,
        }


@dataclass
class CopcInfo:
    center: tuple[float, float, float]
    halfsize: float
    spacing: float
    root_hier_offset: int
    root_hier_size: int

    def to_dict(self) -> dict:
        return {
            "octree_center": list(self.center),
            "octree_halfsize_m": self.halfsize,
            "root_spacing_m": self.spacing,
            "root_hier_offset": self.root_hier_offset,
            "root_hier_size": self.root_hier_size,
        }


@dataclass
class LevelStats:
    nodes: int = 0
    points: int = 0
    bytes: int = 0
    empty_nodes: int = 0
    max_points_in_node: int = 0
    keys: list[tuple[int, int, int, int]] = field(default_factory=list)
    point_counts_by_key: list[int] = field(default_factory=list)


def read_las_header(fh) -> LasHeader:
    fh.seek(0)
    buf = fh.read(LAS_HEADER_SIZE)
    if buf[:4] != b"LASF":
        raise ValueError("not a LAS/LAZ file (missing LASF signature)")
    version = f"{buf[24]}.{buf[25]}"
    software = buf[58:90].rstrip(b"\x00 ").decode("ascii", "replace")
    (offset_to_point_data, vlr_count) = struct.unpack_from("<II", buf, 96)
    point_format = buf[104] & 0x3F  # high bits flag LAZ compression
    (point_length,) = struct.unpack_from("<H", buf, 105)
    (legacy_count,) = struct.unpack_from("<I", buf, 107)
    scale = struct.unpack_from("<3d", buf, 131)
    offset = struct.unpack_from("<3d", buf, 155)
    max_x, min_x, max_y, min_y, max_z, min_z = struct.unpack_from("<6d", buf, 179)
    (evlr_count,) = struct.unpack_from("<I", buf, 243)
    (count_1_4,) = struct.unpack_from("<Q", buf, 247)
    return LasHeader(
        version=version,
        point_format=point_format,
        point_length=point_length,
        point_count=count_1_4 or legacy_count,
        scale=scale,
        offset=offset,
        mins=(min_x, min_y, min_z),
        maxs=(max_x, max_y, max_z),
        offset_to_point_data=offset_to_point_data,
        vlr_count=vlr_count,
        evlr_count=evlr_count,
        generating_software=software,
    )


def read_copc_info(fh) -> CopcInfo:
    fh.seek(LAS_HEADER_SIZE)
    vlr_header = fh.read(VLR_HEADER_SIZE)
    user_id = vlr_header[2:18].rstrip(b"\x00").decode("ascii", "replace")
    (record_id,) = struct.unpack_from("<H", vlr_header, 18)
    if user_id != "copc" or record_id != 1:
        raise ValueError(
            f"first VLR is not the COPC info VLR (user_id={user_id!r}, record_id={record_id}) "
            "- file is LAZ but not COPC"
        )
    fh.seek(COPC_INFO_OFFSET)
    buf = fh.read(COPC_INFO_SIZE)
    cx, cy, cz, halfsize, spacing = struct.unpack_from("<5d", buf, 0)
    root_hier_offset, root_hier_size = struct.unpack_from("<2Q", buf, 40)
    return CopcInfo((cx, cy, cz), halfsize, spacing, root_hier_offset, root_hier_size)


def walk_hierarchy(fh, info: CopcInfo, keep_keys: bool = False) -> dict[int, LevelStats]:
    """Walk every hierarchy page; return per-level stats.

    A page entry with point_count == -1 is a reference to a child page, which is how
    COPC keeps the hierarchy itself lazily loadable. Counting pages tells us how many
    round trips a client makes just to learn the tree shape.
    """
    levels: dict[int, LevelStats] = {}
    pages_read = 0
    pending = [(info.root_hier_offset, info.root_hier_size)]
    while pending:
        offset, size = pending.pop()
        fh.seek(offset)
        buf = fh.read(size)
        pages_read += 1
        for i in range(0, len(buf) - HIER_ENTRY_SIZE + 1, HIER_ENTRY_SIZE):
            level, kx, ky, kz = struct.unpack_from("<4i", buf, i)
            child_offset, byte_size, point_count = struct.unpack_from("<Qii", buf, i + 16)
            if point_count == -1:
                pending.append((child_offset, byte_size))
                continue
            stats = levels.setdefault(level, LevelStats())
            stats.nodes += 1
            stats.points += point_count
            stats.bytes += byte_size
            if point_count == 0:
                stats.empty_nodes += 1
            stats.max_points_in_node = max(stats.max_points_in_node, point_count)
            if keep_keys:
                stats.keys.append((level, kx, ky, kz))
                stats.point_counts_by_key.append(point_count)
    levels["_pages_read"] = pages_read  # type: ignore[index]
    return levels


def summarize(path: Path, keep_keys: bool = False) -> dict:
    with path.open("rb") as fh:
        header = read_las_header(fh)
        info = read_copc_info(fh)
        # Keys are always collected for the reachability validation below. They are
        # only retained in the JSON output when --keys is explicitly requested.
        levels = walk_hierarchy(fh, info, keep_keys=True)

    pages_read = levels.pop("_pages_read")  # type: ignore[arg-type]
    point_counts_by_key = {
        key: count
        for stats in levels.values()
        for key, count in zip(stats.keys, stats.point_counts_by_key)
    }
    all_keys = set(point_counts_by_key)
    root_key = (0, 0, 0, 0)
    reachable: set[tuple[int, int, int, int]] = set()
    invalid_keys = 0
    for key in sorted(all_keys):
        level, x, y, z = key
        limit = 1 << level
        if not (0 <= x < limit and 0 <= y < limit and 0 <= z < limit):
            invalid_keys += 1
            continue
        if key == root_key:
            reachable.add(key)
        elif (level - 1, x // 2, y // 2, z // 2) in reachable:
            reachable.add(key)
    unreachable = all_keys - reachable

    per_level = []
    total_nodes = total_points = total_bytes = 0
    for level in sorted(levels):
        stats = levels[level]
        total_nodes += stats.nodes
        total_points += stats.points
        total_bytes += stats.bytes
        level_report = {
                "level": level,
                "nodes": stats.nodes,
                "points": stats.points,
                "bytes": stats.bytes,
                "mean_points_per_node": round(stats.points / stats.nodes, 1) if stats.nodes else 0,
                "max_points_in_node": stats.max_points_in_node,
                "empty_nodes": stats.empty_nodes,
                "spacing_m": round(info.spacing / (2**level), 4),
        }
        if keep_keys:
            level_report["keys"] = [list(key) for key in stats.keys]
        per_level.append(level_report)

    file_bytes = path.stat().st_size
    return {
        "file": str(path),
        "file_bytes": file_bytes,
        "las_header": header.to_dict(),
        "copc_info": info.to_dict(),
        "hierarchy": {
            "pages_read": pages_read,
            "depth": max(levels) if levels else 0,
            "total_nodes": total_nodes,
            "total_points_in_hierarchy": total_points,
            "total_point_bytes": total_bytes,
            "bytes_per_point": round(total_bytes / total_points, 3) if total_points else 0,
            "header_point_count_matches_hierarchy": total_points == header.point_count,
            "root_node_present": root_key in all_keys,
            "all_nodes_reachable_from_root": len(unreachable) == 0,
            "reachable_nodes": len(reachable),
            "reachable_points": sum(point_counts_by_key[key] for key in reachable),
            "unreachable_nodes": len(unreachable),
            "unreachable_points": sum(point_counts_by_key[key] for key in unreachable),
            "invalid_node_keys": invalid_keys,
            "root_node_points": per_level[0]["points"] if per_level else 0,
            "root_node_bytes": per_level[0]["bytes"] if per_level else 0,
            "per_level": per_level,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", type=Path)
    ap.add_argument("--json", type=Path, help="write the report to this path as JSON")
    ap.add_argument("--keys", action="store_true", help="collect node keys (large output)")
    args = ap.parse_args()

    report = summarize(args.file, keep_keys=args.keys)
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(text + "\n")
        print(f"wrote {args.json}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
