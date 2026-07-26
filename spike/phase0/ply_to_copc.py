#!/usr/bin/env python3
"""Convert a reconstruction PLY into COPC/LAZ, in the local frame, with metrics.

Phase 0 spike tool. The plan's publisher spec calls for `pdal writers.copc`, but PDAL
has no uv-installable form (the PyPI `pdal` sdist needs libpdal on the system, and this
project manages Python deps with uv only). This script does the same job with the
uv-installable pieces:

* octree construction here (numpy, Morton-sorted, Potree/Entwine-style level sampling)
* LAZ chunk compression + COPC file assembly by `copclib` (copc-lib bindings)

What it produces is a spec-conformant COPC file whose LOD behaviour is representative,
which is what Phase 0 needs in order to judge Giro3D. It is *not* the production
publisher: it sorts the whole cloud in RAM, which the plan explicitly forbids for the
streaming publisher. See docs/phase0_spike_findings.md.

Canonical framing choices from the plan, applied here:

* LAS scale 0.001 / offset 0 (mm precision, +/-2147 km, no per-file offset bookkeeping)
* point format 7 (XYZ + RGB, LAS 1.4) - COPC allows only formats 6/7/8
* 8-bit PLY colour is expanded to 16-bit, because readers (Giro3D included) divide by 256
* window provenance goes in the native `PointSourceId` dimension (see --source-index-dir)

Voxel consolidation is done on the octree's own finest grid, so consolidated points stay
aligned to the node grid instead of being quantised twice.

Examples:
    # 2 cm consolidated, window provenance, full file
    python ply_to_copc.py --input aligned_pointcloud.ply --output out/mp7-voxel2cm.copc.laz \\
        --voxel 0.02 --source-index-dir /path/to/mp7/windows --report out/mp7-voxel2cm.json

    # no consolidation (every input point kept)
    python ply_to_copc.py --input aligned_pointcloud.ply --output out/mp7-full.copc.laz \\
        --report out/mp7-full.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import copclib as copc
except ImportError:  # pragma: no cover - environment guard
    sys.exit("copclib is required: uv pip install --python .venv copclib")

SPAN = 128  # points per node edge at the node's own level; COPC/Entwine convention
BITS_PER_SPAN = 7  # log2(SPAN)
MAX_AXIS_BITS = 21  # 3 * 21 = 63 bits, fits int64 Morton codes
PDRF = 7
PDRF7_LEN = 36
LAS_SCALE = 0.001


# --------------------------------------------------------------------------------------
# PLY reading
# --------------------------------------------------------------------------------------

_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


@dataclass
class PlyInfo:
    path: Path
    count: int
    dtype: np.dtype
    data_offset: int
    comments: list[str]

    @property
    def bytes_per_point(self) -> int:
        return self.dtype.itemsize


def parse_ply_header(path: Path) -> PlyInfo:
    """Parse a binary_little_endian PLY header without loading any point data."""
    fields: list[tuple[str, str]] = []
    comments: list[str] = []
    count = 0
    in_vertex = False
    header = bytearray()
    with path.open("rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt_line = fh.readline().decode("ascii", "replace").strip()
        if "binary_little_endian" not in fmt_line:
            raise ValueError(f"only binary_little_endian PLY is supported, got: {fmt_line!r}")
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError("unexpected end of PLY header")
            header += raw
            line = raw.decode("ascii", "replace").strip()
            if line == "end_header":
                break
            if line.startswith("comment"):
                comments.append(line[len("comment"):].strip())
                continue
            if line.startswith("element"):
                parts = line.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
                continue
            if line.startswith("property") and in_vertex:
                _, ply_type, name = line.split(maxsplit=2)
                if ply_type not in _PLY_TYPES:
                    raise ValueError(f"unsupported PLY property type {ply_type!r}")
                fields.append((name.strip(), _PLY_TYPES[ply_type]))
        data_offset = fh.tell()

    names = {name for name, _ in fields}
    missing = {"x", "y", "z"} - names
    if missing:
        raise ValueError(f"PLY vertex element lacks {sorted(missing)}")
    dtype = np.dtype([(name, "<" + kind) for name, kind in fields])
    return PlyInfo(path, count, dtype, data_offset, comments)


def color_field_names(dtype: np.dtype) -> tuple[str, str, str] | None:
    names = set(dtype.names or ())
    for triple in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        if set(triple) <= names:
            return triple  # type: ignore[return-value]
    return None


# --------------------------------------------------------------------------------------
# Morton codes
# --------------------------------------------------------------------------------------

_U64 = np.uint64


def _spread_bits(v: np.ndarray) -> np.ndarray:
    """Insert two zero bits after each of the low 21 bits of each value."""
    v = v.astype(np.uint64, copy=True)
    v = (v | (v << _U64(32))) & _U64(0x1F00000000FFFF)
    v = (v | (v << _U64(16))) & _U64(0x1F0000FF0000FF)
    v = (v | (v << _U64(8))) & _U64(0x100F00F00F00F00F)
    v = (v | (v << _U64(4))) & _U64(0x10C30C30C30C30C3)
    v = (v | (v << _U64(2))) & _U64(0x1249249249249249)
    return v


def morton_encode(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
    """Interleave three grid indices into one int64 Morton code (x in the low bit)."""
    return (
        _spread_bits(qx) | (_spread_bits(qy) << _U64(1)) | (_spread_bits(qz) << _U64(2))
    ).astype(np.int64)


def morton_decode(code: int, axis_bits: int) -> tuple[int, int, int]:
    """Inverse of morton_encode for a single code (used once per node)."""
    x = y = z = 0
    for i in range(axis_bits):
        x |= ((code >> (3 * i)) & 1) << i
        y |= ((code >> (3 * i + 1)) & 1) << i
        z |= ((code >> (3 * i + 2)) & 1) << i
    return x, y, z


# --------------------------------------------------------------------------------------
# Metrics plumbing
# --------------------------------------------------------------------------------------


@dataclass
class Report:
    stages: dict[str, float] = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    def stage(self, name: str):
        return _Stage(self, name)

    def peak_rss_bytes(self) -> int:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class _Stage:
    def __init__(self, report: Report, name: str):
        self.report = report
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"  [{self.name}] ...", flush=True)
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        self.report.stages[self.name] = round(dt, 3)
        self.report._order.append(self.name)
        print(f"  [{self.name}] {dt:.2f}s", flush=True)
        return False


# --------------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------------


def compute_bounds(info: PlyInfo, chunk: int, limit: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    """First pass: exact bounds plus a count of non-finite points (a validation gate)."""
    mm = np.memmap(info.path, dtype=info.dtype, mode="r", offset=info.data_offset, shape=(info.count,))
    n = min(info.count, limit) if limit else info.count
    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    nonfinite = 0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = mm[start:stop]
        xyz = np.stack([block["x"], block["y"], block["z"]], axis=1).astype(np.float64)
        finite = np.isfinite(xyz).all(axis=1)
        nonfinite += int((~finite).sum())
        if finite.any():
            good = xyz[finite]
            lo = np.minimum(lo, good.min(axis=0))
            hi = np.maximum(hi, good.max(axis=0))
    del mm
    return lo, hi, nonfinite


def source_index_from_windows(window_dir: Path, total_points: int) -> tuple[np.ndarray, dict]:
    """Build a per-point window index from the per-window PLY point counts.

    The merged cloud is the concatenation of the per-window clouds in window order, so the
    window index of every point is recoverable exactly - no heuristics. The sum is checked
    against the merged point count and the conversion fails if it disagrees, which is the
    only thing that makes this claim trustworthy.
    """
    window_dirs = sorted(
        (d for d in window_dir.iterdir() if d.is_dir() and re.fullmatch(r"window_\d+", d.name)),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not window_dirs:
        raise ValueError(f"no window_NNN directories under {window_dir}")

    counts: list[int] = []
    names: list[str] = []
    for d in window_dirs:
        ply = d / "pointcloud.ply"
        if not ply.exists():
            candidates = sorted(d.glob("*.ply"))
            if not candidates:
                raise ValueError(f"no PLY in {d}")
            ply = candidates[0]
        counts.append(parse_ply_header(ply).count)
        names.append(d.name)

    total = sum(counts)
    if total != total_points:
        raise ValueError(
            f"window point counts sum to {total:,} but the merged cloud has {total_points:,}; "
            "the merged file is not a plain concatenation of these windows, so window "
            "provenance cannot be derived this way"
        )
    if len(counts) > 65535:
        raise ValueError("more than 65535 windows; PointSourceId (uint16) cannot hold the index")

    index = np.repeat(np.arange(len(counts), dtype=np.uint16), counts)
    lookup = {
        "dimension": "PointSourceId",
        "granularity": "window",
        "windows": [
            {"source_index": i, "name": name, "point_count": c}
            for i, (name, c) in enumerate(zip(names, counts))
        ],
    }
    return index, lookup


def convert(args: argparse.Namespace) -> dict:
    report = Report()
    info = parse_ply_header(args.input)
    n_total = min(info.count, args.limit) if args.limit else info.count
    color_fields = color_field_names(info.dtype)

    print(f"input: {args.input}")
    print(f"  {info.count:,} points, {info.bytes_per_point} bytes/point, "
          f"{args.input.stat().st_size / 1e9:.2f} GB, fields={info.dtype.names}")
    if args.limit:
        print(f"  --limit: converting first {n_total:,} points only")

    # ---- pass 1: bounds -------------------------------------------------------------
    with report.stage("bounds"):
        lo, hi, nonfinite = compute_bounds(info, args.chunk, args.limit)
    extent = hi - lo
    cube_size = float(max(extent.max(), 1e-6)) * (1.0 + 1e-9)
    center = (lo + hi) / 2.0
    cube_min = center - cube_size / 2.0
    print(f"  bounds min={lo.round(3).tolist()} max={hi.round(3).tolist()}")
    print(f"  extent={extent.round(3).tolist()} cube={cube_size:.3f} nonfinite={nonfinite:,}")

    # ---- octree geometry ------------------------------------------------------------
    # Depth is driven by how many points have to fit in a leaf, not by the consolidation
    # grid: subdivision has to continue until leaves are small enough to draw at once.
    root_spacing = cube_size / SPAN
    depth_cap = MAX_AXIS_BITS - BITS_PER_SPAN
    voxel_depth = None
    if args.voxel:
        voxel_depth = int(min(max(math.ceil(math.log2(max(root_spacing / args.voxel, 1.0))), 0), depth_cap))
    if args.max_depth is not None:
        depth = int(min(args.max_depth, depth_cap))
    else:
        need = max(n_total / max(args.max_node_points, 1), 1.0)
        depth = int(min(math.ceil(math.log2(need) / 3) + 3, depth_cap))
    if voxel_depth is not None:
        # The consolidation grid must exist inside the tree, or the requested spacing would
        # be silently coarsened to whatever depth the tree happened to have.
        depth = max(depth, voxel_depth)
    axis_bits = depth + BITS_PER_SPAN
    grid = 1 << axis_bits
    finest_cell = cube_size / grid

    voxel_shift = 0
    effective_voxel = None
    if voxel_depth is not None:
        voxel_shift = 3 * (depth - voxel_depth)
        effective_voxel = cube_size / (1 << (voxel_depth + BITS_PER_SPAN))

    print(f"  octree: depth={depth} root_spacing={root_spacing:.4f} "
          f"finest_cell={finest_cell:.5f} grid={grid}^3 max_node_points={args.max_node_points:,}")
    if args.voxel:
        print(f"  consolidation grid: level {voxel_depth} -> cell {effective_voxel:.5f} "
              f"(requested {args.voxel})")

    # ---- pass 2: load, quantise, Morton ---------------------------------------------
    with report.stage("load+quantize"):
        mm = np.memmap(info.path, dtype=info.dtype, mode="r", offset=info.data_offset,
                       shape=(info.count,))
        xyz_i32 = np.empty((n_total, 3), dtype=np.int32)
        rgb16 = np.zeros((n_total, 3), dtype=np.uint16) if color_fields else None
        morton = np.empty(n_total, dtype=np.int64)
        keep_mask = np.ones(n_total, dtype=bool)
        inv_scale = 1.0 / LAS_SCALE
        color_gain = 257 if args.color_bits == 8 else 1  # 8->16 bit, exact at both ends

        for start in range(0, n_total, args.chunk):
            stop = min(start + args.chunk, n_total)
            block = mm[start:stop]
            xyz = np.stack([block["x"], block["y"], block["z"]], axis=1).astype(np.float64)
            finite = np.isfinite(xyz).all(axis=1)
            if not finite.all():
                keep_mask[start:stop] = finite
                xyz[~finite] = lo  # placeholder; dropped below
            np.multiply(xyz, inv_scale, out=xyz)
            np.rint(xyz, out=xyz)
            xyz_i32[start:stop] = xyz.astype(np.int32)

            q = ((np.stack([block["x"], block["y"], block["z"]], axis=1).astype(np.float64) - cube_min)
                 / cube_size * grid)
            np.floor(q, out=q)
            np.clip(q, 0, grid - 1, out=q)
            qi = q.astype(np.int64)
            morton[start:stop] = morton_encode(qi[:, 0], qi[:, 1], qi[:, 2])

            if color_fields is not None:
                cr, cg, cb = color_fields
                rgb16[start:stop, 0] = block[cr].astype(np.uint16) * color_gain
                rgb16[start:stop, 1] = block[cg].astype(np.uint16) * color_gain
                rgb16[start:stop, 2] = block[cb].astype(np.uint16) * color_gain
        del mm

    # ---- provenance -----------------------------------------------------------------
    source_index = None
    source_lookup = None
    if args.source_index_dir:
        with report.stage("source-index"):
            source_index, source_lookup = source_index_from_windows(
                args.source_index_dir, info.count
            )
            if args.limit:
                source_index = source_index[:n_total]

    # ---- drop non-finite ------------------------------------------------------------
    if not keep_mask.all():
        with report.stage("drop-nonfinite"):
            xyz_i32 = xyz_i32[keep_mask]
            morton = morton[keep_mask]
            if rgb16 is not None:
                rgb16 = rgb16[keep_mask]
            if source_index is not None:
                source_index = source_index[keep_mask]

    n_loaded = len(morton)

    # ---- sort by Morton code --------------------------------------------------------
    with report.stage("morton-sort"):
        # introsort, not stable: mergesort's workspace is another N*8 bytes, and at 147M
        # points that is 1.2 GB we do not have to spend. Ties are points sharing the finest
        # octree cell, where any single survivor is equally valid.
        order = np.argsort(morton, kind="quicksort")
        morton = morton[order]
        xyz_i32 = xyz_i32[order]
        if rgb16 is not None:
            rgb16 = rgb16[order]
        if source_index is not None:
            source_index = source_index[order]
        del order

    # ---- voxel consolidation --------------------------------------------------------
    n_before_voxel = n_loaded
    if args.voxel:
        with report.stage("voxel-consolidate"):
            cell = morton >> np.int64(voxel_shift) if voxel_shift else morton
            first = np.empty(len(morton), dtype=bool)
            first[0] = True
            np.not_equal(cell[1:], cell[:-1], out=first[1:])
            kept = np.flatnonzero(first)
            morton = morton[kept]
            xyz_i32 = xyz_i32[kept]
            if rgb16 is not None:
                rgb16 = rgb16[kept]
            if source_index is not None:
                source_index = source_index[kept]
            del first, kept
        print(f"  voxel {effective_voxel:.5f}: {n_before_voxel:,} -> {len(morton):,} points "
              f"(redundancy factor {n_before_voxel / max(len(morton), 1):.2f}x)")

    n_out = len(morton)

    # ---- build octree levels and write ---------------------------------------------
    with report.stage("octree+write"):
        node_stats = write_copc(
            args.output, morton, xyz_i32, rgb16, source_index,
            depth=depth, axis_bits=axis_bits, center=center, halfsize=cube_size / 2.0,
            root_spacing=root_spacing, lo=lo, hi=hi, wkt=args.wkt or "",
            max_node_points=args.max_node_points,
        )

    out_bytes = args.output.stat().st_size
    written = node_stats["points_written"]
    if written != n_out:
        print(f"  WARNING: wrote {written:,} of {n_out:,} points "
              f"({n_out - written:,} unplaced at the depth cap)")
    result = {
        "tool": "spike/phase0/ply_to_copc.py",
        "input": {
            "path": str(args.input),
            "bytes": args.input.stat().st_size,
            "points": info.count,
            "converted_points": n_total,
            "bytes_per_point": info.bytes_per_point,
            "fields": list(info.dtype.names or ()),
            "comments": info.comments,
        },
        "output": {
            "path": str(args.output),
            "bytes": out_bytes,
            "points": written,
            "point_format": PDRF,
            "las_scale": [LAS_SCALE] * 3,
            "las_offset": [0.0, 0.0, 0.0],
            "compression_ratio_vs_ply": round(args.input.stat().st_size / out_bytes, 2),
            "bytes_per_point": round(out_bytes / max(written, 1), 3),
        },
        "frame": {
            "frame": "model_local" if not args.wkt else "declared",
            "units": "unknown - source model is not metric" if args.assume_non_metric else "metre",
            "bounds_min": lo.tolist(),
            "bounds_max": hi.tolist(),
            "extent": extent.tolist(),
            "octree_cube_center": center.tolist(),
            "octree_cube_size": cube_size,
        },
        "octree": {
            "span": SPAN,
            "max_depth": depth,
            "axis_bits": axis_bits,
            "root_spacing": root_spacing,
            "finest_cell_size": finest_cell,
            **node_stats,
        },
        "consolidation": {
            "requested_voxel_size": args.voxel,
            "effective_voxel_size": effective_voxel,
            "consolidation_octree_level": voxel_depth if args.voxel else None,
            "method": "first point per octree cell at the consolidation level" if args.voxel else None,
            "points_in": n_before_voxel,
            "points_out": n_out,
            "redundancy_factor": round(n_before_voxel / max(n_out, 1), 4),
        },
        "validation": {
            "nonfinite_points_dropped": nonfinite,
            "color_source_bits": args.color_bits,
            "color_gain_applied": 257 if args.color_bits == 8 else 1,
            "has_color": color_fields is not None,
        },
        "provenance": source_lookup,
        "timings_seconds": report.stages,
        "peak_rss_bytes": report.peak_rss_bytes(),
        "wall_clock_seconds": round(sum(report.stages.values()), 3),
    }
    return result


def write_copc(
    out_path: Path,
    morton: np.ndarray,
    xyz_i32: np.ndarray,
    rgb16: np.ndarray | None,
    source_index: np.ndarray | None,
    *,
    depth: int,
    axis_bits: int,
    center: np.ndarray,
    halfsize: float,
    root_spacing: float,
    lo: np.ndarray,
    hi: np.ndarray,
    wkt: str,
    max_node_points: int,
) -> dict:
    """Write nodes level by level, sampling one point per span-cell at each level.

    Points arrive Morton-sorted, so every octree node is a contiguous slice and every
    node's span-grid cells are contiguous runs inside it: level assignment is a couple of
    vectorised diffs per level rather than a per-point traversal.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = copc.CopcConfigWriter(
        PDRF,
        copc.Vector3(LAS_SCALE, LAS_SCALE, LAS_SCALE),
        copc.Vector3(0.0, 0.0, 0.0),
        wkt,
    )
    cfg.copc_info.center_x = float(center[0])
    cfg.copc_info.center_y = float(center[1])
    cfg.copc_info.center_z = float(center[2])
    cfg.copc_info.halfsize = float(halfsize)
    cfg.copc_info.spacing = float(root_spacing)
    cfg.las_header.min = copc.Vector3(*(float(v) for v in lo))
    cfg.las_header.max = copc.Vector3(*(float(v) for v in hi))
    cfg.las_header.generating_software = "mapper phase0 ply_to_copc"

    writer = copc.FileWriter(str(out_path), cfg)

    remaining = np.arange(len(morton), dtype=np.int64)
    per_level: list[dict] = []
    total_nodes = 0
    total_written = 0
    leaves = 0
    truncated_nodes = 0
    written_keys: set[tuple[int, int, int, int]] = set()

    for level in range(depth + 1):
        if remaining.size == 0:
            break
        codes = morton[remaining]
        node_shift = 3 * (axis_bits - level)
        cell_shift = 3 * (depth - level)
        last_level = level == depth

        node_keys = codes >> np.int64(node_shift) if node_shift else codes
        node_starts = np.flatnonzero(
            np.concatenate(([True], node_keys[1:] != node_keys[:-1]))
        )
        node_bounds = np.append(node_starts, len(node_keys))
        node_sizes = np.diff(node_bounds)

        cell_keys = codes >> np.int64(cell_shift) if cell_shift else codes
        # First occurrence of each span-cell inside each node -> this level's sample.
        new_cell = np.concatenate(([True], cell_keys[1:] != cell_keys[:-1]))
        new_cell[node_starts] = True
        keep_flags = new_cell

        # A node small enough to draw at once is a leaf: keep all of its points and stop
        # subdividing. Without this rule the sample-one-per-cell pass silently discards
        # every extra point sharing a cell at the deepest level.
        leaf_nodes = node_sizes <= max_node_points
        if last_level:
            leaf_nodes[:] = True
        for i in np.flatnonzero(leaf_nodes):
            keep_flags[node_bounds[i]:node_bounds[i + 1]] = True
        if last_level:
            truncated_nodes = int((node_sizes > max_node_points).sum())

        level_points = 0
        level_nodes = 0
        level_max = 0
        for i in range(len(node_starts)):
            start, stop = node_bounds[i], node_bounds[i + 1]
            sel = remaining[start:stop][keep_flags[start:stop]]
            if sel.size == 0:
                continue
            code = int(node_keys[start])
            gx, gy, gz = morton_decode(code, level) if level else (0, 0, 0)
            key_tuple = (level, gx, gy, gz)
            if level:
                parent = (level - 1, gx // 2, gy // 2, gz // 2)
                if parent not in written_keys:
                    raise RuntimeError(
                        f"refusing to write unreachable COPC node {key_tuple}: "
                        f"parent {parent} was not written"
                    )
            key = copc.VoxelKey(level, gx, gy, gz)
            buf = pack_pdrf7(xyz_i32[sel], rgb16[sel] if rgb16 is not None else None,
                             source_index[sel] if source_index is not None else None)
            writer.AddNode(key, copc.VectorChar(buf))
            written_keys.add(key_tuple)
            level_points += int(sel.size)
            level_nodes += 1
            level_max = max(level_max, int(sel.size))

        remaining = remaining[~keep_flags]
        total_nodes += level_nodes
        total_written += level_points
        leaves += int(leaf_nodes.sum())
        per_level.append({
            "level": level,
            "nodes": level_nodes,
            "points": level_points,
            "leaf_nodes": int(leaf_nodes.sum()),
            "max_points_in_node": level_max,
            "mean_points_per_node": round(level_points / level_nodes, 1) if level_nodes else 0,
            "spacing_m": round(root_spacing / (2 ** level), 4),
        })
        print(f"    level {level}: {level_nodes:,} nodes, {level_points:,} points "
              f"({remaining.size:,} left)", flush=True)

    writer.Close()
    return {
        "total_nodes": total_nodes,
        "leaf_nodes": leaves,
        "levels_written": len(per_level),
        "points_written": total_written,
        "unplaced_points": int(remaining.size),
        "nodes_exceeding_max_at_depth_cap": truncated_nodes,
        "max_node_points": max_node_points,
        "per_level": per_level,
    }


def pack_pdrf7(
    xyz_i32: np.ndarray, rgb16: np.ndarray | None, source_index: np.ndarray | None
) -> np.ndarray:
    """Build LAS 1.4 point-format-7 records for one node as a flat int8 buffer."""
    n = len(xyz_i32)
    buf = np.zeros((n, PDRF7_LEN), dtype=np.uint8)
    buf[:, 0:12] = np.ascontiguousarray(xyz_i32).view(np.uint8).reshape(n, 12)
    buf[:, 14] = 0x11  # return 1 of 1
    if source_index is not None:
        buf[:, 20:22] = np.ascontiguousarray(source_index).view(np.uint8).reshape(n, 2)
    if rgb16 is not None:
        buf[:, 30:36] = np.ascontiguousarray(rgb16).view(np.uint8).reshape(n, 6)
    return buf.reshape(-1).view(np.int8)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--voxel", type=float, default=None,
                    help="consolidate to roughly this spacing, in source units (e.g. 0.02)")
    ap.add_argument("--max-node-points", type=int, default=100_000,
                    help="a node with at most this many points becomes a leaf (default 100000)")
    ap.add_argument("--max-depth", type=int, default=None,
                    help="override the octree depth (default: derived from point count)")
    ap.add_argument("--source-index-dir", type=Path, default=None,
                    help="directory of window_NNN/ dirs; fills PointSourceId with the window index")
    ap.add_argument("--color-bits", type=int, choices=(8, 16), default=8,
                    help="bit depth of the source PLY colour (default 8, expanded to 16)")
    ap.add_argument("--chunk", type=int, default=8_000_000, help="points per read chunk")
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N points")
    ap.add_argument("--wkt", type=str, default="", help="CRS WKT; omit for a local frame")
    ap.add_argument("--assume-non-metric", action="store_true",
                    help="record in the report that source units are not known to be metres")
    ap.add_argument("--report", type=Path, default=None, help="write metrics JSON here")
    args = ap.parse_args()

    t0 = time.perf_counter()
    result = convert(args)
    result["total_wall_clock_seconds"] = round(time.perf_counter() - t0, 3)

    print(f"\nwrote {args.output} "
          f"({result['output']['bytes'] / 1e6:.1f} MB, {result['output']['points']:,} points) "
          f"in {result['total_wall_clock_seconds']:.1f}s, "
          f"peak RSS {result['peak_rss_bytes'] / 1e9:.2f} GB")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(f"metrics: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
