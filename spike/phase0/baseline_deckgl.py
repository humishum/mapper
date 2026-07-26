#!/usr/bin/env python3
"""Measure the existing deck.gl viewer's data path - the Phase 0 "before" record.

Plan step 5 asks for a baseline of the current viewer. The interesting cost is not React:
it is `viewer/backend/data_service.py`, which reads a whole PLY into RAM, random-samples it
down to MAX_POINTS, converts to GPS with a flat-earth approximation, and ships the result as
**hex-encoded** JSON. This script times that exact code path on the same fixture the COPC
spike uses, so the two are comparable.

Nothing here is modified or fixed: per the plan, that path gets no further engineering.

Usage:
    python baseline_deckgl.py --ply /path/to/aligned_pointcloud.ply --report out/baseline.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--max-points", type=int, default=100_000,
                    help="viewer/backend/config.py MAX_POINTS (default 100000)")
    ap.add_argument("--origin", type=float, nargs=3, default=(37.5108657, -121.8987138, 265.913),
                    help="lat lon alt from the capture metadata.json")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    # Imported here so the timing below covers only the data path.
    from viewer.backend.coordinate_transform import transform_pointcloud_to_gps
    from viewer.backend.data_service import PointCloudDataService

    service = PointCloudDataService(args.ply.parent)
    timings: dict[str, float] = {}

    t = time.perf_counter()
    positions, colors = service._load_ply_file(args.ply, args.max_points)
    timings["read_ply_and_downsample"] = round(time.perf_counter() - t, 3)
    rss_after_read = peak_rss_bytes()

    lat, lon, alt = args.origin
    t = time.perf_counter()
    gps = transform_pointcloud_to_gps(positions, lat, lon, alt)
    timings["transform_to_gps"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    payload = service._format_response(gps, colors, "baseline")
    timings["hex_encode_response"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    body = json.dumps(payload)
    timings["json_serialize"] = round(time.perf_counter() - t, 3)

    ply_bytes = args.ply.stat().st_size
    header_points = _ply_point_count(args.ply)
    wire_bytes = len(body.encode("utf-8"))

    report = {
        "tool": "spike/phase0/baseline_deckgl.py",
        "note": "measures viewer/backend/data_service.py, the path the plan replaces",
        "fixture": {
            "path": str(args.ply),
            "bytes": ply_bytes,
            "points_in_file": header_points,
        },
        "viewer_limits": {
            "max_points_served": args.max_points,
            "fraction_of_artifact_shown": round(args.max_points / header_points, 6),
            "sampling": "uniform random without replacement (np.random.choice), then sorted",
            "lod": "none - one flat buffer, no hierarchy",
        },
        "timings_seconds": timings,
        "total_seconds": round(sum(timings.values()), 3),
        "wire": {
            "response_bytes": wire_bytes,
            "bytes_per_point": round(wire_bytes / args.max_points, 2),
            "encoding": "hex string in JSON (2 chars per byte)",
            "positions_dtype": "float32 x3 (after float64 GPS transform)",
            "range_requests": False,
            "cacheable": False,
        },
        "server_memory": {
            "peak_rss_bytes_after_read": rss_after_read,
            "peak_rss_bytes_total": peak_rss_bytes(),
            "note": "whole PLY is materialised in RAM per request before downsampling",
        },
        "correctness_notes": [
            "GPS transform is a flat-earth approximation (EARTH_RADIUS_M constant, no ellipsoid)",
            "points are placed at the capture's first GPS fix regardless of alignment quality",
            "no alignment status is carried, so unaligned data renders as if georeferenced",
        ],
    }

    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.report}")
    return 0


def _ply_point_count(path: Path) -> int:
    with path.open("rb") as fh:
        while True:
            line = fh.readline()
            if not line or line.strip() == b"end_header":
                break
            if line.startswith(b"element vertex"):
                return int(line.split()[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
