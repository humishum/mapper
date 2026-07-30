# PLY Viewer (deck.gl)

This is an intentionally isolated, lightweight debugging utility for opening a
local PLY by hand. It is not part of the Phase 2 application, production
dependencies, API, catalog, or acceptance path. The production viewer lives
under `viewer/` and streams packaged COPC data through Giro3D.

## Usage

1. Start a simple web server from this directory:

```bash
python -m http.server 8000
```

2. Open `http://localhost:8000` in a browser.
3. Use the file picker or drag/drop a `.ply` file.
4. Or paste a URL served from a Linux host into the URL field and click Load.

Example file from this repo:

`/home/ape/repos/mapper/data/output_kings_canyon/kings_canyon_must3r_20260105_211309/outputs/kings_canyon_1/aligned_pointcloud.ply`

## Notes

- The viewer reads local files via the browser File API; it does not upload anything.
- If colors exist in the PLY, they are used automatically. Otherwise points render white.
- Very large PLYs can exceed browser memory; downsample before loading if needed.

## Downsampling script

Create a lighter PLY while preserving color:

```bash
python downsample_ply.py \\
  --input /path/to/input.ply \\
  --output /path/to/output_downsampled.ply \\
  --target-points 2000000
```

Other modes:

```bash
python downsample_ply.py --input in.ply --output out.ply --stride 10
python downsample_ply.py --input in.ply --output out.ply --fraction 0.05
```

The script supports binary_little_endian PLYs with vertex properties that include `x`, `y`, `z`,
and optional `red`, `green`, `blue` (or `r`, `g`, `b`). It writes a vertex-only PLY.
