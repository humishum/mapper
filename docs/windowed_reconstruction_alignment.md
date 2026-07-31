# Windowed Reconstruction and Pose-Overlap Alignment

Status: Proposal
Owner: Codex (draft)
Last updated: 2025-02-14

## Summary

Long videos exceed per-model memory budgets. The plan is to perform chunked
reconstruction inside each model (no `VideoInput` changes) and align/merge
windowed reconstructions in the experiment runner. The primary alignment method
will be pose-overlap alignment when camera poses are available, with ICP-based
fallback when they are not. GPS/IMU alignment remains a final global step with
guardrails for unreliable telemetry.

## Goals

- Enable long-video reconstruction by chunking at the model level.
- Align windowed reconstructions into a single global point cloud.
- Prefer pose-overlap alignment when poses exist; use ICP fallback otherwise.
- Preserve current `VideoInput` interface and telemetry extraction.
- Keep GPS/IMU alignment optional and robust to poor GPS quality.

## Non-goals

- Implement full SLAM (loop closure, relocalization, map fusion).
- Replace per-model reconstruction logic or external dependencies.
- Overhaul telemetry extraction or add networked data sources.

## Research: Model Output Formats and Pose Conventions

### MASt3R (must3r)

Reference: `/home/ape/repos/must3rdemo/must3r/must3r/demo/gradio.py`,
`/home/ape/repos/must3rdemo/must3r/must3r/demo/inference.py`

- `SceneState.cams2world`: list of `torch.Tensor` shape `(4, 4)` per frame.
  These are camera-to-world transforms (`c2w`), used directly in the demo.
- `scene.x_out[i]['pts3d']`: per-frame point map, shape `(H, W, 3)` in world
  coordinates when global alignment is applied. `scene.x_out[i]['pts3d_local']`
  is in local camera coordinates; it is transformed via `geotrf(c2w, ...)`.
- `scene.x_out[i]['conf']`: shape `(H, W)` confidence map.

Implication: MASt3R poses are already camera-to-world, so camera positions can
be read from the translation component `c2w[:3, 3]` without inversion.

### VGGT

Reference: `/home/ape/repos/vggt/README.md`,
`/home/ape/repos/vggt/vggt/models/vggt.py`

- `pose_encoding_to_extri_intri` returns `extrinsic` shape `(B, S, 3, 4)`.
  README states extrinsics follow OpenCV convention: camera-from-world
  (world-to-camera) transforms.
- `depth` output shape `(B, S, H, W, 1)`, `depth_conf` shape `(B, S, H, W)`.

Implication: VGGT extrinsics must be converted to `4x4` and inverted to get
camera-to-world before using camera positions for alignment. Current wrapper in
`src/models/vggt.py` does not invert; this should be corrected for alignment.

### DA3-Streaming (future model)

Reference: `/home/ape/repos/da3-streaming/da3_streaming/README.md`

- Outputs `camera_poses.txt` (per-frame extrinsics) and `intrinsic.txt`.
- Produces `pcd/combined_pcd.ply` directly.

Implication: For DA3-Streaming, poses are available but likely world-to-camera
extrinsics; they should be inverted for pose-overlap alignment.

## Proposed Configuration Changes (YAML)

Add chunking and alignment settings. Keep these in `model_config` for model-
specific behavior, and add a top-level `alignment_config` for global alignment.

```yaml
model_config:
  use_chunking: true
  window_size: 500
  window_overlap: 50

alignment_config:
  window_alignment_method: pose_overlap  # pose_overlap | icp | none
  min_overlap_frames: 10
  allow_scale: auto      # auto | true | false
  pose_overlap_max_rmse: 2.0
  icp_voxel_size: 0.1
  icp_max_corr_dist: 0.2
  icp_init: none         # none | ransac
  gps_scale_min_std_dev_m: 0.5
  save_alignment_debug: true
```

Notes:
- `allow_scale: auto` means enable Sim(3) for non-metric outputs and SE(3) for
  metric outputs. This relies on `PointCloud.is_metric` and/or model metadata.
- For GPS alignment, metric models should skip scale adjustment when GPS
  variance is low; record GPS variance metrics for later analysis.
- `max_frames` becomes a hard cap, not the window size. Windowing uses
  `window_size` and `window_overlap` when `use_chunking` is true.

## Pipeline Changes

### 1) Model-level Chunking

Each model decides how to chunk images. Suggested behavior:

- If `use_chunking` is false or `window_size` is null, run as today.
- If `use_chunking` is true:
  - Build sliding windows of frame paths: size `window_size`, step
    `window_size - window_overlap`.
  - For each window, run the model on that subset and collect a
    `ReconstructionResult`.
  - Attach window metadata: `window_id`, `frame_start`, `frame_end`,
    `frame_indices`, `frame_paths`, `overlap_with_prev`.

This requires the model to return **multiple** results. Proposed type change:

- Extend `ReconstructionResult` with `chunks: Optional[list[ReconstructionResult]]`
  and `window_metadata: Optional[dict]`.
- The top-level result should carry summary metadata and maybe the first
  chunk’s pointcloud if a single object is still required by callers.

### 2) Runner-level Alignment and Merge

`ExperimentRunner._process_video` should detect `result.chunks` and, when
present, align and merge them:

1. Build overlap correspondences between consecutive windows using
   `frame_indices` in `window_metadata`.
2. If poses exist for both windows, compute a Sim(3) (or SE(3)) transform using
   Kabsch/Umeyama over overlapping camera positions.
3. Compose transforms into a global frame, apply to each chunk’s point cloud
   and (optionally) to its poses.
4. Concatenate transformed point clouds (optionally voxel downsample).
5. Run `GPSAligner` on merged output if enabled and telemetry is valid.
6. Compute metrics on the merged output only.

Terminology:
- **Map frame**: the coordinate system of window 0 (arbitrary scale/rotation).
- **ENU frame**: the GPS-aligned local East-North-Up frame anchored at the first
  GPS sample; this is applied after merging to orient the map in real-world
  coordinates.

### 3) Pose-Overlap Alignment Details

Inputs:
- `CameraPoses.poses` shape `(M, 4, 4)` expected to be camera-to-world.
- Overlap frame indices `I` between window i and i+1.

Steps:
- Extract camera positions `Pi` and `Pj` for overlapping indices.
- If `allow_scale` is true, solve Sim(3) using Umeyama (scale+R+t).
- If `allow_scale` is false, solve SE(3) using Kabsch (R+t).
- Reject alignment if:
  - overlap count < `min_overlap_frames`
  - RMS error > `pose_overlap_max_rmse`
  - non-finite values detected
- Fallback to ICP if alignment is rejected and `window_alignment_method=icp`.

### 4) GPS/IMU Alignment

Use existing `GPSAligner` on the merged point cloud and merged poses.
Guardrails already exist:

- Minimum GPS motion length and standard deviation.
- Non-finite checks.

Recommended update:
- Allow GPS alignment to skip scale changes for metric models
  (`allow_scale=false` in `GPSAligner` or a new flag in `alignment_config`).

## Data Structure Updates (Proposed)

In `src/core/types.py`:

- Add optional chunk support:
  - `ReconstructionResult.chunks: Optional[list[ReconstructionResult]]`
  - `ReconstructionResult.window_metadata: Optional[dict]`
- Add optional `frame_indices` or `frame_paths` to `CameraPoses` or store them
  in `window_metadata` to allow alignment by frame overlap.

## Output Artifacts

Per video:

- `outputs/<video>/windows/window_000/*` (model outputs per window).
- `outputs/<video>/aligned_pointcloud.ply` (merged, GPS-aligned point cloud).
- `outputs/<video>/metadata.json` (updated to reflect merged stats).
- Alignment details live in `results.json` via `model_metadata` by default;
  optional per-video files can be added later if metadata becomes too large.

## Edge Cases and Fallbacks

- If a model does not output poses, use ICP for alignment (if enabled).
- If overlap frames are missing or too few, fall back to ICP or skip alignment.
- If GPS is missing or low quality, skip GPS alignment and keep local frame.
- If `max_frames` is set and `use_chunking` is false, behavior remains
  truncation as today.

## Testing Plan

- Unit test window slicing and overlap index calculation.
- Unit test pose inversion (VGGT extrinsic -> c2w).
- Unit test Sim(3) alignment on synthetic data (known transform).
- End-to-end smoke test on a short video with `use_chunking=true`.

## Open Questions

- Should `ReconstructionResult` be expanded or should we add a separate
  `WindowedReconstructionResult` type?
- Where should alignment outputs (transforms) live: metadata vs separate file?
- For metric models with GPS, should we allow scale changes only if GPS
  variance is low?
