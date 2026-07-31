# Reconstruction follow-up task list

Status: **complete**  
Started: 2026-07-29

## 1. Adjacent-window transforms

- [x] Audit the current window alignment implementation and its tests.
- [x] Research stronger adjacent-window registration methods and how production
  SLAM systems create and optimize local-map constraints.
- [x] Separate immediately implementable improvements from larger pose-graph
  work.
- [x] Document a recommended transform-estimation and validation pipeline.

## 2. Reconstruction adapters

- [x] Audit every registered adapter, dependency pin, capability declaration,
  pose convention, scale claim, and output contract.
- [x] Harden the DA3-Streaming adapter.
- [x] Add a VGGT-Long adapter and model configuration.
- [x] Add a MASt3R-SLAM adapter and model configuration.
- [x] Add a VGGT-Ω adapter and model configuration.
- [x] Correct or explicitly disable misleading behavior in the current
  `orb_slam` adapter.
- [x] Preserve the repository's existing adapter format and defer optional
  framework improvements for user approval.
- [x] Add focused unit tests for adapters that do not require model weights or
  GPUs.

## 3. Capture normalization and keyframes

- [x] Audit `src/preprocessing/video_processor.py` and telemetry ingestion.
- [x] Load telemetry before keyframe selection while preserving exact source
  frame indices and timestamps.
- [x] Implement deterministic capture normalization metadata.
- [x] Implement configurable fixed-rate and quality/motion-aware keyframe
  selection.
- [x] Persist keyframe selection reasons and relevant telemetry/calibration
  metadata.
- [x] Add focused preprocessing tests.

## 4. Dataset upload and processing cost

- [x] Inventory the current local video dataset without modifying it.
- [x] Estimate upload size, storage cost, transfer cost, and compute cost for
  the recommended model experiments.
- [x] Compare upload-on-demand, persistent originals, and experiment-cache
  strategies.
- [x] Document assumptions and a recommended storage lifecycle.

## 5. Integration and validation

- [x] Review subagent changes together and resolve contract mismatches.
- [x] Run formatting, type/static checks, and targeted tests.
- [x] Update this checklist with completed items and remaining blockers.
- [x] Summarize implemented changes, research conclusions, cost estimate, and
  user decisions still needed.

## Deferred decisions requiring approval

The requested work is complete. The following adjacent improvements were
identified but intentionally not implemented:

- rename the misleading public `orb_slam` registry key to `opencv_orb_vo` with
  a migration path;
- add a durable submap/depth/confidence result contract needed for same-pixel
  overlap alignment;
- change window-alignment failure from silent identity placement to fail-closed
  or quarantined disconnected components;
- consolidate external process/revision/output parsing and optionally bootstrap
  isolated model environments;
- add exact embedded-PTS plumbing for unusual variable-frame-rate captures,
  richer GPMF camera-setting extraction, and model-feedback keyframe selection.

## 6. Scale and model setup follow-up

- [x] Verify metric-scale claims for DA3, VGGT, VGGT-Long, MUSt3R,
  MASt3R-SLAM, and VGGT-Omega.
- [x] Explain why camera intrinsics improve reconstruction but do not resolve
  monocular absolute scale by themselves.
- [x] Add a canonical ignored `weights/` layout and update repository configs
  and examples to use it.
- [x] Add a top-level checkpoint dispatcher and individual model download
  scripts with dry-run support.
- [x] Document richer GoPro GPMF camera settings.
- [x] Expand the proposed model-feedback keyframe and exact VFR PTS designs.
