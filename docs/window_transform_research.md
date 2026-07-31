# Adjacent-window transform research

Status: research and implementation recommendation
Last updated: 2026-07-29

## Decision summary

Mapper should not treat whole-cloud ICP as the primary way to join adjacent
reconstruction windows. Adjacent windows deliberately contain the **same source
frames**, so they offer much stronger correspondences than nearest-neighbor
geometry:

1. Preserve the per-frame point maps and confidences for overlap frames.
2. Pair the 3D values at the same valid pixels in the same source frame.
3. Estimate a confidence-weighted, robust SE(3) or Sim(3) transform.
4. Refine that initializer with coarse-to-fine robust point-to-plane ICP or
   generalized ICP on overlap geometry only.
5. Validate the transform on held-out correspondences and emit a relative edge
   with an information matrix. Do not silently substitute identity on failure.
6. Compose those edges only to initialize the map. Optimize them later in a
   pose graph with loop, GPS, gravity, and cross-session constraints.

This is close to the adjacent-chunk method used by VGGT-Long and DA3-Streaming.
Both align confidence-filtered point-map predictions for matching pixels in
overlap frames, use robust weighted SE(3)/Sim(3), and then apply loop correction.
VGGT-Long's current implementation can use Huber IRLS, while DA3-Streaming also
offers a depth-scale estimate followed by SE(3).

The current implementation is not quite “vanilla ICP only.” It first aligns
overlapping **camera centers** with an unweighted SE(3)/Sim(3) fit and falls back
to single-scale point-to-plane ICP. That is a reasonable prototype, but it is
not the way a professional visual SLAM system normally forms and maintains
motion constraints.

## What Mapper does today

The active implementation is
[`src/alignment/window_aligner.py`](../src/alignment/window_aligner.py).

### Pairwise path

For each adjacent pair, it:

1. Finds shared `frame_indices`, or assumes that the last and first
   `min_overlap_frames` correspond.
2. Extracts only the camera **positions**, not camera orientations or image/3D
   correspondences.
3. Fits an unweighted closed-form SE(3) or Sim(3) transform.
4. Accepts the result if its camera-center RMSE is finite and below one absolute
   threshold.
5. If that fails, downsamples both complete window point clouds and runs
   single-scale point-to-plane ICP. Its default initial transform is identity;
   FPFH/RANSAC is optional.
6. If both paths fail, records `method: none` and applies identity.

Each new transform is solved directly into the already accumulated global frame.
The output is a chain of absolute window transforms, not a set of reusable
relative constraints with uncertainty.

### What is already good

- Exact frame indices are used when available.
- SE(3) is selected for outputs declared metric and Sim(3) for nonmetric output.
- Pose-overlap is preferred to uninitialized ICP.
- Source windows remain separate during publication.
- The applied scale, rotation, and translation are retained as provenance.
- The FPFH/RANSAC initializer exists and can be enabled.

### Important weaknesses

| Area | Current behavior | Why it matters on hiking video |
| --- | --- | --- |
| Pose data | Fits camera centers only | A nearly straight hiking trajectory is close to collinear. Rotation around the direction of travel is weakly observable from centers even though every pose contains an orientation. |
| Robustness | Ordinary least squares | One bad pose, duplicated pose, or locally inconsistent scale can move the entire transform. |
| Scale | Independent Sim(3) on every nonmetric edge | Per-edge scale noise becomes multiplicative “scale breathing” along a long route. |
| Threshold | One `2.0` RMSE in model units | It has no stable meaning for arbitrary-scale output and can be too permissive for a small window or too strict for a large one. |
| ICP scope | Entire source and target windows | Non-overlap geometry, sky, hikers, vegetation motion, and duplicate surfaces all enter correspondence search even though overlap-frame identity is known. |
| ICP initializer | Identity by default | ICP is local and may converge to a plausible wrong surface or return a low-information solution. |
| ICP schedule | One voxel size and one correspondence radius | It sacrifices either capture range or final accuracy and is not scale-adaptive. |
| ICP loss | Unweighted point-to-plane least squares | Dynamic points and erroneous learned depth receive full influence. |
| Acceptance | ICP RMSE alone is returned and always accepted | Fitness, inlier count, information/conditioning, scale, transform magnitude, and validation-set error are not checked. A small RMSE can describe a tiny accidental overlap. |
| Composition | Solves current window directly into the accumulated global frame | A bad edge immediately contaminates every later window, and the relative measurement cannot be reoptimized cleanly. |
| Failure | Identity transform | This can publish an arbitrary local window frame as though it were aligned. Failure should stop or quarantine the disconnected component. |
| Metadata | No covariance/information or rejection reasons | A graph backend cannot weight the constraint, and experiments cannot distinguish weak geometry from bad initialization. |

The fallback without exact frame indices also uses exactly
`min_overlap_frames`, rather than proving which frames correspond. This should
be a legacy-only path: normalization and keyframe selection should make exact
source frame IDs mandatory.

There is also a concrete configuration mismatch:
[`configs/experiments/vggt_test.yaml`](../configs/experiments/vggt_test.yaml)
sets `min_overlap_frames: 1`, while `_align_sim3` rejects fewer than two
positions. Even two center correspondences do not fully constrain a general 3D
rotation. The gate should be based on geometric rank/excitation, not merely a
small count.

### Test coverage

[`tests/test_window_alignment_streaming.py`](../tests/test_window_alignment_streaming.py)
only exercises `window_alignment_method: none`, source-unit preservation,
compatibility merging, and metric aggregation. It does not test:

- recovery of a known SE(3) or Sim(3);
- exact overlap-index pairing;
- pose convention handling;
- outliers, collinear motion, or scale degeneracy;
- ICP initialization or acceptance;
- transform composition;
- a rejected/disconnected window;
- uncertainty or alignment metadata.

[`src/aligner.py`](../src/aligner.py) contains older standalone sequential,
point-to-plane, and colored-ICP experiments. It is not used by the experiment
runner and does not add pose-graph correction or validation.

## Does this match professional SLAM?

Only partially.

Professional SLAM usually separates a **frontend** that creates local
measurements from a **backend** that jointly estimates poses:

```text
frames / IMU
     |
     v
tracking + data association
     |
     +--> local keyframe/submap optimization
     |         |
     |         +--> relative pose + information
     v
place recognition --> geometrically verified loop edges
                           |
                           v
               pose/factor graph optimization
                           |
                           v
                  globally corrected map
```

Visual SLAM normally estimates motion from tracked image features, dense
reprojection/flow, or learned point-map matches and optimizes several keyframes
together. ORB-SLAM3 has distinct tracking, local mapping, loop closing, and
multi-map components; it uses place recognition and nonlinear optimization
rather than dense cloud ICP as its visual odometry primitive.

LiDAR and RGB-D systems do commonly use ICP-like registration, but normally:

- against a local submap rather than only the previous raw scan;
- initialized by a motion model, IMU, or another registration frontend;
- with multiple resolutions and correspondence rejection;
- converted into a constraint with an information matrix;
- followed by pose-graph/factor-graph optimization.

Open3D's own multiway-registration design distinguishes reliable adjacent
odometry edges from uncertain loop edges, computes information matrices, and
optimizes a pose graph. Cartographer similarly performs local scan-to-submap
matching and adds globally searched scan-to-submap loop constraints. LIO-SAM
uses IMU initialization, scan-to-local-map registration, selective keyframes,
and a factor graph.

The closest professional analogue to Mapper's foundation-model windows is
VGGT-Long/DA3-Streaming, not classic scan-to-scan ICP. Those systems exploit
pixel identity in overlap frames for robust point-map alignment and add a
lightweight loop optimizer. MASt3R-SLAM goes further: it tracks against a
keyframe using learned point-map matching, fuses local point maps, verifies
retrieval candidates, adds graph edges, and performs global second-order
optimization.

## Adjacent-window methods worth considering

### 1. Same-frame, same-pixel point-map alignment — recommended primary

For an overlap source frame `f` and valid pixel `u`, window `i` predicts a 3D
point `P_i(f,u)` and window `j` predicts `P_j(f,u)`. The source frame and pixel
already establish the correspondence:

```text
P_i(f,u)  <-------------------------->  P_j(f,u)
      same decoded frame, same normalized pixel
```

Use confidence, finite-depth, sky, border, and dynamic-object masks, then solve:

```text
argmin_(s,R,t) sum_k w_k rho(
    || P_i(k) - (s R P_j(k) + t) || / sigma_k
)
```

Use `s = 1` for validated metric models. For nonmetric models, solve Sim(3) or
estimate relative depth scale robustly and then solve SE(3). Huber or Cauchy IRLS
is a straightforward baseline. Spatially stratified sampling prevents a dense
patch of nearby trail or distant vegetation from dominating the fit.

Advantages:

- no nearest-neighbor correspondence ambiguity;
- millions of potential correspondences across several views;
- confidence and semantic masks remain attached;
- directly follows VGGT-Long and DA3-Streaming;
- works even when the overlap camera centers have little translational
  excitation.

Required repository contract change: preserve per-frame point maps (or depth,
intrinsics, and poses sufficient to regenerate them), confidence, normalized
pixel geometry, and exact source frame IDs until alignment is complete. A
flattened `PointCloud` cannot express these correspondences.

### 2. Full-pose robust alignment — recommended inexpensive fallback

When only camera poses are available, use their rotations as well as their
centers. Estimate a consensus relative rotation from corresponding orientations,
then estimate scale and translation from centers. Refine all terms with a robust
pose residual.

This is much better conditioned than center-only fitting on a mostly straight
trail. It is still limited by correlated model errors: the two window pose sets
come from the same model and may each be internally smooth but geometrically
wrong.

Validation must detect poor translational excitation. A minimum frame count is
not enough; inspect the singular values of the centered camera trajectory and
the spatial/angular baseline.

### 3. Learned image or point-map correspondences

If capture normalization changes pixels between windows, or if two submaps do
not share identical frames, use learned descriptors/matches from the
reconstruction model and solve a robust relative pose or Sim(3). MASt3R-SLAM is
the relevant control implementation: it uses point-map matching for tracking
and geometrically verifies retrieval candidates before adding graph edges.

This is essential for loop and cross-session edges. It is unnecessarily
expensive for ordinary adjacent windows when same-pixel correspondences are
available.

### 4. Global point-cloud registration as a recovery initializer

FPFH + RANSAC, Fast Global Registration, or TEASER++ can recover a coarse
transform when a trustworthy pose/visual initializer is unavailable.
TEASER++ is robust to extreme correspondence outlier rates, but it still
depends on the quality and observability of the supplied 3D correspondences.

These methods should run on overlap-only, downsampled, masked geometry. They
should not be the normal adjacent-window path because they discard known image
correspondence and are more expensive and ambiguous in repetitive vegetation.

### 5. Robust coarse-to-fine ICP/GICP — recommended refinement

After a good initializer:

- crop to overlap-frame geometry;
- normalize thresholds from model scale or median scene depth;
- run coarse-to-fine voxel levels and decreasing correspondence radii;
- use point-to-plane ICP or GICP with Huber/Tukey/Cauchy loss;
- reject incompatible normals and optionally use reciprocal
  correspondences;
- retain fitness, inlier count, RMSE, and the information/Hessian matrix.

Open3D explicitly recommends multi-scale ICP over single-scale ICP for large
clouds and documents robust kernels for downweighting outliers.

Colored ICP can help where geometry is weak, but GoPro auto-exposure,
white-balance changes, shadows, and moving foliage violate photometric
consistency. It should be an optional ablation, not the default.

### 6. Local bundle adjustment / joint overlap refinement

Optimize the poses and shared geometry for all overlap keyframes together,
rather than reducing the overlap immediately to one rigid transform. Useful
residuals include reprojection/ray consistency, learned point-map agreement,
relative-pose priors, and gravity.

This most closely resembles professional visual SLAM, but is a larger change.
MASt3R-SLAM or another complete SLAM adapter should be the benchmark before
building a custom local optimizer.

## Recommended local transform pipeline

### Stage A: construct exact overlap data

Require:

- source video checksum;
- exact decoded source frame index and timestamp;
- capture-normalization profile/hash;
- normalized image size/crop;
- per-frame point map or depth + intrinsics + camera pose;
- per-pixel confidence;
- sky/dynamic/invalid masks where available;
- camera pose convention and coordinate-frame declaration.

Do not align frames that merely occupy the same ordinal slot if their exact
source identity is unknown.

### Stage B: filter and sample correspondences

1. Intersect valid masks from both windows.
2. Reject non-finite and non-positive depth.
3. Reject sky and known dynamic regions.
4. Remove low confidence with model-specific calibrated thresholds.
5. Apply a depth-discontinuity/border mask.
6. Sample across frames, image tiles, depth bands, and viewing directions.
7. Reserve a deterministic held-out subset for validation.

Sampling must be seeded and recorded so cloud reruns are reproducible.

### Stage C: estimate a robust initializer

Priority:

1. confidence-weighted same-pixel point-map SE(3)/Sim(3);
2. full-pose robust SE(3)/Sim(3);
3. learned visual/point-map matching with RANSAC;
4. FPFH/FGR/TEASER++ global 3D registration.

For claimed metric models, default to SE(3). Allow Sim(3) only when a metric
scale validation explicitly fails, and record that the metric claim was
overridden.

For nonmetric models, compare:

- direct robust Sim(3);
- robust depth-scale estimate followed by SE(3), as exposed by
  DA3-Streaming.

The second option can reduce coupling between scale and rotation. Neither
should permit unconstrained scale without bounds and diagnostics.

### Stage D: geometric refinement

Run robust multi-scale point-to-plane ICP or GICP on overlap-only geometry,
initialized by Stage C. Keep the initializer if refinement fails validation or
makes the held-out score worse.

### Stage E: fail-closed validation

Every accepted transform should pass:

- finite values and a proper rotation (`det(R)` near `+1`);
- sufficient valid correspondence count across multiple frames;
- sufficient image-space coverage, depth spread, and viewpoint/trajectory
  excitation;
- configured scale bounds and scale consistency with neighboring edges;
- minimum inlier ratio and overlap fitness;
- robust train residual and held-out residual;
- forward/reverse transform consistency;
- improvement over the initializer for an ICP refinement;
- acceptable information-matrix eigenvalues/condition number;
- a maximum rotation/translation rate compatible with the overlap duration,
  unless a stronger initializer explains it.

Thresholds should be normalized by estimated noise and scene scale. Store raw
metrics as well as the pass/fail decision; tune gates from successful and failed
trail captures rather than hard-coding a single universal RMSE.

On failure:

- retry a different initializer/refiner;
- increase overlap only in a subsequent reconstruction experiment;
- otherwise mark the new window as a disconnected component and stop canonical
  publication for that component.

Never convert failure to identity while labeling the result aligned.

### Stage F: emit a relative constraint

Produce a native-frame edge, not only an absolute transform:

```yaml
from_window: 12
to_window: 13
group: sim3                 # se3 | sim3
measurement: ...
information: ...
initializer: pointmap_irls
refiner: multiscale_point_to_plane
status: accepted
metrics:
  overlap_frames: 30
  candidate_correspondences: ...
  inliers: ...
  fitness: ...
  train_rmse_normalized: ...
  heldout_rmse_normalized: ...
  forward_reverse_error: ...
  scale: ...
  condition_number: ...
```

Compose accepted edges to get initial window poses, but retain native
measurements and uncertainty for graph optimization.

## Local registration versus pose-graph optimization

These are separate problems and should have separate APIs and test suites.

### Local registration answers

> Given two windows believed to overlap, what relative transform best explains
> their shared observations, and how trustworthy is it?

It owns correspondence, robust estimation, local refinement, validation, and
information estimation. Better local registration reduces drift but cannot
remove accumulated bias by itself.

### Pose-graph optimization answers

> Given many relative and absolute constraints, what set of window poses best
> satisfies them jointly?

Each window/submap is a node. Adjacent accepted transforms are odometry edges.
Geometrically verified revisits are loop edges. GPS, gravity, and cross-session
matches add other factors. Robust losses or switchable/line-process constraints
limit damage from false loops.

A pure adjacent chain has no redundancy: every accepted error must propagate.
Graph optimization becomes useful when it has additional edges or absolute
factors. It should not be presented as a substitute for validating local
registration.

For nonmetric reconstruction, optimize Sim(3) node poses or explicit log-scale
variables with regularization. For validated metric reconstruction, keep an
SE(3) graph and attach GPS/gravity factors. Avoid silently mixing SE(3) and
Sim(3) constraints.

## Implementation sequence

No implementation changes were made as part of this research task.

### Immediate, bounded improvements

These retain the current sequential publisher:

1. Add synthetic tests for the existing pose-overlap solver and composition.
2. Fail closed instead of accepting identity after an alignment failure.
3. Log ICP fitness, correspondence count, initializer, transform magnitude, and
   rejection reason.
4. Use exact frame-index intersections only.
5. Add trajectory excitation and scale bounds to pose-overlap validation.
6. Use camera orientations in a robust full-pose fallback.
7. Replace single-scale ICP with initialized, robust multi-scale ICP on
   overlap-only geometry.

Items 2 and 4 change failure behavior and should be approved before
implementation because existing runs that currently “complete” may instead
stop as invalid.

### Required adapter/output-contract improvement

Preserve overlap-frame point maps/depth, confidence, masks, intrinsics, and
source-frame identity. Then add confidence-weighted same-pixel robust
SE(3)/Sim(3) as the primary estimator. This is the highest-value adjacent-window
change.

The user should approve whether these arrays live:

- directly in a richer source-unit object;
- as bounded temporary artifacts referenced by the source unit; or
- in a model-specific alignment callback.

Temporary artifact references best preserve the existing bounded-memory
publisher.

### Larger backend work

Add a window/submap constraint type and a GTSAM or equivalent pose-graph
backend. Ingest adjacent, loop, GPS, gravity, and cross-session edges; optimize
before dense refusion/publication. This is architecture work, not an “ICP
upgrade.”

## Validation experiment

Run all methods on identical normalized keyframes and windows:

| ID | Initializer | Refinement | Backend |
| --- | --- | --- | --- |
| A0 | Current camera centers | Current ICP fallback | Chain |
| A1 | Robust full poses | None | Chain |
| A2 | Same-pixel point maps | None | Chain |
| A3 | Same-pixel point maps | Robust multi-scale ICP/GICP | Chain |
| A4 | Same as A3 | Same | Pose graph, adjacent edges only |
| A5 | Same as A3 | Same | Pose graph + verified loops |

Use at least:

- a short sequence with an independently checkable transform;
- Kings Canyon without usable GPS;
- Tahoe with distributed GPS retained only for evaluation;
- a loop/return route;
- one intentionally difficult seam with blur, vegetation motion, or little
  translation.

Report per-edge:

- acceptance and failure reason;
- runtime and peak memory;
- overlap frames, inliers, coverage, and fitness;
- train and held-out normalized residual;
- rotation/translation/scale error when reference exists;
- information eigenvalues and condition number;
- forward/reverse consistency.

Report end-to-end:

- relative pose error;
- aligned trajectory error and drift percentage;
- loop endpoint error before GPS;
- adjacent seam distance;
- repeated-route separation;
- number and length of disconnected components.

Expected interpretation:

- A2 versus A0 measures the value of known point-map correspondence.
- A3 versus A2 measures geometric refinement.
- A4 should be nearly identical to A3 without redundant constraints; a large
  difference indicates weighting or convention problems.
- A5 measures the actual value of global graph correction.

## Primary sources

- [VGGT-Long official repository](https://github.com/DengKaiCQ/VGGT-Long) and
  [alignment implementation](https://github.com/DengKaiCQ/VGGT-Long/blob/main/vggt_long.py):
  overlap point maps, confidence filtering, robust weighted SE(3)/Sim(3), and
  loop optimization.
- [VGGT-Long paper](https://arxiv.org/abs/2507.16443): kilometer-scale chunking,
  overlap alignment, and lightweight loop correction.
- [DA3-Streaming official documentation](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md)
  and
  [implementation](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/da3_streaming.py):
  same-pixel point-map alignment, optional depth-scale + SE(3), and loop
  optimization.
- [MASt3R-SLAM, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html)
  and [official repository](https://github.com/rmurai0610/MASt3R-SLAM):
  point-map tracking, local fusion, graph construction, loop closure, and
  global optimization.
- [Open3D multi-scale ICP](https://www.open3d.org/docs/latest/tutorial/t_pipelines/t_icp_registration.html):
  coarse-to-fine registration, robust kernels, metrics, and information.
- [Open3D multiway registration](https://www.open3d.org/docs/latest/tutorial/pipelines/multiway_registration.html):
  odometry versus uncertain loop edges, information matrices, and pose-graph
  optimization.
- [Open3D global registration](https://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html):
  RANSAC/FGR initialization before local ICP.
- [TEASER++ official repository](https://github.com/MIT-SPARK/TEASER-plusplus):
  certifiably robust registration from outlier-contaminated correspondences.
- [ORB-SLAM3 official repository](https://github.com/UZ-SLAMLab/ORB_SLAM3):
  tracking, local mapping, loop closing, and multi-map SLAM.
- [Cartographer loop-closure paper](https://research.google/pubs/real-time-loop-closure-in-2d-lidar-slam/):
  local submaps and globally searched scan-to-submap constraints.
- [LIO-SAM paper](https://arxiv.org/abs/2007.00258):
  IMU-initialized scan-to-local-map registration, keyframes, and factor-graph
  smoothing.
- [GTSAM official repository](https://github.com/borglab/gtsam): factor-graph
  smoothing and mapping backend.
