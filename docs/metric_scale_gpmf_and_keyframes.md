# Metric scale, GoPro metadata, and model-aware keyframes

## Which reconstruction outputs are metric?

| Pipeline | Metric from RGB alone? | Mapper status | Why |
| --- | --- | --- | --- |
| DA3 Main / ordinary DA3-Streaming | No | Relative | The any-view checkpoints predict consistent geometry, but do not claim absolute scale. Streaming's default Sim(3) alignment is also free to change scale per chunk. |
| DA3Metric-Large | Yes, learned monocular metric depth | Not integrated | The documented conversion is `depth_m = focal_px * network_output / 300`. This formula is specific to this trained checkpoint. |
| DA3Nested-Giant-Large-1.1 | Yes, learned metric visual geometry | Downloaded for DA3-Streaming, but conservatively marked relative after streaming fusion | The nested model combines any-view geometry with a metric estimator and returns depth in meters. To preserve that scale through a long sequence, chunk alignment must be SE(3), not unconstrained Sim(3), and must be validated against known distances. |
| VGGT | No published metric guarantee | Relative | It predicts intrinsics, extrinsics, depth, and point maps in one internally consistent reconstruction gauge. |
| VGGT-Long with VGGT backend | No | Relative | It extends a relative base model using Sim(3) chunk alignment and loop correction. |
| VGGT-Long with MapAnything (`Map-Long`) | Yes, subject to the base model's metric accuracy | Not integrated | VGGT-Long explicitly added SE(3) alignment for metric base models and reports Map-Long as metric/real-scale. |
| MUSt3R | No published absolute-scale guarantee | Relative | Multi-view consistency and camera prediction do not remove monocular gauge freedom. |
| MASt3R / MASt3R-SLAM | No reliable absolute-scale guarantee from a monocular GoPro run | Relative | The filename contains `metric`, but its published training configuration uses scale normalization and the SLAM system is monocular. Known calibration improves geometry and tracking, but is not a physical baseline. |
| VGGT-Omega | No published metric guarantee | Relative | It improves camera/depth reconstruction in static and dynamic scenes, but does not claim world units. |

Primary references: [DA3 model zoo and metric-depth
formula](https://github.com/ByteDance-Seed/Depth-Anything-3#-model-cards),
[VGGT outputs](https://github.com/facebookresearch/vggt#overview),
[VGGT-Long metric-base-model support](https://github.com/DengKaiCQ/VGGT-Long#updates),
[MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM),
and [VGGT-Omega](https://arxiv.org/abs/2605.15195).

### Why the camera matrix is not enough

The intrinsic matrix

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

converts an image pixel into a camera ray. It fixes projection geometry and
removes focal-length ambiguity, but it contains no physical translation or
known length. Multiplying every camera translation and every scene point by the
same positive scalar creates the same monocular images. Therefore a calibrated
`K` cannot, by itself, resolve scale.

DA3Metric is not an exception to that geometry: it learned a statistical prior
for real-world depth during training and uses focal length to normalize that
learned prediction. Applying its `/ 300` formula to VGGT, MUSt3R, or
MASt3R-SLAM output would be invalid.

Absolute scale requires at least one scale-bearing observation:

- a metric-depth model such as DA3Nested or MapAnything;
- stereo/multi-camera images with a measured baseline;
- visual-inertial estimation with synchronized, calibrated IMU measurements and
  estimated biases/gravity;
- one or more known distances, camera heights, or surveyed control points; or
- GPS positions sufficiently separated and accurate to estimate a robust
  trajectory Sim(3).

GoPro calibration is still valuable. Use the real per-mode intrinsics and
distortion profile to improve ray geometry, reprojection residuals, and pose
stability. A profile must match resolution, lens/FOV mode, crop, orientation,
and stabilization state. HyperSmooth can apply a time-varying electronic crop,
so a factory nominal focal length is not automatically the effective
per-frame `K`.

Recent GoPro metadata does publish more than a generic FOV label: the `FOVL`
device can contain diagonal FOV (`ZFOV`), visual FOV mode (`VFOV`), radial
polynomial coefficients (`POLY`/`PYCF`), zoom normalization (`ZMPL`), and
SuperView/HyperView warp coefficients (`MAPX`, `MAPY`). These describe a
pixel-to-ray/lens-warp model, not necessarily a ready-made OpenCV `K` plus
distortion vector. We can derive or fit an effective calibration from them,
but should validate principal point and stabilization crop against a checkerboard
or feature tracks for each capture mode. See GoPro's [official FOVL
description](https://github.com/gopro/gpmf-parser#dvid-fovl-large-fov---lens-distortion).

### Recommended metric experiment

1. Run the downloaded DA3 nested checkpoint on short surveyed clips.
2. Preserve its scale with SE(3) adjacent-chunk alignment.
3. Compare reconstructed camera travel and point-to-point distances against GPS
   baselines and a few tape/laser measurements.
4. Keep a Sim(3) fallback when the metric prior fails, but record the recovered
   scale and residual rather than silently calling the result metric.
5. For kilometer sequences, benchmark Map-Long as the next explicitly metric
   long-sequence baseline.

## Richer GPMF camera settings

The current telemetry path mainly consumes GPS, accelerometer, gyroscope,
gravity, and orientation. GoPro's GPMF track can also expose per-frame imaging
conditions and capture-wide settings. Availability varies by camera and
firmware, so ingestion should inventory keys first and preserve unknown keys.

Useful per-frame or periodic signals include:

| FourCC | Meaning | Reconstruction use |
| --- | --- | --- |
| `SHUT` | Exposure time in seconds | Predict motion blur; reject or downweight long-exposure frames. |
| `ISOG` / `ISOE` | Sensor gain / ISO | Detect noisy low-light frames and sudden exposure changes. |
| `WBAL` / `WRGB` | White balance temperature / channel gains | Identify appearance discontinuities that can hurt matching. |
| `ALLD` | Auto-low-light extended frame duration | Detect effective frame cadence changes and blur risk. |
| `YAVG`, `HUES`, `UNIF`, `SCEN` | Luma and simple scene statistics | Cheap exposure/texture/scene-change priors. |
| `SROT` | Sensor readout time | Rolling-shutter correction or uncertainty inflation during fast rotation. |
| `CORI`, `IORI`, `GRAV` | Camera/image orientation and gravity | Gravity alignment, orientation sanity checks, and stabilization diagnostics. |
| `MSKP` / `LSKP` | Encoded frame skips or duplicates | Exact video/telemetry synchronization and VFR diagnostics. |
| `GPS5` / `GPS9`, `GPSF`, `GPSP`, `GPSU` | Position, fix, precision, and UTC | Quality-gated georeferencing and clock alignment. |

Capture-wide header fields worth persisting include camera model and firmware
(`MINF`, `FMWR`), capture identity (`CPID`, `CPIN`), HDR (`HDRV`), orientation
(`OREN`), digital zoom (`DZOM`, `DZST`), Protune settings (`PRTN`, `PTWB`,
`PTSH`, `PTCL`, `EXPT`, `PIMN`, `PIMX`, `PTEV`), and stabilization state
(`EISE`, `EISA`). GoPro documents both the per-frame streams and these header
settings in its [official GPMF parser
reference](https://github.com/gopro/gpmf-parser#where-to-find-gpmf-data).

The practical next schema should contain:

- a raw-key inventory with FourCC, stream name, units, count, and time range;
- normalized per-frame fields for shutter, ISO, white balance, readout time,
  stabilization, and skipped-frame count;
- the original raw values for auditability;
- explicit timebase and clock-domain metadata for every stream; and
- a calibration-profile lookup key derived from camera model, dimensions,
  lens/FOV mode, stabilization, crop, and firmware.

## Model-feedback keyframe selection

The current `quality_motion` selector is a useful first pass: it gates blur,
exposure, clipping, elapsed time, sparse optical-flow motion, and IMU angular
speed. It cannot know whether a candidate actually helps the reconstruction.
Model feedback adds that missing geometric information.

A practical two-stage selector is:

1. **Cheap candidate gate:** decode at a bounded rate (for example 10 Hz), reject
   severe blur/exposure failures using image metrics plus `SHUT`/ISO, and enforce
   a minimum time interval.
2. **Geometry preview:** run the model's inexpensive tracking/encoding path
   against the active keyframe set.
3. **Keep a frame when it adds information:** sufficient translational
   parallax, new visible area, improved track coverage, high depth/match
   confidence, or a strong nonlocal place-recognition/loop candidate.
4. **Reject or defer it when it is redundant or harmful:** mostly pure rotation,
   tiny baseline, low static-scene support, high dynamic-pixel fraction,
   inconsistent depth, or poor feature coverage.
5. **Safety constraints:** force a frame at the maximum time/distance gap, keep
   frames on either side of a detected cut or tracking loss, and use hysteresis
   so selection does not oscillate around one threshold.

A portable model-feedback record could look like:

```json
{
  "overlap_fraction": 0.62,
  "static_support_fraction": 0.91,
  "median_depth_confidence": 4.8,
  "median_parallax_px": 18.2,
  "translation_over_median_depth": 0.047,
  "coverage_gain_fraction": 0.13,
  "loop_candidate_score": 0.88,
  "tracking_inlier_fraction": 0.74,
  "decision": "selected",
  "reasons": ["coverage_gain", "translation_parallax"]
}
```

The adapter boundary matters:

- MASt3R-SLAM already owns tracking, keyframes, graph construction, and loop
  closure, so Mapper should give it a reasonably dense, quality-gated stream
  and import its selected keyframes rather than pre-pruning aggressively.
- DA3-Streaming and VGGT-Long need enough temporal overlap for chunk alignment.
  Preselection should retain continuity, while model confidence can tune the
  next sampling interval.
- VGGT and VGGT-Omega bounded windows can use predicted pose/depth confidence
  after each tentative insertion and discard redundant candidates before the
  final window inference.
- VGGT-Omega supports dynamic scenes, but that is not the same as guaranteed
  foreground removal. Dynamic/static support should be measured or masked
  explicitly before map fusion.

## Exact VFR PTS plumbing

The current code asks FFmpeg's `fps` filter for regularly sampled JPEGs, then
maps candidate ordinal `i` to target time `i / candidate_fps` and chooses the
nearest source timestamp reported by `ffprobe`. This is deterministic for
ordinary constant-frame-rate video, but it does not prove which source frame
FFmpeg actually selected. With variable frame rate, duplicate/drop behavior,
edit lists, discontinuities, or GoPro `MSKP`, the recorded source frame identity
can be wrong by one or more frames.

The robust design is to make timestamps a property of decoded frames, not an
after-the-fact nearest-neighbor guess:

1. Demux and decode frames sequentially with a library that exposes the decoded
   frame `pts` and stream `time_base` (PyAV is the straightforward option).
2. For every decoded frame, record source decode ordinal, integer PTS, timebase
   numerator/denominator, `timestamp_s = pts * time_base`, packet DTS when
   available, keyframe flag, and any skip/duplicate metadata.
3. Apply candidate cadence and keyframe scoring against those exact timestamps.
4. Write the JPEG and manifest row from that same decoded frame object.
5. Align GPMF samples using the MP4 metadata-track timebase and an explicit
   estimated clock offset; never assume telemetry sample ordinal equals video
   frame ordinal.
6. Preserve both source identity and selected-output ordinal through every
   adapter and trajectory export.

If adding PyAV is undesirable, an FFmpeg-only alternative is to compute exact
source-frame ordinals from `ffprobe -show_frames`, build a `select` expression,
extract with passthrough timestamps (`-vsync 0` / modern `-fps_mode passthrough`),
and parse `showinfo` for each emitted frame. It is workable, but more brittle
than keeping PTS attached to the decoded frame in-process.

Acceptance tests should include synthetic VFR files with irregular durations,
deliberate duplicate frames, non-zero start PTS, and a timestamp discontinuity.
The selected image hash, source ordinal, integer PTS, and seconds value should
all agree with an independent `ffprobe -show_frames` inventory.
