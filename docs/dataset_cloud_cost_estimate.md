# Dataset upload and reconstruction cost estimate

Status: **planning estimate**  
Date: 2026-07-29

## Scope and inventory

The checkout contains two different notions of the current dataset:

- `data/input_kings_canyon/` physically contains three GoPro originals totaling
  1,686,169,494 bytes, or **1.57 GiB**.
- The unique source-video metadata under `data/output_102425/*/metadata.json`
  describes 24 videos totaling 9,764,946,992 bytes, or **9.09 GiB**. Those
  videos represent 15,470 frames at the prior 10 FPS extraction rate, or about
  **25.8 minutes** of source footage.

Only the first quantity can be uploaded from this checkout today. The 9.09 GiB
quantity is the best recoverable estimate of the full source corpus; the other
21 originals must be located before a full upload.

The `data/` tree is about 398 GiB because it contains extracted frames and
reconstruction outputs. Those derived artifacts should not be treated as the
upload corpus. Originals, telemetry, small manifests, final packages, and
selected resumable state are durable; decoded frames and raw point/depth
intermediates are disposable.

## Upload cost and time

Ordinary object-storage ingress is free for the providers considered here.
Cloudflare R2 Standard currently includes 10 GB-month of storage per month and
free direct egress. The estimated 9.76 decimal GB source corpus fits just under
that free storage allowance.

| Corpus | Bytes | 10 Mbps upload | 20 Mbps upload | 100 Mbps upload | R2 Standard/month |
| --- | ---: | ---: | ---: | ---: | ---: |
| Originals physically present | 1.57 GiB | about 22 min | about 11 min | about 2.2 min | $0 within free tier |
| Full metadata-described corpus | 9.09 GiB | about 130 min | about 65 min | about 13 min | $0 within free tier |
| Existing 398 GiB derived tree, not recommended | about 398 GiB | about 95 h | about 47.5 h | about 9.5 h | roughly $6.25/month after free allowance |

Times are payload-only estimates and exclude connection variability,
multipart overhead, retries, and checksum verification.

## Compute assumptions

The source corpus contains 15,470 frames at 10 FPS. The proposed reconstruction
profiles contain approximately:

| Selection profile | Frames across corpus |
| --- | ---: |
| 1 FPS fixed-rate | 1,547 |
| 2 FPS fixed-rate | 3,094 |
| Existing 10 FPS extraction | 15,470 |

DA3-Streaming reports 8.51 FPS on an A100 for its published long-sequence
benchmark. VGGT-Long reports 2.91 FPS in the same DA3 comparison. Those rates
exclude or do not fully characterize video decoding, model cold start, place
retrieval, loop optimization, dense export, package publication, retries, and
GoPro-specific resolution/aspect-ratio effects. They are useful for an
inference floor, not a budget guarantee.

### One full-corpus run

| Model/profile | Inference GPU-hours | Modal A100 40 GB at $2.10/h | Modal A100 80 GB at $2.50/h | Runpod A100 80 GB Pod at $1.39/h |
| --- | ---: | ---: | ---: | ---: |
| DA3, 1 FPS | 0.05 | $0.11 | $0.13 | $0.07 |
| DA3, 2 FPS | 0.10 | $0.21 | $0.25 | $0.14 |
| DA3, 10 FPS | 0.51 | $1.06 | $1.26 | $0.70 |
| VGGT-Long, 1 FPS | 0.15 | $0.31 | $0.37 | $0.21 |
| VGGT-Long, 2 FPS | 0.30 | $0.62 | $0.74 | $0.41 |
| VGGT-Long, 10 FPS | 1.48 | $3.10 | $3.69 | $2.05 |

CPU, RAM, storage operations, and output transfer are excluded. A practical
job allowance should be **2–4 times** the inference floor until Mapper records
real end-to-end timings.

MASt3R-SLAM and VGGT-Ω should initially be budgeted from measured pilot runs,
not nominal paper throughput. VGGT-Ω's official memory table reaches 43.15 GB
at 500 frames, so its window size can force a 48 or 80 GB tier even when its
wall-clock runtime is modest.

## Experiment budgets

These are deliberately ranges because export and retry costs are not measured:

| Experiment level | Work | Approximate GPU budget | Modal A100 80 GB | Runpod A100 Pod |
| --- | --- | ---: | ---: | ---: |
| Adapter smoke | Short capture through all four new/control adapters | 1–2 GPU-h | $2.50–$5 | $1.39–$2.78 |
| First useful comparison | Full corpus, four adapters, 1 FPS and 2 FPS, one retry allowance | 4–10 GPU-h | $10–$25 | $5.56–$13.90 |
| Alignment/keyframe sweep | Above plus three alignment modes, keyframe ablations, and failed-run allowance | 15–40 GPU-h | $37.50–$100 | $20.85–$55.60 |
| Sustained research month | Several configuration sweeps or new long captures | 40–100 GPU-h | $100–$250 | $55.60–$139 |

Modal's Starter tier currently includes $30/month of compute credit, which
could cover the first useful comparison if the adapters stay within the
estimate. Runpod Pods are cheaper per GPU-hour for sustained work, but only if
an automated worker destroys the Pod after each queue drains; a forgotten
A100 Pod costs more in idle time than this entire first corpus costs to
process.

## Upload-on-demand versus persistent originals

For this corpus, **upload each original once and retain it**:

- the entire metadata-described corpus fits within R2's current 10 GB-month
  free tier;
- repeated upload time is more expensive operationally than storage;
- a stable object URI plus SHA-256 digest makes experiments reproducible;
- all model variants can reuse the same source without coupling jobs to the
  workstation being online; and
- serverless cold starts should cache model weights, not create new copies of
  source videos.

Upload-on-demand becomes reasonable only for a one-off privacy-sensitive
capture that will not be rerun, or when originals grow to hundreds of
gigabytes and experiments are rare. Even then, a short-lived object with an
automatic expiration is preferable to streaming the MP4 inside an RPC.

For repeated experimentation:

1. retain original MP4/MOV plus GPMF/telemetry indefinitely;
2. retain exact keyframe tables, configs, image/weight digests, logs, metrics,
   optimized graphs, and accepted canonical packages;
3. cache decoded keyframes for 7–30 days only when several runs will reuse the
   same selection digest;
4. checkpoint submaps for 7–30 days while an experiment sweep is active;
5. keep raw depths, temporary PLYs, and model-native scratch on ephemeral local
   NVMe and delete them after validation; and
6. use lifecycle rules rather than manual cleanup.

The storage decision should be revisited when the source-video corpus exceeds
the free tier, but even 100 GB of originals would cost only about $1.35/month
on R2 after the first 10 GB. Compute and engineering time will remain the
dominant costs.

## Price sources

- [Modal pricing](https://modal.com/pricing)
- [Runpod pricing](https://www.runpod.io/pricing)
- [Cloudflare R2 pricing](https://developers.cloudflare.com/r2/pricing/)
- [Amazon S3 pricing](https://aws.amazon.com/s3/pricing/)
- [DA3-Streaming runtime comparison](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md)
- [VGGT-Ω runtime and GPU memory](https://github.com/facebookresearch/vggt-omega#runtime-and-gpu-memory)
