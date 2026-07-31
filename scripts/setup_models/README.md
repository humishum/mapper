# Model checkpoint setup

All checkpoint bytes are stored under the repository-root `weights/` directory,
which is ignored by Git. External pipelines that hard-code `weights/` or
`checkpoints/` receive a symlink back to this canonical directory.

Examples:

```bash
# One in-process model
scripts/setup_models/download_models.sh vggt

# Several models
DA3_STREAMING_PATH=../Depth-Anything-3 \
MAST3R_SLAM_PATH=../MASt3R-SLAM \
scripts/setup_models/download_models.sh da3-streaming mast3r-slam

# Inspect every action without downloading
DA3_STREAMING_PATH=../Depth-Anything-3 \
VGGT_LONG_PATH=../VGGT-Long \
MAST3R_SLAM_PATH=../MASt3R-SLAM \
scripts/setup_models/download_models.sh all --dry-run
```

VGGT-Omega is gated. Request access to
[`facebook/VGGT-Omega`](https://huggingface.co/facebook/VGGT-Omega) and set
`HF_TOKEN` (or run `hf auth login`) before downloading it.

The scripts do not clone or update model source repositories. Adapter configs
pin those repositories separately, and the external checkout variables must
point at those exact revisions.
