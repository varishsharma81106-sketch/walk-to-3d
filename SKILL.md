---
name: walk-to-3d
description: Reconstructs an indoor place in 3D from a phone walkthrough video using a local open-weight multi-view model (VGGT), returning a point cloud, camera path and top-down map. Use when you need a private 3D scan of a room without LiDAR or cloud upload.
---
# Walk to 3D

## Steps
1. `python -m walk3d.cli --video <path> --out <dir> [--loop]`
2. Read `<dir>/stats.json`; open `<dir>/scene.ply` and `<dir>/map.png`.

## Requirements
Python 3.10+, CUDA GPU (≥6 GB), VGGT weights per README.

## Limits
Static scenes; one room plus an adjoining area; drift accumulates on long walks; reflective or blank surfaces degrade results.
