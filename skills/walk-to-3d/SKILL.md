---
name: walk-to-3d
description: Reconstructs a small scene from a phone walkthrough video with local Depth Anything 3 inference and exports a colored PLY point cloud plus camera path. Use when someone wants a local first-pass 3D scan from a video without uploading the video to a service.
---
# Walk to 3D

## Steps
1. Run `python -m walk3d.cli --video <path> --out <dir> [--frames 12] [--process-res 308]`.
2. Open `<dir>/scene.ply`; read `<dir>/stats.json` and `<dir>/camera_path.json`.
3. For the browser interface, run `python -m walk3d.app` and open the local Gradio URL.

## Requirements
Python 3.12+, the DA3 source checkout under `third_party/da3`, and DA3-BASE weights. CPU inference works but can be slow; a CUDA-enabled PyTorch install uses a compatible NVIDIA driver. The CUDA smoke path was exercised on an RTX 4050 Laptop GPU; full walkthrough accuracy has not been evaluated.

## Limits
This is a single-window first pass, not a production SLAM system. Static scenes, low texture, reflections, and long walks can produce weak geometry or pose drift. No floor map, loop closure, or ground-truth accuracy claim is included.
