# Walk→3D

Turn a short phone walkthrough into a colored point cloud and camera path on your own machine. The first version samples sharp frames, runs one Depth Anything 3 window, unprojects depth with the predicted camera matrices, and writes a PLY file.

## Current status

DA3-BASE smoke inference ran on CPU and on an NVIDIA GeForce RTX 4050 Laptop GPU on 2026-10-03. The CUDA smoke used 8 frames at `process_res=308`; depth and confidence were `(8, 168, 308)`, extrinsics were `(8, 3, 4)`, and all output arrays were finite. The bundled example also ran through the CLI and exported a 351,856-point PLY. The browser page returned HTTP 200; its upload workflow and reconstruction quality have not been ground-truth validated. See [`docs/findings.md`](docs/findings.md).

## Setup

Use Python 3.12 and install a PyTorch wheel compatible with your NVIDIA driver if you want CUDA. The tested environment used `torch==2.14.0+cu130` and `torchvision==0.29.0+cu130`; CPU-only PyTorch also works.

```powershell
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git third_party/da3
python -m pip install --no-deps -e third_party/da3
python -m pip install -e .
python -m pip install einops huggingface_hub imageio opencv-python omegaconf safetensors pillow 'numpy<2' addict tqdm trimesh moviepy==1.0.3 pyyaml requests scipy matplotlib plyfile
```

DA3-BASE weights download on first run into the Hugging Face cache. On the original Windows setup, `scripts/env.ps1` directs temporary files and the model cache to `D:\octfest`.

## Run

```powershell
python -m walk3d.cli --video path\to\walkthrough.mp4 --out out\scan --frames 12 --process-res 308
python -m walk3d.app
```

The command writes `scene.ply`, `camera_path.json`, and `stats.json`. The app listens on `0.0.0.0:7860` for local-network access.

## Limits

The current reconstruction uses one inference window. Long videos, reflective or textureless surfaces, and pose drift can produce incomplete or distorted geometry. CUDA was smoke-tested on one RTX 4050 Laptop GPU, but the full pipeline has not been validated on a ground-truth scan. No general VRAM or accuracy claim is made.
