# Measured findings

## DA3-BASE smoke runs

- Date: 2026-10-03
- Input: bundled `robot_unitree.mp4`, 8 sampled frames, `process_res=308`.
- CPU run: `torch 2.14.1+cpu`, fp32; bf16 autocast does not apply on CPU. Depth and confidence `(8, 168, 308)`, extrinsics `(8, 3, 4)`, intrinsics `(8, 3, 3)`, processed images `(8, 168, 308, 3)`. All outputs finite; inference 19.83 s, total 24.28 s.
- CUDA run: `torch 2.14.0+cu130`, NVIDIA GeForce RTX 4050 Laptop GPU, CUDA available. Same output shapes; all outputs finite; inference 1.62 s, total 9.67 s; peak allocated 1003.3 MiB; bf16 supported and observed active.
- These are single smoke runs, not general performance or memory guarantees.

## Minimal pipeline run

- Ran `walk3d.cli` on the bundled example with 8 frames at `process_res=308` on CUDA.
- Exported `scene.ply` with 351,856 points plus `camera_path.json` and `stats.json` under the ignored `out/demo` directory.
- Rendered `out/demo/preview.png` for visual inspection. The preview shows plausible scene surfaces, but no ground-truth accuracy check was performed.
- Started the Gradio app on `0.0.0.0:7860`; `http://127.0.0.1:7860` returned HTTP 200. The upload-to-reconstruction browser interaction was not exercised.

## Scope limits

- The shipped pipeline uses one DA3 window. Camera poses and the generated PLY have not been validated against a ground-truth scan.
- The CUDA CLI path ran on one RTX 4050 Laptop GPU; the browser upload-to-reconstruction interaction remains unverified.
