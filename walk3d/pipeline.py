"""Orchestrate keyframe sampling, one-window inference, and scene export."""

from __future__ import annotations

import json
from pathlib import Path

from .frames import sample_sharp_frames
from .ply import write_ply
from .recon import reconstruct


def run_video(
    video: str | Path,
    output_dir: str | Path,
    *,
    target_frames: int = 12,
    process_res: int = 308,
    device: str | None = None,
) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = sample_sharp_frames(video, output / "frames", target_frames=target_frames)
    result = reconstruct(frames, process_res=process_res, device=device)
    ply_path = write_ply(output / "scene.ply", result.points, result.colors)
    path_file = output / "camera_path.json"
    path_file.write_text(json.dumps(result.camera_centers.tolist(), indent=2), encoding="utf-8")
    summary = {
        "input_video": str(Path(video).resolve()),
        "frame_count": len(frames),
        "process_res": process_res,
        "device": device or "auto",
        "point_count": len(result.points),
        "depth_shape": result.depth_shape,
        "confidence_shape": result.confidence_shape,
        "extrinsics_shape": result.extrinsics_shape,
        "inference_runtime_seconds": round(result.runtime_seconds, 3),
        "point_cloud": str(ply_path),
        "camera_path": str(path_file),
        "limitations": "Single DA3 window; no loop closure or ground-truth accuracy claim.",
    }
    (output / "stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
