"""Single-window DA3 inference and camera-aware depth unprojection."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .da3_compat import load_depth_anything3


@dataclass
class Reconstruction:
    points: np.ndarray
    colors: np.ndarray
    camera_centers: np.ndarray
    depth_shape: tuple[int, ...]
    confidence_shape: tuple[int, ...]
    extrinsics_shape: tuple[int, ...]
    runtime_seconds: float


def reconstruct(
    frame_paths: list[str | Path],
    *,
    process_res: int = 308,
    device: str | None = None,
    confidence_percentile: float = 15.0,
    max_points: int = 600_000,
) -> Reconstruction:
    """Infer one window, unproject confident pixels into a shared world frame."""
    repo = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(repo.parent / ".hf-cache"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    model_cls = load_depth_anything3()
    model = model_cls.from_pretrained("depth-anything/DA3-BASE").to(device).eval()
    with torch.inference_mode():
        pred = model.inference(
            [str(path) for path in frame_paths],
            process_res=process_res,
            process_res_method="upper_bound_resize",
            ref_view_strategy="middle",
        )
    runtime = time.perf_counter() - start

    depth = np.asarray(pred.depth, dtype=np.float32)
    conf = np.asarray(pred.conf, dtype=np.float32)
    extrinsics = np.asarray(pred.extrinsics, dtype=np.float64)
    intrinsics = np.asarray(pred.intrinsics, dtype=np.float64)
    images = np.asarray(pred.processed_images, dtype=np.uint8)
    if extrinsics.shape[-2:] == (3, 4):
        ext4 = np.broadcast_to(np.eye(4), (len(extrinsics), 4, 4)).copy()
        ext4[:, :3, :4] = extrinsics
    elif extrinsics.shape[-2:] == (4, 4):
        ext4 = extrinsics
    else:
        raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")

    all_points, all_colors, centers = [], [], []
    for i, (z, confidence, ext, k, rgb) in enumerate(zip(depth, conf, ext4, intrinsics, images)):
        camera_to_world = np.linalg.inv(ext)
        centers.append(camera_to_world[:3, 3])
        threshold = np.nanpercentile(confidence, confidence_percentile)
        valid = np.isfinite(z) & (z > 0) & np.isfinite(confidence) & (confidence >= threshold)
        v, u = np.nonzero(valid)
        fx, fy = k[0, 0], k[1, 1]
        cx, cy = k[0, 2], k[1, 2]
        camera_points = np.column_stack(((u - cx) * z[v, u] / fx, (v - cy) * z[v, u] / fy, z[v, u]))
        world_points = camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
        all_points.append(world_points.astype(np.float32))
        all_colors.append(rgb[v, u, :3])

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if len(points) > max_points:
        keep = np.random.default_rng(7).choice(len(points), max_points, replace=False)
        points, colors = points[keep], colors[keep]
    return Reconstruction(
        points=points,
        colors=colors,
        camera_centers=np.asarray(centers, dtype=np.float32),
        depth_shape=depth.shape,
        confidence_shape=conf.shape,
        extrinsics_shape=extrinsics.shape,
        runtime_seconds=runtime,
    )
