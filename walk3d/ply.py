"""Small dependency-free binary PLY writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("points and colors must have the same number of rows")
    vertex = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertex.tofile(stream)
    return path
