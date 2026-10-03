"""Evenly sample sharp keyframes from a walkthrough video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sample_sharp_frames(
    video: str | Path,
    output_dir: str | Path,
    *,
    target_frames: int = 12,
    min_frames: int = 8,
    sharpness_floor: float = 25.0,
) -> list[Path]:
    """Save an even-in-time subset of sharp frames, capped to 8–16 images."""
    target_frames = min(max(int(target_frames), min_frames), 16)
    min_frames = min(max(int(min_frames), 1), target_frames)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if reported < min_frames:
        capture.release()
        raise ValueError(f"Video has only {reported} reported frames; need {min_frames}.")

    candidate_count = min(reported, target_frames * 3)
    # Some containers over-report their last decodable frame by one.
    indices = np.linspace(0, max(reported - 2, 0), candidate_count).round().astype(int)
    selected: dict[int, tuple[float, np.ndarray]] = {}
    index_set = set(indices.tolist())
    for frame_index in range(int(indices[-1]) + 1):
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in index_set:
            sharpness = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            selected[frame_index] = (sharpness, frame)
    capture.release()

    candidates = [(i, score, frame) for i, (score, frame) in selected.items() if score >= sharpness_floor]
    if len(candidates) < min_frames:
        candidates = [(i, score, frame) for i, (score, frame) in selected.items()]
    if len(candidates) < min_frames:
        raise ValueError(f"Only {len(candidates)} usable frames decoded; need {min_frames}.")

    # Pick the sharpest candidate in each temporal bin for even coverage.
    candidates.sort(key=lambda item: item[0])
    bins = np.array_split(np.arange(len(candidates)), min(target_frames, len(candidates)))
    ranked = [max((candidates[i] for i in group), key=lambda item: item[1]) for group in bins if len(group)]
    if len(ranked) < min_frames:
        extras = sorted(candidates, key=lambda item: item[1], reverse=True)
        chosen = {item[0] for item in ranked}
        ranked.extend(item for item in extras if item[0] not in chosen)
        ranked = sorted(ranked[:target_frames], key=lambda item: item[0])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, (_, _, frame) in enumerate(ranked):
        path = out / f"frame_{number:02d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise OSError(f"Could not write keyframe: {path}")
        paths.append(path)
    return paths
