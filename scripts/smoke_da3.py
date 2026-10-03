"""Run a local DA3-BASE inference smoke test on bundled example media."""

from __future__ import annotations

import os
import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT.parent / ".hf-cache"))
sys.path.insert(0, str(ROOT / "third_party" / "da3" / "src"))
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from walk3d.da3_compat import load_depth_anything3

DepthAnything3 = load_depth_anything3()


def extract_example_frames(video: Path, output: Path, count: int = 8) -> list[Path]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open example video: {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < count:
        capture.release()
        raise RuntimeError(f"Example video has only {total} frames; need {count}")

    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    # CAP_PROP_FRAME_COUNT can over-report the decodable frames by one.
    selected = set(np.linspace(0, max(total - 2, 0), count).round().astype(int).tolist())
    number = 0
    for index in range(max(selected) + 1):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Could not decode frame {index} from {video}")
        if index not in selected:
            continue
        frame_path = output / f"frame_{number:02d}.jpg"
        if not cv2.imwrite(str(frame_path), frame):
            capture.release()
            raise RuntimeError(f"Could not write smoke-test frame: {frame_path}")
        paths.append(frame_path)
        number += 1
    capture.release()
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--process-res", type=int, default=308)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    source = ROOT / "third_party" / "da3" / "assets" / "examples" / "robot_unitree.mp4"
    paths = extract_example_frames(source, ROOT / "work" / "smoke_frames", count=args.frames)
    print(f"HF_HOME: {os.environ['HF_HOME']}")
    print(f"Frames: {len(paths)}")
    print(f"device: {args.device}")
    print("precision: fp32" if args.device == "cpu" else "precision: model default")

    started = time.perf_counter()
    model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE").to(args.device)
    observed: list[tuple[bool, torch.dtype]] = []
    hook = None
    if args.device == "cuda":
        hook = model.model.register_forward_pre_hook(
            lambda _module, _inputs: observed.append(
                (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
            )
        )
        torch.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    prediction = model.inference(
        [str(path) for path in paths],
        process_res=args.process_res,
        process_res_method="upper_bound_resize",
        ref_view_strategy="middle",
    )
    inference_seconds = time.perf_counter() - inference_started
    if hook is not None:
        hook.remove()

    arrays = {
        "depth": prediction.depth,
        "conf": prediction.conf,
        "extrinsics": prediction.extrinsics,
        "intrinsics": prediction.intrinsics,
        "processed_images": prediction.processed_images,
    }
    for name, array in arrays.items():
        print(f"{name}: {None if array is None else np.asarray(array).shape}")
    print(f"extrinsic_shape: {np.asarray(prediction.extrinsics).shape}")
    print(f"all_outputs_finite: {all(np.isfinite(array).all() for array in arrays.values() if array is not None)}")
    print(f"inference_runtime_seconds: {inference_seconds:.2f}")
    print(f"total_runtime_seconds: {time.perf_counter() - started:.2f}")
    if args.device == "cuda":
        print(f"cuda_peak_allocated_mib: {torch.cuda.max_memory_allocated() / (1024**2):.1f}")
        print(f"bf16_supported: {torch.cuda.is_bf16_supported()}")
        print(f"autocast_observed: {observed}")
    else:
        print("bf16_autocast_active: False (not applicable on CPU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
