"""Command-line interface for Walk→3D."""

from __future__ import annotations

import argparse
import json

from .pipeline import run_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local point cloud from a walkthrough video.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out", default="out/scan", help="Output directory")
    parser.add_argument("--frames", type=int, default=12, help="Target sharp keyframes (8–16)")
    parser.add_argument("--process-res", type=int, default=308)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    summary = run_video(
        args.video,
        args.out,
        target_frames=args.frames,
        process_res=args.process_res,
        device=None if args.device == "auto" else args.device,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
