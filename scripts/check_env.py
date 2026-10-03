"""Print the local runtime details needed for Walk→3D."""

from __future__ import annotations

import platform
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DA3_SOURCE = ROOT / "third_party" / "da3" / "src"
if DA3_SOURCE.exists():
    sys.path.insert(0, str(DA3_SOURCE))
os.environ.setdefault("HF_HOME", str(ROOT.parent / ".hf-cache"))


def main() -> int:
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    try:
        import torch
    except ImportError:
        print("PyTorch: not installed")
        print("GPU: unavailable to PyTorch")
        print("DA3 import: not checked (install dependencies first)")
        return 1

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda or 'CPU-only build'}")
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if available:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        vram_gib = properties.total_memory / (1024**3)
        print(f"GPU: {properties.name}")
        print(f"VRAM: {vram_gib:.2f} GiB")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    else:
        print("GPU: unavailable to PyTorch")
        print("bf16 supported: False")

    try:
        import vggt

        import depth_anything_3

        print(f"DA3 import: OK ({Path(depth_anything_3.__file__).resolve()})")
    except Exception as exc:
        print(f"DA3 import: FAILED ({type(exc).__name__}: {exc})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
