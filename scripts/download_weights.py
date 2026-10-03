"""Download DA3-BASE weights into the local Hugging Face cache."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT.parent / ".hf-cache"))

from huggingface_hub import snapshot_download


def main() -> None:
    snapshot_download("depth-anything/DA3-BASE")
    print(f"DA3-BASE weights are available under {os.environ['HF_HOME']}.")


if __name__ == "__main__":
    main()
