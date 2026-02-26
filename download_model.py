"""
download_model.py
─────────────────
Downloads the pre-converted Llama-Prompt-Guard-2-86M ONNX model
from the community repo (no Meta approval needed).

Source: gravitee-io/Llama-Prompt-Guard-2-86M-onnx

Usage:
    python download_model.py
"""

import sys
from pathlib import Path

MODEL_REPO = "gravitee-io/Llama-Prompt-Guard-2-86M-onnx"
OUTPUT_DIR = Path(__file__).parent / "model_onnx"


def download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "❌  Missing dependency.\n"
            "    Run:  pip install -r requirements.txt\n"
        )

    if (OUTPUT_DIR / "model.onnx").exists():
        print(f"✅  Model already downloaded at {OUTPUT_DIR}")
        print("    Delete the folder and re-run for a fresh download.")
        return

    print(f"⬇️   Downloading {MODEL_REPO} …")
    print(f"    (This is an ungated community model — no Meta approval needed)\n")

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(OUTPUT_DIR),
        local_dir_use_symlinks=False,
    )

    # Verify key files exist
    expected = ["model.onnx", "tokenizer.json", "config.json"]
    missing = [f for f in expected if not (OUTPUT_DIR / f).exists()]

    if missing:
        print(f"⚠️   Warning: missing expected files: {missing}")
    else:
        print(f"\n✅  Download complete → {OUTPUT_DIR}")
        print(f"    Files: {', '.join(expected)}")
        print(f"\n    You can now run:  python demo.py")


if __name__ == "__main__":
    download()

