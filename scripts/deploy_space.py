"""
deploy_space.py - publish Pronunciation Coach as a Hugging Face Space (a free, permanent public URL).

    hf auth login
    python scripts/deploy_space.py

Run it again after any change to update the Space. Only the files the app needs are
uploaded: never your history, saved recordings, report or downloaded models.
"""

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_NAME = sys.argv[1] if len(sys.argv) > 1 else "pronunciation-coach"

INCLUDE = ["*.py", "requirements.txt", "packages.txt", "data/sentences.json", "data/calibration.json",
           "data/cache/*.wav", "demo/*.wav", "demo/demo_files.json", "scripts/*.py", "tests/*.py",
           "experiments/*.py", "experiments/results/results.md", "pytest.ini"]


def space_readme(gradio_version):
    """README.md with the settings block Hugging Face reads to build the Space."""
    header = f"""---
title: Pronunciation Coach
emoji: 🗣️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Accent-aware pronunciation practice (Whisper + DTW)
---

"""
    return header + (ROOT / "README.md").read_text(encoding="utf-8")


def main():
    """Log in check, create the Space if needed, copy the needed files, upload them."""
    import gradio
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        print("You are not logged in to Hugging Face. Run:  hf auth login")
        sys.exit(1)
    repo_id = f"{user}/{SPACE_NAME}"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
    print(f"Space ready: {repo_id}")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        count = 0
        for pattern in INCLUDE:
            for src in ROOT.glob(pattern):
                if "__pycache__" in src.parts or not src.is_file():
                    continue
                dest = staging / src.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                count += 1
        (staging / "README.md").write_text(space_readme(gradio.__version__), encoding="utf-8")
        print(f"Uploading {count + 1} files (this can take a minute)...")
        api.upload_folder(folder_path=str(staging), repo_id=repo_id, repo_type="space",
                          commit_message="Deploy Pronunciation Coach")

    print("\nDone. The Space now builds itself (about 5 to 10 minutes the first time).")
    print(f"Public link: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
