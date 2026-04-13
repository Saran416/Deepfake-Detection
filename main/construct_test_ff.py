"""
build_test_set.py
=================
Constructs a test-video folder from the FaceForensics++ C23 dataset.
Edit the CONFIG block below, then run:  python build_test_set.py
"""

import os
import shutil
import random
from pathlib import Path

# ══════════════════════════════════════════════════════
#  CONFIG  —  edit these values
# ══════════════════════════════════════════════════════

DATA_ROOT  = "../ff/FaceForensics++_C23"   # root of the FF++ dataset
OUTPUT_DIR = "./test_videos_ff"               # where the test set will be written

# Set to True to include ALL fake methods,
# or set to False and list only the ones you want in FAKE_METHODS below.
USE_ALL_METHODS = True

FAKE_METHODS = [       # only used when USE_ALL_METHODS = False
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]

NUM_REAL           = 100   # max real videos to copy
NUM_FAKE_PER_METHOD = 50  # max fake videos per method
SEED               = 42    # for reproducible random sampling

# ══════════════════════════════════════════════════════

ALL_FAKE_METHODS = [
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]

REAL_FOLDER = "original"


def collect_videos(folder: Path, limit: int = None) -> list:
    videos = sorted(folder.rglob("*.mp4"))
    if limit and len(videos) > limit:
        random.shuffle(videos)
        videos = videos[:limit]
    return videos


def build_test_set(
    data_root: str,
    output_dir: str,
    fake_methods: list,
    num_real: int,
    num_fake_per_method: int,
    seed: int = 42,
):
    random.seed(seed)
    data_root  = Path(data_root)
    output_dir = Path(output_dir)

    real_out = output_dir / "real"
    fake_out = output_dir / "fake"
    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    # ── Real videos ──────────────────────────────────────────────────────────
    real_src = data_root / REAL_FOLDER
    if not real_src.exists():
        print(f"[ERROR] Real folder not found: {real_src}")
        return

    real_videos = collect_videos(real_src, limit=num_real)
    print(f"\n[Real] Found source videos — copying {len(real_videos)} ...")
    for v in real_videos:
        dest = real_out / v.name
        if not dest.exists():
            shutil.copy2(v, dest)
    print(f"[Real] ✅ {len(real_videos)} videos → {real_out}")

    # ── Fake videos ──────────────────────────────────────────────────────────
    total_fake = 0
    for method in fake_methods:
        method_src = data_root / method
        if not method_src.exists():
            print(f"[WARN] Method folder not found, skipping: {method_src}")
            continue

        method_out = fake_out / method
        method_out.mkdir(parents=True, exist_ok=True)

        videos = collect_videos(method_src, limit=num_fake_per_method)
        print(f"[{method}] Copying {len(videos)} videos ...")
        for v in videos:
            dest = method_out / v.name
            if not dest.exists():
                shutil.copy2(v, dest)

        total_fake += len(videos)
        print(f"[{method}] ✅ {len(videos)} videos → {method_out}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("  TEST SET SUMMARY")
    print("═" * 50)
    print(f"  Real videos       : {len(real_videos)}")
    print(f"  Fake videos       : {total_fake}  (across {len(fake_methods)} method(s))")
    print(f"  Total             : {len(real_videos) + total_fake}")
    print(f"  Output dir        : {output_dir}")
    print("═" * 50)


if __name__ == "__main__":
    methods = ALL_FAKE_METHODS if USE_ALL_METHODS else FAKE_METHODS

    print("=" * 50)
    print("  BUILD TEST SET")
    print("=" * 50)
    print(f"  Data root         : {DATA_ROOT}")
    print(f"  Output dir        : {OUTPUT_DIR}")
    print(f"  Fake methods      : {', '.join(methods)}")
    print(f"  Real videos       : up to {NUM_REAL}")
    print(f"  Fake / method     : up to {NUM_FAKE_PER_METHOD}")
    print(f"  Seed              : {SEED}")
    print("=" * 50)

    build_test_set(
        data_root           = DATA_ROOT,
        output_dir          = OUTPUT_DIR,
        fake_methods        = methods,
        num_real            = NUM_REAL,
        num_fake_per_method = NUM_FAKE_PER_METHOD,
        seed                = SEED,
    )