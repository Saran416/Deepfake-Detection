"""
build_test_set_celebdf.py
=========================
Constructs a test-video folder from the Celeb-DF dataset.
Run: python build_test_set_celebdf.py
"""

import shutil
import random
from pathlib import Path

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════

DATA_ROOT  = "../celeb-df"     # path to celeb-df folder
OUTPUT_DIR = "./test_videos_celeb"

NUM_REAL = 100                 # total real videos (from both folders combined)
NUM_FAKE = 250                 # total fake videos
SEED     = 42

# ══════════════════════════════════════════════════════

REAL_FOLDERS = ["Celeb-real", "YouTube-real"]
FAKE_FOLDER  = "Celeb-synthesis"


def collect_videos(folders, limit=None):
    videos = []
    for folder in folders:
        videos.extend(sorted(folder.rglob("*.mp4")))

    if limit and len(videos) > limit:
        random.shuffle(videos)
        videos = videos[:limit]

    return videos


def build_test_set(data_root, output_dir, num_real, num_fake, seed=42):
    random.seed(seed)

    data_root  = Path(data_root)
    output_dir = Path(output_dir)

    real_out = output_dir / "real"
    fake_out = output_dir / "fake"

    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    # ── REAL VIDEOS ─────────────────────────────────────
    real_src_folders = [data_root / f for f in REAL_FOLDERS]

    for f in real_src_folders:
        if not f.exists():
            print(f"[ERROR] Missing folder: {f}")
            return

    real_videos = collect_videos(real_src_folders, limit=num_real)

    print(f"\n[Real] Copying {len(real_videos)} videos...")
    for v in real_videos:
        dest = real_out / v.name
        if not dest.exists():
            shutil.copy2(v, dest)

    print(f"[Real] ✅ {len(real_videos)} → {real_out}")

    # ── FAKE VIDEOS ─────────────────────────────────────
    fake_src = data_root / FAKE_FOLDER

    if not fake_src.exists():
        print(f"[ERROR] Missing folder: {fake_src}")
        return

    fake_videos = collect_videos([fake_src], limit=num_fake)

    print(f"\n[Fake] Copying {len(fake_videos)} videos...")
    for v in fake_videos:
        dest = fake_out / v.name
        if not dest.exists():
            shutil.copy2(v, dest)

    print(f"[Fake] ✅ {len(fake_videos)} → {fake_out}")

    # ── SUMMARY ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  TEST SET SUMMARY")
    print("=" * 50)
    print(f"  Real videos : {len(real_videos)}")
    print(f"  Fake videos : {len(fake_videos)}")
    print(f"  Total       : {len(real_videos) + len(fake_videos)}")
    print(f"  Output dir  : {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    print("=" * 50)
    print("  BUILD CELEB-DF TEST SET")
    print("=" * 50)
    print(f"  Data root : {DATA_ROOT}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"  Real      : {NUM_REAL}")
    print(f"  Fake      : {NUM_FAKE}")
    print(f"  Seed      : {SEED}")
    print("=" * 50)

    build_test_set(
        data_root=DATA_ROOT,
        output_dir=OUTPUT_DIR,
        num_real=NUM_REAL,
        num_fake=NUM_FAKE,
        seed=SEED,
    )