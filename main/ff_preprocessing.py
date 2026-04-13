import os
import torch
import numpy as np
import cv2
from decord import VideoReader, cpu
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== CONFIG =====
DATA_ROOT = "../ff/FaceForensics++_C23"
SAVE_ROOT = "../data/processed"

NUM_FRAMES = 20
NUM_REGIONS = 4
NUM_WORKERS = os.cpu_count() // 2  # more aggressive
CHUNK_SIZE = 256

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

os.makedirs(SAVE_ROOT, exist_ok=True)

# ===== GLOBAL FACE DETECTOR (per-process lazy init) =====
_face_detector = None

def get_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_detector


# ===== FRAME SAMPLING (FAST) =====
def sample_frames(video_path, num_regions=4, frames_per_region=5):
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)

    total_samples = num_regions * frames_per_region
    indices = np.linspace(0, total - 1, total_samples, dtype=int)

    # 🔥 batch fetch (much faster)
    frames = vr.get_batch(indices).asnumpy()
    return frames


# ===== VIDEO PROCESSING =====
def process_video(video_path):
    face_cascade = get_face_detector()

    frames = sample_frames(
        video_path,
        num_regions=NUM_REGIONS,
        frames_per_region=NUM_FRAMES // NUM_REGIONS
    )

    processed = []
    last_box = None
    region_size = NUM_FRAMES // NUM_REGIONS

    for i, f in enumerate(frames):
        # 🔥 detect only once per region
        if i % region_size == 0:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces):
                last_box = max(faces, key=lambda b: b[2] * b[3])

        if last_box is not None:
            x, y, w, h = last_box
            crop = f[y:y+h, x:x+w]
            if crop.size == 0:
                crop = f
        else:
            crop = f

        crop = cv2.resize(crop, (224, 224))

        rgb = crop.astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        tensor = torch.from_numpy(rgb.transpose(2, 0, 1))  # no extra float()

        processed.append(tensor)

    return torch.stack(processed)  # (T, 3, 224, 224)


# ===== SINGLE TASK =====
def process_single(video_path, label):
    save_path = os.path.join(
        SAVE_ROOT,
        os.path.relpath(video_path, DATA_ROOT)
    ).replace(".mp4", ".pt")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        return None

    try:
        tensor = process_video(video_path)
        torch.save(
            {"video": tensor, "label": label},
            save_path,
            _use_new_zipfile_serialization=False  # faster
        )
    except Exception as e:
        return f"Error: {video_path} {e}"

    return None


# ===== DATA COLLECTION =====
def collect_videos(split_dir, label):
    tasks = []
    for root, _, files in os.walk(split_dir):
        for f in files:
            if f.endswith(".mp4"):
                tasks.append((os.path.join(root, f), label))
    return tasks


# ===== MAIN =====
if __name__ == "__main__":
    all_tasks = []

    # real
    all_tasks += collect_videos(os.path.join(DATA_ROOT, "real"), 0)

    # fake
    fake_root = os.path.join(DATA_ROOT, "fake")
    FAKE_FOLDERS = [
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures"
    ]

    for sub in FAKE_FOLDERS:
        path = os.path.join(fake_root, sub)
        if os.path.exists(path):
            all_tasks += collect_videos(path, 1)

    print(f"Total videos: {len(all_tasks)}")
    print(f"Using {NUM_WORKERS} workers")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for i in range(0, len(all_tasks), CHUNK_SIZE):
            chunk = all_tasks[i:i + CHUNK_SIZE]

            futures = [
                executor.submit(process_single, path, label)
                for path, label in chunk
            ]

            for f in tqdm(as_completed(futures), total=len(futures)):
                err = f.result()
                if err:
                    print(err)

    print("✅ Parallel preprocessing done!")