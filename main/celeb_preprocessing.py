import os
import torch
import numpy as np
import cv2
from decord import VideoReader, cpu
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== CONFIG =====
DATA_ROOT = "../celeb-df"
SAVE_ROOT = "../processed_celeb"

NUM_FRAMES = 20  # must be divisible by 5
NUM_WORKERS = os.cpu_count() // 2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

os.makedirs(SAVE_ROOT, exist_ok=True)

# ===== FACE DETECTOR =====
def get_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

def get_face_crop(frame, face_cascade, size=224):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces):
        x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
        crop = frame[y:y+h, x:x+w]
        if crop.size > 0:
            return cv2.resize(crop, (size, size))

    return cv2.resize(frame, (size, size))


# ===== FFT =====
def compute_fft(gray):
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    return mag[..., None]


# ===== FRAME SAMPLING =====
def sample_frames(video_path, num_regions=4, frames_per_region=5, mode="region"):
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)

    if total == 0:
        raise ValueError("Empty video")

    if mode == "region":
        indices = []
        half = frames_per_region // 2
        region_size = total / num_regions

        for r in range(num_regions):
            center = int((r + 0.5) * region_size)

            for offset in range(-half, half + 1):
                idx = max(0, min(total - 1, center + offset))
                indices.append(idx)

    elif mode == "uniform":
        total_samples = num_regions * frames_per_region
        indices = np.linspace(0, total - 1, total_samples, dtype=int).tolist()

    else:
        raise ValueError("mode must be 'region' or 'uniform'")

    return [vr[i].asnumpy() for i in indices]


# ===== VIDEO PROCESSING =====
def process_video(video_path):
    face_cascade = get_face_detector()  # per-process init

    frames = sample_frames(video_path, num_regions=NUM_FRAMES//5, frames_per_region=5)

    processed = []
    for f in frames:
        crop = get_face_crop(f, face_cascade)

        rgb = crop.astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()  # (3, H, W)

        processed.append(tensor)

    return torch.stack(processed)  # (T, 3, 224, 224)


# ===== VIDEO LENGTH =====
def get_video_length(video_path):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        return len(vr)
    except:
        return None


# ===== SAVE SINGLE =====
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

        torch.save({
            "video": tensor,
            "label": label
        }, save_path)

    except Exception as e:
        return f"Error: {video_path} | {e}"

    return None


# ===== COLLECT =====
def collect_videos(split_dir, label):
    tasks = []
    lengths = []

    for root, _, files in os.walk(split_dir):
        for f in files:
            if f.endswith(".mp4"):
                path = os.path.join(root, f)

                tasks.append((path, label))

                l = get_video_length(path)
                if l is not None:
                    lengths.append(l)

    return tasks, lengths


# ===== MAIN =====
if __name__ == "__main__":
    all_tasks = []
    all_lengths = []

    # ===== REAL =====
    for real_folder in ["Celeb-real", "YouTube-real"]:
        path = os.path.join(DATA_ROOT, real_folder)
        if os.path.exists(path):
            tasks, lengths = collect_videos(path, 0)
            all_tasks += tasks
            all_lengths += lengths

    # ===== FAKE =====
    fake_path = os.path.join(DATA_ROOT, "Celeb-synthesis")
    if os.path.exists(fake_path):
        tasks, lengths = collect_videos(fake_path, 1)
        all_tasks += tasks
        all_lengths += lengths

    print(f"Total videos: {len(all_tasks)}")
    print(f"Using {NUM_WORKERS} workers")

    # ===== LENGTH STATS =====
    if len(all_lengths) > 0:
        lengths_np = np.array(all_lengths)

        print("\n📊 Video Length Stats (frames):")
        print(f"Min:  {lengths_np.min()}")
        print(f"Max:  {lengths_np.max()}")
        print(f"Avg:  {lengths_np.mean():.2f}")
    else:
        print("⚠️ No valid video lengths found")

    # ===== PROCESS =====
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_single, path, label)
            for path, label in all_tasks
        ]

        for f in tqdm(as_completed(futures), total=len(futures)):
            err = f.result()
            if err:
                print(err)

    print("✅ Celeb-DF preprocessing done!")