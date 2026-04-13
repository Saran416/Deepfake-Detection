"""
evaluate_cross_data.py
======================
Evaluates a trained deepfake detection model on an external test folder.
Reports PR-AUC and ROC-AUC for cross-dataset generalisation analysis.

Folder structure expected:
  test_videos/
  ├── real/       (label = 0)
  └── fake/       (label = 1, may have subfolders per method)

Edit the CONFIG block, then run:  python evaluate_cross_data.py
"""

# ══════════════════════════════════════════════════════
#  CONFIG  —  edit these values
# ══════════════════════════════════════════════════════

CHECKPOINT  = "./models_celeb/best_model_residual_htsm.pth"
TEST_FOLDER = "./test_videos_celeb"

NUM_FRAMES  = 20
NUM_REGIONS = 4

# ══════════════════════════════════════════════════════

import os
import warnings
import numpy as np
import torch
import cv2
from pathlib import Path
from decord import VideoReader, cpu as decord_cpu
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from models import (
    DeepfakeEdgeModel_GRU, DeepfakeEdgeModel_STSM, DeepfakeEdgeModel_HTSM, 
    DeepfakeEdgeModel_Mobile, DeepfakeEdgeModel_Xception, DeepfakeEdgeModel_Residual_HTSM
)

# ─────────────────────────────────────────────────────
# Pre-processing  (identical to preprocess.py)
# ─────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_face_detector = None

def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_detector


def preprocess_video(video_path: str) -> torch.Tensor:
    face_cascade = _get_face_detector()
    vr = VideoReader(video_path, ctx=decord_cpu(0))
    total = len(vr)
    total_samples = NUM_REGIONS * (NUM_FRAMES // NUM_REGIONS)
    indices = np.linspace(0, total - 1, total_samples, dtype=int)
    frames = vr.get_batch(indices).asnumpy()

    processed = []
    last_box   = None
    region_size = NUM_FRAMES // NUM_REGIONS

    for i, f in enumerate(frames):
        if i % region_size == 0:
            gray  = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces):
                last_box = max(faces, key=lambda b: b[2] * b[3])

        if last_box is not None:
            x, y, w, h = last_box
            crop = f[y:y+h, x:x+w]
            crop = f if crop.size == 0 else crop
        else:
            crop = f

        crop = cv2.resize(crop, (224, 224))
        rgb  = crop.astype(np.float32) / 255.0
        rgb  = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        processed.append(torch.from_numpy(rgb.transpose(2, 0, 1)))

    return torch.stack(processed)   # (T, 3, 224, 224)


# ─────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────
def collect_samples(test_folder: str):
    """
    Returns list of (video_path, label) tuples.
    real/ → 0,  fake/ (any subfolder) → 1
    """
    root = Path(test_folder)
    samples = []

    real_dir = root / "real"
    fake_dir = root / "fake"

    if real_dir.exists():
        for v in sorted(real_dir.rglob("*.mp4")):
            samples.append((str(v), 0))
    else:
        print(f"[WARN] No 'real' folder found under {test_folder}")

    if fake_dir.exists():
        for v in sorted(fake_dir.rglob("*.mp4")):
            samples.append((str(v), 1))
    else:
        print(f"[WARN] No 'fake' folder found under {test_folder}")

    return samples


# ─────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────
def run_inference(model, samples, device):
    all_probs  = []
    all_labels = []
    errors     = []

    for video_path, label in tqdm(samples, desc="Inference"):
        try:
            tensor = preprocess_video(video_path)          # (T, 3, 224, 224)
            x = tensor.unsqueeze(0).to(device, dtype=torch.float32)   # (1, T, 3, 224, 224)

            with torch.no_grad():
                logit = model(x)
                prob  = torch.sigmoid(logit).item()

            all_probs.append(prob)
            all_labels.append(label)

        except Exception as e:
            errors.append((video_path, str(e)))

    if errors:
        print(f"\n[WARN] {len(errors)} video(s) failed during inference:")
        for path, err in errors:
            print(f"  {path}: {err}")

    return np.array(all_labels), np.array(all_probs)


# ─────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────
def compute_metrics(labels, probs):
    roc_auc = roc_auc_score(labels, probs)
    pr_auc  = average_precision_score(labels, probs)

    # best threshold (maximises macro F1)
    thresholds = np.linspace(0.0, 1.0, 201)
    best_f1, best_thr = -1, 0.5
    for t in thresholds:
        f1 = f1_score(labels, (probs > t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    preds_05   = (probs > 0.50).astype(int)
    preds_best = (probs > best_thr).astype(int)

    return {
        "roc_auc":        roc_auc,
        "pr_auc":         pr_auc,
        "f1@0.5":         f1_score(labels, preds_05,   average="macro", zero_division=0),
        "precision@0.5":  precision_score(labels, preds_05,   zero_division=0),
        "recall@0.5":     recall_score(labels, preds_05,      zero_division=0),
        "f1@best":        best_f1,
        "precision@best": precision_score(labels, preds_best, zero_division=0),
        "recall@best":    recall_score(labels, preds_best,    zero_division=0),
        "best_threshold": best_thr,
        "num_real":       int((labels == 0).sum()),
        "num_fake":       int((labels == 1).sum()),
        "total":          len(labels),
    }


# ─────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────
def save_curves(labels, probs, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    RocCurveDisplay.from_predictions(labels, probs, ax=axes[0], name="Model")
    axes[0].set_title("ROC Curve")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)

    PrecisionRecallDisplay.from_predictions(labels, probs, ax=axes[1], name="Model")
    axes[1].set_title("Precision-Recall Curve")

    plt.tight_layout()
    out_path = out_dir / "curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n[INFO] Curves saved → {out_path}")


# ─────────────────────────────────────────────────────
# Pretty print + LaTeX
# ─────────────────────────────────────────────────────
def print_results(metrics: dict, checkpoint: str, test_folder: str):
    print("\n" + "═" * 60)
    print("  CROSS-DATASET EVALUATION RESULTS")
    print("═" * 60)
    print(f"  Checkpoint   : {checkpoint}")
    print(f"  Test folder  : {test_folder}")
    print(f"  Total videos : {metrics['total']}  "
          f"(real: {metrics['num_real']}, fake: {metrics['num_fake']})")

    print(f"\n{'Metric':<25} {'Value':>10}")
    print("  " + "─" * 35)
    print(f"  {'ROC-AUC':<23} {metrics['roc_auc']:>10.4f}")
    print(f"  {'PR-AUC':<23} {metrics['pr_auc']:>10.4f}")
    print(f"  {'F1  @ 0.50':<23} {metrics['f1@0.5']:>10.4f}")
    print(f"  {'Precision @ 0.50':<23} {metrics['precision@0.5']:>10.4f}")
    print(f"  {'Recall @ 0.50':<23} {metrics['recall@0.5']:>10.4f}")
    print(f"  {'F1  @ best thr':<23} {metrics['f1@best']:>10.4f}")
    print(f"  {'Precision @ best':<23} {metrics['precision@best']:>10.4f}")
    print(f"  {'Recall @ best':<23} {metrics['recall@best']:>10.4f}")
    print(f"  {'Best threshold':<23} {metrics['best_threshold']:>10.2f}")

    print(f"\n[LaTeX row]")
    print(
        f"  Ours & "
        f"{metrics['roc_auc']:.4f} & "
        f"{metrics['pr_auc']:.4f} & "
        f"{metrics['f1@best']:.4f} \\\\"
    )
    print("═" * 60)

def merge_checkpoints(model, ckpt1_path, ckpt2_path, device, alpha=0.5):
    """
    Merge two checkpoints into the given model.

    Args:
        model        : model instance (already initialized)
        ckpt1_path   : path to checkpoint 1
        ckpt2_path   : path to checkpoint 2
        device       : torch device
        alpha        : weight for ckpt1 (default = 0.5 → simple average)

    Returns:
        model with merged weights
    """

    print(f"[INFO] Merging checkpoints:")
    print(f"       ckpt1: {ckpt1_path}")
    print(f"       ckpt2: {ckpt2_path}")
    print(f"       alpha: {alpha}")

    state1 = torch.load(ckpt1_path, map_location=device)
    state2 = torch.load(ckpt2_path, map_location=device)

    merged_state = {}

    for k in state1.keys():
        if k in state2 and state1[k].shape == state2[k].shape:
            merged_state[k] = alpha * state1[k] + (1 - alpha) * state2[k]
        else:
            # fallback: use first checkpoint
            merged_state[k] = state1[k]
            print(f"[WARN] Skipping merge for key: {k}")

    model.load_state_dict(merged_state)
    return model

# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device      : {device}")
    print(f"[INFO] Checkpoint  : {CHECKPOINT}")
    print(f"[INFO] Test folder : {TEST_FOLDER}")

    # ── load model ───────────────────────────────────
    model = DeepfakeEdgeModel_Residual_HTSM()
    state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state)

    # ── load + merge model ───────────────────────────
    # model = DeepfakeEdgeModel_Residual_HTSM()
    # CKPT1 = "./models_celeb/best_model_residual_htsm.pth"
    # CKPT2 = "./models_ff/best_model_residual_htsm.pth"
    # model = merge_checkpoints(model, CKPT1, CKPT2, device, alpha=0.5)

    model.to(device)
    model.eval()
    print("[INFO] Merged model loaded ✅")

    print("[INFO] Model loaded ✅")

    # ── collect samples ──────────────────────────────
    samples = collect_samples(TEST_FOLDER)
    if not samples:
        print("[ERROR] No .mp4 files found. Check TEST_FOLDER structure.")
        return

    real_count = sum(1 for _, l in samples if l == 0)
    fake_count = sum(1 for _, l in samples if l == 1)
    print(f"[INFO] Samples     : {len(samples)}  (real: {real_count}, fake: {fake_count})")

    # ── run inference ────────────────────────────────
    labels, probs = run_inference(model, samples, device)

    # ── global metrics ───────────────────────────────
    metrics = compute_metrics(labels, probs)
    print_results(metrics, CHECKPOINT, TEST_FOLDER)

    # ── save ROC + PR curves ─────────────────────────
    out_dir = Path("eval_output")
    out_dir.mkdir(exist_ok=True)
    save_curves(labels, probs, out_dir)


if __name__ == "__main__":
    main()