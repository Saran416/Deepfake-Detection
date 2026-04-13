"""
evaluate_cross_data.py
======================
Evaluates multiple deepfake detection checkpoints on an external test folder.
Reports ROC-AUC, PR-AUC, F1, FPR@95%TPR, TPR@1%FPR, EER, MCC, BalAcc.

Folder structure expected:
  test_videos/
  ├── real/       (label = 0)
  └── fake/       (label = 1, may have subfolders per method)

Edit the CONFIG block, then run:  python evaluate_cross_data.py
"""

# ══════════════════════════════════════════════════════
#  CONFIG  —  edit these values
# ══════════════════════════════════════════════════════

TEST_FOLDER = "./test_videos_celeb"
NUM_FRAMES  = 20
NUM_REGIONS = 4

CHECKPOINTS = {
    "XceptionNet":               "./models_celeb/best_model_xception.pth",
    "MobileNet":                 "./models_celeb/best_model_mobile.pth",
    "MobileNet + TSM":           "./models_celeb/best_model_single_tsm.pth",
    "MobileNet + HTSM":          "./models_celeb/best_model_htsm.pth",
    "MobileNet + Residual HTSM": "./models_celeb/best_model_residual_htsm.pth",
}

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
    matthews_corrcoef,
    roc_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from models import (
    DeepfakeEdgeModel_GRU,
    DeepfakeEdgeModel_STSM,
    DeepfakeEdgeModel_HTSM,
    DeepfakeEdgeModel_Mobile,
    DeepfakeEdgeModel_Xception,
    DeepfakeEdgeModel_Residual_HTSM,
)

# ─────────────────────────────────────────────────────
# Model registry — maps checkpoint name → model class
# ─────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "XceptionNet":               DeepfakeEdgeModel_Xception,
    "MobileNet":                 DeepfakeEdgeModel_Mobile,
    "MobileNet + TSM":           DeepfakeEdgeModel_STSM,
    "MobileNet + HTSM":          DeepfakeEdgeModel_HTSM,
    "MobileNet + Residual HTSM": DeepfakeEdgeModel_Residual_HTSM,
}

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

    processed  = []
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

    for video_path, label in tqdm(samples, desc="  Inference", leave=False):
        try:
            tensor = preprocess_video(video_path)          # (T, 3, 224, 224)
            x = tensor.unsqueeze(0).to(device, dtype=torch.float32)

            with torch.no_grad():
                logit = model(x)
                prob  = torch.sigmoid(logit).item()

            all_probs.append(prob)
            all_labels.append(label)

        except Exception as e:
            errors.append((video_path, str(e)))

    if errors:
        print(f"\n  [WARN] {len(errors)} video(s) failed during inference:")
        for path, err in errors:
            print(f"    {path}: {err}")

    return np.array(all_labels), np.array(all_probs)


# ─────────────────────────────────────────────────────
# Extended metrics
# ─────────────────────────────────────────────────────
def compute_metrics(labels, probs):
    """
    Returns a dict with:
      roc_auc, pr_auc, f1@best, best_threshold,
      fpr_at_95tpr, tpr_at_1fpr, eer, mcc, bal_acc,
      num_real, num_fake, total
    """
    fpr_arr, tpr_arr, thr_arr = roc_curve(labels, probs)

    # ── ROC-AUC & PR-AUC ────────────────────────────
    roc_auc = roc_auc_score(labels, probs)
    pr_auc  = average_precision_score(labels, probs)

    # ── FPR @ 95 % TPR ──────────────────────────────
    idx_95 = np.searchsorted(tpr_arr, 0.95)
    idx_95 = min(idx_95, len(fpr_arr) - 1)
    fpr_at_95tpr = float(fpr_arr[idx_95])

    # ── TPR @ 1 % FPR ───────────────────────────────
    idx_1fpr = np.searchsorted(fpr_arr, 0.01)
    idx_1fpr = min(idx_1fpr, len(tpr_arr) - 1)
    tpr_at_1fpr = float(tpr_arr[idx_1fpr])

    # ── EER  (Equal Error Rate) ──────────────────────
    fnr_arr = 1.0 - tpr_arr
    eer_idx = np.argmin(np.abs(fpr_arr - fnr_arr))
    eer = float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2.0)

    # ── Best threshold (maximises macro F1) ──────────
    thresholds = np.linspace(0.0, 1.0, 201)
    best_f1, best_thr = -1.0, 0.5
    for t in thresholds:
        f1 = f1_score(labels, (probs > t).astype(int),
                      average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    preds_best = (probs > best_thr).astype(int)

    # ── MCC ─────────────────────────────────────────
    mcc = float(matthews_corrcoef(labels, preds_best))

    # ── Balanced Accuracy ────────────────────────────
    tp = int(((preds_best == 1) & (labels == 1)).sum())
    tn = int(((preds_best == 0) & (labels == 0)).sum())
    fp = int(((preds_best == 1) & (labels == 0)).sum())
    fn = int(((preds_best == 0) & (labels == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (sens + spec) / 2.0

    return {
        "roc_auc":        roc_auc,
        "pr_auc":         pr_auc,
        "f1@best":        best_f1,
        "best_threshold": best_thr,
        "fpr_at_95tpr":   fpr_at_95tpr,
        "tpr_at_1fpr":    tpr_at_1fpr,
        "eer":            eer,
        "mcc":            mcc,
        "bal_acc":        bal_acc,
        "num_real":       int((labels == 0).sum()),
        "num_fake":       int((labels == 1).sum()),
        "total":          len(labels),
    }


# ─────────────────────────────────────────────────────
# Plots — all models on shared axes
# ─────────────────────────────────────────────────────
def save_curves(results: dict, out_dir: Path):
    """
    results: { model_name: {"labels": ..., "probs": ..., "metrics": ...} }
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, data in results.items():
        lbl   = data["labels"]
        probs = data["probs"]
        roc   = data["metrics"]["roc_auc"]
        pr    = data["metrics"]["pr_auc"]

        RocCurveDisplay.from_predictions(
            lbl, probs, ax=axes[0], name=f"{name} (AUC={roc:.3f})"
        )
        PrecisionRecallDisplay.from_predictions(
            lbl, probs, ax=axes[1], name=f"{name} (AP={pr:.3f})"
        )

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_title("ROC Curves — All Models")
    axes[1].set_title("Precision-Recall Curves — All Models")

    for ax in axes:
        ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    out_path = out_dir / "curves_all_models.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n[INFO] Curves saved → {out_path}")


# ─────────────────────────────────────────────────────
# Pretty-print comparison table
# ─────────────────────────────────────────────────────
METRIC_COLS = [
    ("roc_auc",      "ROC"),
    ("pr_auc",       "PR"),
    ("f1@best",      "F1"),
    ("fpr_at_95tpr", "FPR@95"),
    ("tpr_at_1fpr",  "TPR@1%"),
    ("eer",          "EER"),
    ("mcc",          "MCC"),
    ("bal_acc",      "BalAcc"),
]

def print_table(all_metrics: dict, test_folder: str):
    col_w    = 9          # width of each metric column
    name_w   = max(len(n) for n in all_metrics) + 2

    header_metric = "  ".join(f"{hdr:>{col_w}}" for _, hdr in METRIC_COLS)
    sep           = "─" * (name_w + 2 + len(header_metric))

    print("\n" + "═" * len(sep))
    print("  CROSS-DATASET EVALUATION — ALL MODELS")
    print("═" * len(sep))
    print(f"  Test folder : {test_folder}")
    first = next(iter(all_metrics.values()))
    print(f"  Samples     : {first['total']}  "
          f"(real: {first['num_real']}, fake: {first['num_fake']})")
    print()

    # header row
    print(f"  {'Model':<{name_w}}  {header_metric}")
    print("  " + sep)

    for name, m in all_metrics.items():
        row = "  ".join(f"{m[key]:>{col_w}.4f}" for key, _ in METRIC_COLS)
        print(f"  {name:<{name_w}}  {row}")

    # ── LaTeX table ──────────────────────────────────
    print(f"\n[LaTeX rows]")
    for name, m in all_metrics.items():
        vals = " & ".join(f"{m[key]:.4f}" for key, _ in METRIC_COLS)
        print(f"  {name} & {vals} \\\\")

    print("═" * len(sep))


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device      : {device}")
    print(f"[INFO] Test folder : {TEST_FOLDER}")
    print(f"[INFO] Models      : {list(CHECKPOINTS.keys())}\n")

    # ── collect samples once ─────────────────────────
    samples = collect_samples(TEST_FOLDER)
    if not samples:
        print("[ERROR] No .mp4 files found. Check TEST_FOLDER structure.")
        return

    real_count = sum(1 for _, l in samples if l == 0)
    fake_count = sum(1 for _, l in samples if l == 1)
    print(f"[INFO] Samples     : {len(samples)}  (real: {real_count}, fake: {fake_count})\n")

    out_dir = Path("eval_output")
    out_dir.mkdir(exist_ok=True)

    # ── evaluate each checkpoint ─────────────────────
    all_metrics = {}   # name → metrics dict
    curve_data  = {}   # name → {labels, probs, metrics}

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"[{'─'*56}]")
        print(f"  Model      : {name}")
        print(f"  Checkpoint : {ckpt_path}")

        if not Path(ckpt_path).exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}\n")
            continue

        # instantiate correct model class
        if name not in MODEL_REGISTRY:
            print(f"  [SKIP] No model class registered for '{name}'\n")
            continue

        model = MODEL_REGISTRY[name]()
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        print(f"  Loaded ✅")

        labels, probs = run_inference(model, samples, device)

        metrics = compute_metrics(labels, probs)
        all_metrics[name] = metrics
        curve_data[name]  = {"labels": labels, "probs": probs, "metrics": metrics}

        # free GPU memory between runs
        model.cpu()
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
        print()

    if not all_metrics:
        print("[ERROR] No models were successfully evaluated.")
        return

    # ── print combined table ─────────────────────────
    print_table(all_metrics, TEST_FOLDER)

    # ── save ROC + PR curves (all models) ────────────
    save_curves(curve_data, out_dir)


if __name__ == "__main__":
    main()