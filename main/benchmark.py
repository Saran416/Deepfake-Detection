"""
benchmark_inference.py
======================
Research-grade inference benchmarking for deepfake detection models.

Measures:
  - Inference latency (mean ± std, ms)  [CPU-only, single-core, no AMP]
  - Energy consumption via CodeCarbon  (kWh, CO₂eq g)
  - Model size  (disk MB + parameter count)

Standards followed:
  - CPU-only, single-threaded (OMP/MKL threads = 1)
  - Warm-up runs excluded from timing
  - Multiple timed runs → mean ± std (CI-ready)
  - torch.no_grad() + eval() throughout
  - All numbers printed in a LaTeX-ready table

Usage:
  python benchmark_inference.py \
      --checkpoint best_model.pth \
      --video_dir /path/to/test_videos \
      [--num_frames 20] \
      [--num_regions 4] \
      [--warmup_runs 10] \
      [--timed_runs 50] \
      [--output_json results.json]
"""

# ── force single-core BEFORE importing torch/numpy ──────────────────────────
import os
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"]  = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]   = "1"

import argparse
import json
import time
import warnings
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2
from decord import VideoReader, cpu as decord_cpu

warnings.filterwarnings("ignore")

# ── try importing codecarbon ─────────────────────────────────────────────────
try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False
    print("[WARNING] codecarbon not found. Install with: pip install codecarbon")
    print("          Energy metrics will be skipped.\n")

# ── local model import ────────────────────────────────────────────────────────

from models import (
    DeepfakeEdgeModel_GRU, DeepfakeEdgeModel_STSM, DeepfakeEdgeModel_HTSM, 
    DeepfakeEdgeModel_Mobile, DeepfakeEdgeModel_Xception, DeepfakeEdgeModel_Residual_HTSM
)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing  (mirrors preprocess.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
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


def preprocess_video(video_path: str, num_frames: int = 20, num_regions: int = 4) -> torch.Tensor:
    """
    Replicate the exact preprocessing from preprocess.py:
      - Uniform temporal sampling across N regions
      - Haar-cascade face detection (once per region)
      - Resize to 224×224
      - ImageNet normalisation
    Returns: (T, 3, 224, 224) float32 tensor  (no batch dim)
    """
    face_cascade = _get_face_detector()
    vr = VideoReader(video_path, ctx=decord_cpu(0))
    total = len(vr)
    total_samples = num_regions * (num_frames // num_regions)
    indices = np.linspace(0, total - 1, total_samples, dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) uint8

    processed = []
    last_box = None
    region_size = num_frames // num_regions

    for i, f in enumerate(frames):
        if i % region_size == 0:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
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


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    model = DeepfakeEdgeModel_Xception()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def model_disk_size_mb(checkpoint_path: str) -> float:
    return os.path.getsize(checkpoint_path) / (1024 ** 2)


def model_param_count(model: nn.Module):
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────
def time_inference(model: nn.Module, tensor: torch.Tensor,
                   warmup: int = 10, runs: int = 50) -> dict:
    """
    Returns mean ± std latency in milliseconds over `runs` timed runs.
    `warmup` runs are discarded first.
    CPU-only, no AMP, no torch.compile.
    """
    x = tensor.unsqueeze(0)   # (1, T, 3, 224, 224)

    with torch.no_grad():
        # warm-up (fills cache, JIT paths, etc.)
        for _ in range(warmup):
            _ = model(x)

        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1_000)   # → ms

    arr = np.array(latencies)
    return {
        "mean_ms":   float(arr.mean()),
        "std_ms":    float(arr.std(ddof=1)),
        "min_ms":    float(arr.min()),
        "max_ms":    float(arr.max()),
        "median_ms": float(np.median(arr)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "runs":      runs,
        "warmup":    warmup,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Energy  (CodeCarbon)
# ─────────────────────────────────────────────────────────────────────────────
def measure_energy(model: nn.Module, tensor: torch.Tensor, runs: int = 50) -> dict:
    """
    Uses CodeCarbon to track CPU energy during `runs` inference calls.
    Returns kWh consumed and estimated CO₂eq in grams.
    """
    if not HAS_CODECARBON:
        return {"kwh": None, "co2_g": None, "note": "codecarbon not installed"}

    x = tensor.unsqueeze(0)

    tracker = EmissionsTracker(
        log_level="error",
        save_to_file=False,
        tracking_mode="process",   # CPU process-level tracking
        gpu_ids=[],                # explicitly no GPU
    )

    tracker.start()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(x)
    emissions = tracker.stop()    # kg CO₂eq

    # CodeCarbon exposes energy_consumed (kWh) via tracker._total_energy
    try:
        kwh = float(tracker._total_energy.kWh)
    except Exception:
        kwh = None

    return {
        "kwh":        kwh,
        "co2_g":      float(emissions) * 1_000 if emissions else None,   # kg→g
        "runs":       runs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing / LaTeX table
# ─────────────────────────────────────────────────────────────────────────────
HLINE = "─" * 60

def print_results(results: dict):
    print("\n" + "═" * 60)
    print("  INFERENCE BENCHMARK RESULTS  (research-grade)")
    print("═" * 60)

    print("\n[Environment]")
    print(f"  Device             : {results['environment']['device']}")
    print(f"  CPU threads        : {results['environment']['cpu_threads']}  (forced single-core)")
    print(f"  AMP / GPU accel.   : disabled")
    print(f"  torch.no_grad()    : enabled")
    print(f"  PyTorch version    : {results['environment']['torch_version']}")
    print(f"  Videos benchmarked : {results['environment']['num_videos']}")

    print(f"\n[Model Size]")
    print(f"  Checkpoint file    : {results['model_size']['disk_mb']:.2f} MB")
    print(f"  Total parameters   : {results['model_size']['total_params']:,}")
    print(f"  Trainable params   : {results['model_size']['trainable_params']:,}")

    print(f"\n[Inference Latency  —  per video, CPU, single-core]")
    lat = results['latency']
    print(f"  Mean ± Std         : {lat['mean_ms']:.2f} ± {lat['std_ms']:.2f} ms")
    print(f"  Median             : {lat['median_ms']:.2f} ms")
    print(f"  Min / Max          : {lat['min_ms']:.2f} / {lat['max_ms']:.2f} ms")
    print(f"  P95                : {lat['p95_ms']:.2f} ms")
    print(f"  Warm-up runs       : {lat['warmup']}  |  Timed runs: {lat['runs']}")

    print(f"\n[Energy Consumption  (CodeCarbon)]")
    en = results.get("energy", {})
    if en.get("kwh") is not None:
        print(f"  Total energy       : {en['kwh']*1e6:.4f} µWh  ({en['kwh']:.6e} kWh)")
        print(f"  Per inference      : {en['kwh']/en['runs']*1e9:.4f} nWh")
        if en.get("co2_g") is not None:
            print(f"  CO₂ equivalent     : {en['co2_g']:.6f} g  ({en['co2_g']/en['runs']*1e6:.4f} µg / inference)")
    else:
        print(f"  {en.get('note', 'N/A')}")

    # ── LaTeX table snippet ──────────────────────────────────────────────────
    print(f"\n[LaTeX Table Snippet]")
    lat_str = f"${lat['mean_ms']:.1f} \\pm {lat['std_ms']:.1f}$"
    size_str = f"{results['model_size']['disk_mb']:.1f}"
    params_str = f"{results['model_size']['total_params']/1e6:.2f}M"
    en_str = (
        f"{en['kwh']/en['runs']*1e9:.2f}" if en.get("kwh") else "N/A"
    )
    print("\\begin{table}[h]")
    print("  \\centering")
    print("  \\begin{tabular}{lcccc}")
    print("    \\toprule")
    print("    Model & Latency (ms) & Size (MB) & \\#Params & Energy (nWh/inf) \\\\")
    print("    \\midrule")
    print(f"    Ours & {lat_str} & {size_str} & {params_str} & {en_str} \\\\")
    print("    \\bottomrule")
    print("  \\end{tabular}")
    print("  \\caption{Inference benchmarks (CPU, single-core, no GPU acceleration).}")
    print("\\end{table}")

    print("\n" + "═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Research-grade inference benchmark for deepfake detection model."
    )
    p.add_argument("--checkpoint",   required=True,  help="Path to .pth checkpoint")
    p.add_argument("--video_dir",    required=True,  help="Folder containing .mp4 test videos")
    p.add_argument("--num_frames",   type=int, default=20)
    p.add_argument("--num_regions",  type=int, default=4)
    p.add_argument("--warmup_runs",  type=int, default=10,
                   help="Warm-up runs excluded from timing")
    p.add_argument("--timed_runs",   type=int, default=50,
                   help="Timed runs for latency statistics")
    p.add_argument("--energy_runs",  type=int, default=50,
                   help="Runs during energy measurement window")
    p.add_argument("--output_json",  default=None,
                   help="Optional path to save results as JSON")
    return p.parse_args()


def main():
    args = parse_args()

    # ── enforce single-core via torch ────────────────────────────────────────
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    device = torch.device("cpu")   # CPU-only: reproducible, no CUDA variability

    print(f"[INFO] Checkpoint   : {args.checkpoint}")
    print(f"[INFO] Video dir    : {args.video_dir}")
    print(f"[INFO] Device       : CPU  (single-core, no AMP, no GPU)")

    # ── collect videos ───────────────────────────────────────────────────────
    video_files = sorted(Path(args.video_dir).rglob("*.mp4"))
    if not video_files:
        print("[ERROR] No .mp4 files found in video_dir.")
        sys.exit(1)
    print(f"[INFO] Found {len(video_files)} video(s)")

    # ── load model ───────────────────────────────────────────────────────────
    print("[INFO] Loading model ...")
    model = load_model(args.checkpoint, device)
    total_params, trainable_params = model_param_count(model)
    disk_mb = model_disk_size_mb(args.checkpoint)

    # ── pick one representative video for timing & energy ───────────────────
    # (use the first video; latency is per-video, not per-dataset)
    rep_video = str(video_files[0])
    print(f"[INFO] Preprocessing representative video: {rep_video}")
    tensor = preprocess_video(rep_video, args.num_frames, args.num_regions)
    # tensor: (T, 3, 224, 224)

    # ── latency benchmark ────────────────────────────────────────────────────
    print(f"[INFO] Timing inference ({args.warmup_runs} warm-up + {args.timed_runs} timed runs) ...")
    latency = time_inference(model, tensor, warmup=args.warmup_runs, runs=args.timed_runs)

    # ── energy benchmark ─────────────────────────────────────────────────────
    print(f"[INFO] Measuring energy ({args.energy_runs} runs via CodeCarbon) ...")
    energy = measure_energy(model, tensor, runs=args.energy_runs)

    # ── assemble results ─────────────────────────────────────────────────────
    results = {
        "environment": {
            "device":        "CPU",
            "cpu_threads":   1,
            "amp_enabled":   False,
            "gpu_enabled":   False,
            "torch_version": torch.__version__,
            "num_videos":    len(video_files),
        },
        "model_size": {
            "disk_mb":         disk_mb,
            "total_params":    total_params,
            "trainable_params": trainable_params,
        },
        "latency":  latency,
        "energy":   energy,
    }

    print_results(results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Results saved to {args.output_json}")


if __name__ == "__main__":
    main()