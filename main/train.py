import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
import subprocess

from models import (
    DeepfakeEdgeModel_GRU, DeepfakeEdgeModel_STSM, DeepfakeEdgeModel_HTSM, 
    DeepfakeEdgeModel_Mobile, DeepfakeEdgeModel_Xception, DeepfakeEdgeModel_Residual_HTSM
)

import warnings
warnings.filterwarnings("ignore")

# ==============================
# Reproducibility
# ==============================
SEED = 42
# SEED = 69
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==============================
# FAST DATASET (PREPROCESSED)
# ==============================
class FastDeepfakeDataset(Dataset):
    def __init__(self, processed_root):
        self.samples = []
        
        for root, _, files in os.walk(processed_root):
            for f in files:
                if f.endswith(".pt"):
                    self.samples.append(os.path.join(root, f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx])
        return data["video"], data["label"]

def build_dataloaders(
    processed_root,
    batch_size=16,
    num_workers=8,
    val_frac=0.15,
    test_frac=0.10,
):
    full_ds = FastDeepfakeDataset(processed_root)

    n = len(full_ds)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    def extract_labels_from_indices(dataset, indices):
        labels = []
        for i in indices:
            path = dataset.samples[i]   # path string
            data = torch.load(path)
            labels.append(data["label"])
        return torch.tensor(labels)

    # ===== LOG DATASET =====
    def log_dataset(ds, name):
        labels = extract_labels_from_indices(full_ds, ds.indices)
        num_real = (labels == 0).sum().item()
        num_fake = (labels == 1).sum().item()

        print(f"[{name}] {len(ds)} samples | real: {num_real}, fake: {num_fake}")

    log_dataset(train_ds, "TRAIN")
    log_dataset(val_ds, "VAL")
    log_dataset(test_ds, "TEST")

    # ===== BUILD WEIGHTED SAMPLER (TRAIN ONLY) =====
    train_labels = extract_labels_from_indices(full_ds, train_ds.indices)

    train_labels = torch.tensor(train_labels).long()

    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()


    # weight per sample
    sample_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        **kw
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **kw
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **kw
    )

    return train_loader, val_loader, test_loader

# ==============================
# TRAINING
# ==============================
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)

            with torch.amp.autocast(device_type=device.type):
                logits = model(x)
                y = y.view(-1, 1).float()
                loss = criterion(logits, y)

            total_loss += loss.item()

            probs = torch.sigmoid(logits).view(-1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.view(-1).cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ----------------------------
    # Metrics @ 0.5 threshold
    # ----------------------------
    preds_05 = (all_probs > 0.5).astype(int)

    precision_05 = precision_score(all_labels, preds_05, zero_division=0)
    recall_05 = recall_score(all_labels, preds_05, zero_division=0)
    f1_05 = f1_score(all_labels, preds_05, average='macro')

    # ----------------------------
    # Find best threshold (maximize macro F1)
    # ----------------------------
    thresholds = np.linspace(0.0, 1.0, 101)
    best_f1 = -1
    best_threshold = 0.5

    for t in thresholds:
        preds = (all_probs > t).astype(int)
        f1 = f1_score(all_labels, preds, average='macro')

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    # Metrics at best threshold
    preds_best = (all_probs > best_threshold).astype(int)

    precision_best = precision_score(all_labels, preds_best, zero_division=0)
    recall_best = recall_score(all_labels, preds_best, zero_division=0)
    f1_best = f1_score(all_labels, preds_best, average='macro')

    # ----------------------------
    # AUC metrics (threshold-independent)
    # ----------------------------
    if len(set(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
    else:
        roc_auc, pr_auc = 0.0, 0.0

    return total_loss / len(loader), {
        "precision@0.5": precision_05,
        "recall@0.5": recall_05,
        "f1@0.5": f1_05,
        "precision@best": precision_best,
        "recall@best": recall_best,
        "f1@best": f1_best,
        "best_threshold": best_threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def train(model, train_loader, val_loader, device, epochs=20, criterion=None, frac=""):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=5e-4
    )

    scaler = torch.amp.GradScaler()

    best_val_pr_auc = -1.0
    patience = 5
    counter = 0
    min_delta = 1e-3

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader)

        for x, y in pbar:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            y = y.view(-1, 1).float()

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            pbar.set_description(f"Loss: {loss.item():.4f}")

        # ---- Validation ----
        val_loss, metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {val_loss:.4f} | "
            f"P@0.5: {metrics['precision@0.5']:.4f} | "
            f"R@0.5: {metrics['recall@0.5']:.4f} | "
            f"F1@0.5: {metrics['f1@0.5']:.4f} | "
            f"F1@best: {metrics['f1@best']:.4f} | "
            f"BestThr: {metrics['best_threshold']:.2f} | "
            f"PR-AUC: {metrics['pr_auc']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )

        # ---- Early stopping on PR-AUC ----
        val_pr_auc = metrics["pr_auc"]

        if val_pr_auc > best_val_pr_auc + min_delta:
            best_val_pr_auc = val_pr_auc
            counter = 0
            torch.save(model.state_dict(), f"best_model{frac}.pth")
            print("✅ Best model saved (PR-AUC improved)")
        else:
            counter += 1
            print(f"⚠️ No PR-AUC improvement ({counter}/{patience})")

        if counter >= patience:
            print("🛑 Early stopping triggered")
            break
        
# ==============================
# MAIN
# ==============================

def get_free_gpu():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"]
    )
    memory_free = [int(x) for x in result.decode("utf-8").strip().split("\n")]
    return int(np.argmax(memory_free))



def main():
    PROCESSED_ROOT = "../processed_sampler20_6000"
    BATCH_SIZE = 32 # 128 when using mobilenet, 32 when using xception
    EPOCHS = 50
    NUM_WORKERS = 16

    criterion = nn.BCEWithLogitsLoss()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        gpu_id = get_free_gpu()
        print(f"Using GPU {gpu_id}")
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    train_loader, val_loader, test_loader = build_dataloaders(PROCESSED_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = DeepfakeEdgeModel_Residual_HTSM().to(device)

    model_path = "best_model.pth"

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded model weights from model path")
    else:
        print("model path not found, training from scratch")


    train(model, train_loader, val_loader, device, epochs=EPOCHS, criterion=criterion)

    test_loss, metrics = evaluate(model, test_loader, criterion, device)

    print("\n===== Test Results =====")
    print(
        f"P@0.5: {metrics['precision@0.5']:.4f} | "
        f"R@0.5: {metrics['recall@0.5']:.4f} | "
        f"F1@0.5: {metrics['f1@0.5']:.4f} | "
        f"F1@best: {metrics['f1@best']:.4f} | "
        f"BestThr: {metrics['best_threshold']:.2f} | "
        f"PR-AUC: {metrics['pr_auc']:.4f} | "
        f"ROC-AUC: {metrics['roc_auc']:.4f}"
    )

if __name__ == "__main__":
    main()