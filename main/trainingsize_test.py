import matplotlib.pyplot as plt
from torch.utils.data import Subset
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from train import build_dataloaders, train, evaluate, get_free_gpu
from models import DeepfakeEdgeModel_Residual_HTSM

import warnings
warnings.filterwarnings("ignore")


def train_with_varying_data_sizes(
    processed_root,
    device,
    batch_size=128,
    num_workers=6,
    epochs=50,
):
    train_loader_full, val_loader, test_loader = build_dataloaders(
        processed_root,
        batch_size=batch_size,
        num_workers=num_workers
    )

    full_dataset = train_loader_full.dataset

    fractions = [0.05, 0.1, 0.25, 0.5, 1.0]

    sizes = []
    pr_aucs = []
    roc_aucs = []
    class_distributions = []

    for frac in fractions:
        print(f"\n===== Training with {int(frac*100)}% data =====")

        subset_size = int(len(full_dataset) * frac)
        indices = np.random.choice(len(full_dataset), subset_size, replace=False)

        subset = Subset(full_dataset, indices)

        # -----------------------------
        # Extract labels
        # -----------------------------
        labels = []
        for i in indices:
            path = full_dataset.dataset.samples[full_dataset.indices[i]]
            data = torch.load(path)
            labels.append(data["label"])

        labels = torch.tensor(labels).long()
        class_counts = torch.bincount(labels)

        if len(class_counts) < 2:
            print("⚠️ Only one class in subset — skipping")
            continue

        # -----------------------------
        # Detect imbalance
        # -----------------------------
        majority = torch.max(class_counts).item()
        minority = torch.min(class_counts).item()
        imbalance_ratio = majority / (minority + 1e-6)

        print(f"Before resampling: {class_counts.tolist()} | Ratio: {imbalance_ratio:.2f}")

        # -----------------------------
        # Conditional resampling
        # -----------------------------
        RESAMPLE_THRESHOLD = 2.0

        if imbalance_ratio > RESAMPLE_THRESHOLD:
            print("⚠️ Applying oversampling")

            class0_idx = [i for i, l in enumerate(labels) if l == 0]
            class1_idx = [i for i, l in enumerate(labels) if l == 1]

            if len(class0_idx) > len(class1_idx):
                majority_idx, minority_idx = class0_idx, class1_idx
            else:
                majority_idx, minority_idx = class1_idx, class0_idx

            extra_needed = len(majority_idx) - len(minority_idx)
            sampled_minority = np.random.choice(minority_idx, extra_needed, replace=True)

            new_indices = np.concatenate([np.arange(len(labels)), sampled_minority])

            subset = Subset(subset, new_indices)
            labels = labels[new_indices]

            class_counts = torch.bincount(labels)
            print(f"After resampling: {class_counts.tolist()}")

        # Store class distribution
        class_distributions.append(class_counts.tolist())

        # -----------------------------
        # Weighted sampler
        # -----------------------------
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(
            subset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        # -----------------------------
        # Model
        # -----------------------------
        model = DeepfakeEdgeModel_Residual_HTSM().to(device)

        # Optional: class-weighted loss
        pos_weight = class_weights[1].to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train(
            model,
            train_loader,
            val_loader,
            device,
            epochs=epochs,
            criterion=criterion,
            frac=str(frac)
        )

        # -----------------------------
        # Evaluate
        # -----------------------------
        _, metrics = evaluate(model, test_loader, criterion, device)

        print(f"PR-AUC: {metrics['pr_auc']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

        sizes.append(len(labels))
        pr_aucs.append(metrics["pr_auc"])
        roc_aucs.append(metrics["roc_auc"])

        del model
        torch.cuda.empty_cache()

    # -----------------------------
    # Plotting
    # -----------------------------
    labels_for_legend = [
        f"{s} (C0:{cd[0]}, C1:{cd[1]})"
        for s, cd in zip(sizes, class_distributions)
    ]

    # PR-AUC Plot
    plt.figure()
    for i in range(len(sizes)):
        plt.scatter(sizes[i], pr_aucs[i], label=labels_for_legend[i])

    plt.plot(sizes, pr_aucs)
    plt.xlabel("Training Dataset Size")
    plt.ylabel("PR-AUC")
    plt.title("PR-AUC vs Dataset Size")
    plt.legend(fontsize=8)
    plt.grid()
    plt.savefig("pr_auc_vs_size.png")
    plt.close()

    # ROC-AUC Plot
    plt.figure()
    for i in range(len(sizes)):
        plt.scatter(sizes[i], roc_aucs[i], label=labels_for_legend[i])

    plt.plot(sizes, roc_aucs)
    plt.xlabel("Training Dataset Size")
    plt.ylabel("ROC-AUC")
    plt.title("ROC-AUC vs Dataset Size")
    plt.legend(fontsize=8)
    plt.grid()
    plt.savefig("roc_auc_vs_size.png")
    plt.close()

    print("\nSaved plots: pr_auc_vs_size.png, roc_auc_vs_size.png")

    return sizes, pr_aucs, roc_aucs


def main():
    PROCESSED_ROOT = "../processed_sampler20_6000"

    if torch.cuda.is_available():
        gpu_id = get_free_gpu()
        print(f"Using GPU {gpu_id}")
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    sizes, pr_aucs, roc_aucs = train_with_varying_data_sizes(
        PROCESSED_ROOT,
        device,
        batch_size=128,
        epochs=50
    )

    print("\nFinal Results:")
    for s, p, r in zip(sizes, pr_aucs, roc_aucs):
        print(f"Size: {s} | PR-AUC: {p:.4f} | ROC-AUC: {r:.4f}")


if __name__ == "__main__":
    main()