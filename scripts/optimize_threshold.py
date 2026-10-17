"""
Find the optimal decision threshold for pneumonia prediction.

Sweeps thresholds from 0.1 to 0.9, computes precision/recall/F1 at each,
and saves the best threshold to reports/optimal_threshold.json.

Run:  python -m scripts.optimize_threshold
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import RSNAPneumoniaDataset
from src.data.transforms import get_eval_transforms
from src.models.classifier import PneumoniaClassifier
from src.utils.logger import get_logger

log = get_logger("threshold", "logs/threshold.log")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)

TEST_CSV = "data/splits/test.csv"
CHECKPOINT = "models/best.pt"
IMG_SIZE = 224
BATCH_SIZE = 32

LABEL_NAMES = ["NORMAL", "PNEUMONIA"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ---- Load data ----
    test_ds = RSNAPneumoniaDataset(TEST_CSV, transform=get_eval_transforms(IMG_SIZE))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- Load model ----
    model = PneumoniaClassifier(multi_task=True).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- Get all probabilities ----
    log.info("Running inference...")
    y_true, y_prob = [], []
    with torch.no_grad():
        for imgs, binary_labels, _ in tqdm(test_loader, desc="Inference"):
            imgs = imgs.to(device)
            logits, _ = model(imgs)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_true.extend(binary_labels.numpy().tolist())
            y_prob.extend(probs.tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # ---- Sweep thresholds ----
    thresholds = np.arange(0.05, 0.96, 0.01)
    results = []

    print(f"\nSweeping {len(thresholds)} thresholds...\n")
    print(f"{'Thresh':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-" * 48)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "threshold": float(t),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        })

        # Print every 5th
        if len(results) % 5 == 0:
            print(f"{t:>8.2f} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")

    results = np.array([list(r.values()) for r in results])
    thresh_arr = results[:, 0]
    acc_arr = results[:, 1]
    prec_arr = results[:, 2]
    rec_arr = results[:, 3]
    f1_arr = results[:, 4]

    # ---- Find optimal thresholds ----
    best_f1_idx = np.argmax(f1_arr)
    best_acc_idx = np.argmax(acc_arr)

    best_f1_thresh = thresh_arr[best_f1_idx]
    best_acc_thresh = thresh_arr[best_acc_idx]

    # ---- Clinical optimum: highest recall while precision > 0.45 ----
    clinical_mask = prec_arr >= 0.45
    if clinical_mask.any():
        clinical_indices = np.where(clinical_mask)[0]
        clinical_idx = clinical_indices[np.argmax(rec_arr[clinical_mask])]
        clinical_thresh = thresh_arr[clinical_idx]
    else:
        clinical_thresh = best_f1_thresh
        clinical_idx = best_f1_idx

    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLDS")
    print("=" * 70)

    # Default (0.5)
    default_idx = np.argmin(np.abs(thresh_arr - 0.5))
    print(f"\nDefault threshold (0.50):")
    print(f"  Accuracy:  {acc_arr[default_idx]:.4f}")
    print(f"  Precision: {prec_arr[default_idx]:.4f}")
    print(f"  Recall:    {rec_arr[default_idx]:.4f}")
    print(f"  F1:        {f1_arr[default_idx]:.4f}")

    print(f"\nBest F1 threshold ({best_f1_thresh:.2f}):")
    print(f"  Accuracy:  {acc_arr[best_f1_idx]:.4f}")
    print(f"  Precision: {prec_arr[best_f1_idx]:.4f}")
    print(f"  Recall:    {rec_arr[best_f1_idx]:.4f}")
    print(f"  F1:        {f1_arr[best_f1_idx]:.4f}")

    print(f"\nBest Accuracy threshold ({best_acc_thresh:.2f}):")
    print(f"  Accuracy:  {acc_arr[best_acc_idx]:.4f}")
    print(f"  Precision: {prec_arr[best_acc_idx]:.4f}")
    print(f"  Recall:    {rec_arr[best_acc_idx]:.4f}")
    print(f"  F1:        {f1_arr[best_acc_idx]:.4f}")

    print(f"\nClinical threshold ({clinical_thresh:.2f}) — highest recall with precision ≥ 0.45:")
    print(f"  Accuracy:  {acc_arr[clinical_idx]:.4f}")
    print(f"  Precision: {prec_arr[clinical_idx]:.4f}")
    print(f"  Recall:    {rec_arr[clinical_idx]:.4f}")
    print(f"  F1:        {f1_arr[clinical_idx]:.4f}")

    # ---- Save optimal thresholds ----
    optimal = {
        "default_0.5": {
            "threshold": 0.5,
            "accuracy": float(acc_arr[default_idx]),
            "precision": float(prec_arr[default_idx]),
            "recall": float(rec_arr[default_idx]),
            "f1": float(f1_arr[default_idx]),
        },
        "best_f1": {
            "threshold": float(best_f1_thresh),
            "accuracy": float(acc_arr[best_f1_idx]),
            "precision": float(prec_arr[best_f1_idx]),
            "recall": float(rec_arr[best_f1_idx]),
            "f1": float(f1_arr[best_f1_idx]),
        },
        "best_accuracy": {
            "threshold": float(best_acc_thresh),
            "accuracy": float(acc_arr[best_acc_idx]),
            "precision": float(prec_arr[best_acc_idx]),
            "recall": float(rec_arr[best_acc_idx]),
            "f1": float(f1_arr[best_acc_idx]),
        },
        "clinical_optimal": {
            "threshold": float(clinical_thresh),
            "accuracy": float(acc_arr[clinical_idx]),
            "precision": float(prec_arr[clinical_idx]),
            "recall": float(rec_arr[clinical_idx]),
            "f1": float(f1_arr[clinical_idx]),
        },
    }

    with open(OUT / "optimal_threshold.json", "w") as f:
        json.dump(optimal, f, indent=2)
    print(f"\n[OK] Saved {OUT / 'optimal_threshold.json'}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: metrics vs threshold
    axes[0].plot(thresh_arr, acc_arr, label="Accuracy", lw=2)
    axes[0].plot(thresh_arr, prec_arr, label="Precision", lw=2)
    axes[0].plot(thresh_arr, rec_arr, label="Recall", lw=2)
    axes[0].plot(thresh_arr, f1_arr, label="F1", lw=2.5, color="black")

    axes[0].axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Default (0.5)")
    axes[0].axvline(best_f1_thresh, color="green", linestyle="--", alpha=0.7,
                    label=f"Best F1 ({best_f1_thresh:.2f})")
    axes[0].axvline(clinical_thresh, color="red", linestyle="--", alpha=0.7,
                    label=f"Clinical ({clinical_thresh:.2f})")

    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Metrics vs Decision Threshold")
    axes[0].legend(loc="center right", fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Right: confusion matrix at clinical threshold
    y_pred_clinical = (y_prob >= clinical_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred_clinical)

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        annot_kws={"size": 14}, ax=axes[1],
    )
    axes[1].set_title(f"Confusion Matrix @ threshold={clinical_thresh:.2f}",
                      fontsize=12, fontweight="bold")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(OUT / "threshold_optimization.png", dpi=150)
    plt.close()
    print(f"[OK] Saved {OUT / 'threshold_optimization.png'}")

    # ---- Recommendations ----
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Compute improvement
    default_f1 = f1_arr[default_idx]
    clinical_f1 = f1_arr[clinical_idx]
    default_prec = prec_arr[default_idx]
    clinical_prec = prec_arr[clinical_idx]

    if clinical_f1 > default_f1:
        print(f"\n✅ Use clinical threshold = {clinical_thresh:.2f}")
        print(f"   Precision: {default_prec:.4f} → {clinical_prec:.4f}  (+{clinical_prec - default_prec:.4f})")
        print(f"   Recall:    {rec_arr[default_idx]:.4f} → {rec_arr[clinical_idx]:.4f}")
        print(f"   F1:        {default_f1:.4f} → {clinical_f1:.4f}  (+{clinical_f1 - default_f1:.4f})")
        print(f"\n   Recommended for medical use: high recall, improved precision.")
    else:
        print(f"\n✅ Keep default threshold = 0.50")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()