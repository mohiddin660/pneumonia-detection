"""
Full evaluation on the held-out test set.

Loads models/best.pt and evaluates on data/splits/test.csv.

Generates:
    reports/confusion_matrix.png         — binary confusion matrix
    reports/roc_curve.png                — ROC curve
    reports/precision_recall_curve.png   — PR curve
    reports/stage_confusion_matrix.png   — 3x3 stage confusion
    reports/classification_report.txt    — text report
    reports/test_metrics.json            — machine-readable metrics
    reports/predictions.csv              — per-sample predictions

Run:  python -m scripts.evaluate
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
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import RSNAPneumoniaDataset
from src.data.transforms import get_eval_transforms
from src.models.classifier import PneumoniaClassifier
from src.utils.logger import get_logger

log = get_logger("evaluate", "logs/evaluate.log")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)

# ---------------- CONFIG ----------------
TEST_CSV = "data/splits/test.csv"
CHECKPOINT = "models/best.pt"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0      # Windows-safe

LABEL_NAMES = ["NORMAL", "PNEUMONIA"]
STAGE_NAMES = ["MILD", "MODERATE", "SEVERE"]


# ---------------- MAIN ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 70)
    log.info(f"Evaluating on device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info("=" * 70)

    # ---- Data ----
    log.info(f"Loading test set: {TEST_CSV}")
    test_ds = RSNAPneumoniaDataset(TEST_CSV, transform=get_eval_transforms(IMG_SIZE))
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    log.info(f"Test samples: {len(test_ds)}")

    # ---- Model ----
    log.info(f"Loading model: {CHECKPOINT}")
    model = PneumoniaClassifier(multi_task=True).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info(f"  Trained epoch: {ckpt.get('epoch', '?')}")
    log.info(f"  Best val AUC:  {ckpt.get('best_auc', 0):.4f}")

    # ---- Inference ----
    log.info("Running inference on test set...")
    all_binary_true, all_binary_pred, all_binary_prob = [], [], []
    all_stage_true, all_stage_pred = [], []
    all_patient_ids = []

    with torch.no_grad():
        for imgs, binary_labels, stage_labels in tqdm(test_loader, desc="Test"):
            imgs = imgs.to(device, non_blocking=True)
            binary_labels = binary_labels.to(device, non_blocking=True)
            stage_labels = stage_labels.to(device, non_blocking=True)

            binary_logits, stage_logits = model(imgs)

            binary_probs = F.softmax(binary_logits, dim=1)[:, 1].cpu().numpy()
            binary_preds = binary_logits.argmax(dim=1).cpu().numpy()
            stage_preds = stage_logits.argmax(dim=1).cpu().numpy()

            all_binary_true.extend(binary_labels.cpu().numpy().tolist())
            all_binary_pred.extend(binary_preds.tolist())
            all_binary_prob.extend(binary_probs.tolist())

            # Stage metrics only on pneumonia samples
            mask = binary_labels == 1
            if mask.any():
                all_stage_true.extend(stage_labels[mask].cpu().numpy().tolist())
                all_stage_pred.extend(stage_preds[mask.cpu().numpy()].tolist())

    y_true = np.array(all_binary_true)
    y_pred = np.array(all_binary_pred)
    y_prob = np.array(all_binary_prob)

    # ---- Metrics ----
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "n_samples": int(len(y_true)),
        "n_normal": int((y_true == 0).sum()),
        "n_pneumonia": int((y_true == 1).sum()),
    }

    # ---- Print report ----
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION — TEST SET RESULTS")
    print("=" * 70)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:12s}: {v:.4f}")
        else:
            print(f"  {k:12s}: {v}")

    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4)
    print("\n" + report)

    # Save report
    with open(OUT / "classification_report.txt", "w") as f:
        f.write("BINARY CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n" + report)

    # Save JSON metrics
    with open(OUT / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        annot_kws={"size": 14},
    )
    plt.title(f"Confusion Matrix — Test Set (n={len(y_true)})", fontsize=13, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"[OK] Saved {OUT / 'confusion_matrix.png'}")

    # ---- ROC Curve ----
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = metrics["auc"]

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"Model (AUC = {auc:.4f})", lw=2.5, color="darkorange")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1.5, label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve — Test Set", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=150)
    plt.close()
    print(f"[OK] Saved {OUT / 'roc_curve.png'}")

    # ---- Precision-Recall Curve ----
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(recalls, precisions, lw=2.5, color="steelblue")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve — Test Set", fontsize=13, fontweight="bold")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "precision_recall_curve.png", dpi=150)
    plt.close()
    print(f"[OK] Saved {OUT / 'precision_recall_curve.png'}")

    # ---- Stage Metrics ----
    if all_stage_true:
        y_stage_true = np.array(all_stage_true)
        y_stage_pred = np.array(all_stage_pred)

        stage_acc = accuracy_score(y_stage_true, y_stage_pred)
        stage_f1 = f1_score(y_stage_true, y_stage_pred, average="macro", zero_division=0)

        print("\n" + "=" * 70)
        print("STAGE CLASSIFICATION — TEST SET RESULTS (pneumonia only)")
        print("=" * 70)
        print(f"  Stage accuracy: {stage_acc:.4f}")
        print(f"  Stage F1 (macro): {stage_f1:.4f}")
        print()
        print(classification_report(
            y_stage_true, y_stage_pred,
            target_names=STAGE_NAMES, digits=4, zero_division=0,
        ))

        metrics["stage_accuracy"] = float(stage_acc)
        metrics["stage_f1_macro"] = float(stage_f1)
        metrics["n_stage_samples"] = int(len(y_stage_true))

        # Save updated JSON
        with open(OUT / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Stage confusion matrix
        cm_stage = confusion_matrix(y_stage_true, y_stage_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_stage, annot=True, fmt="d", cmap="Oranges", cbar=False,
            xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
            annot_kws={"size": 14},
        )
        plt.title(f"Stage Confusion Matrix (n={len(y_stage_true)})",
                  fontsize=13, fontweight="bold")
        plt.ylabel("True Stage")
        plt.xlabel("Predicted Stage")
        plt.tight_layout()
        plt.savefig(OUT / "stage_confusion_matrix.png", dpi=150)
        plt.close()
        print(f"[OK] Saved {OUT / 'stage_confusion_matrix.png'}")

    # ---- Save per-sample predictions ----
    import pandas as pd
    df = test_ds.df.copy()
    df["pred_label"] = y_pred
    df["pred_prob_pneumonia"] = y_prob
    df["correct"] = (y_true == y_pred).astype(int)
    df.to_csv(OUT / "predictions.csv", index=False)
    print(f"[OK] Saved {OUT / 'predictions.csv'}")

    # ---- Final Summary ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Binary AUC:        {metrics['auc']:.4f}")
    print(f"  Binary Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Binary F1:         {metrics['f1']:.4f}")
    print(f"  Binary Precision:  {metrics['precision']:.4f}")
    print(f"  Binary Recall:     {metrics['recall']:.4f}")
    if "stage_accuracy" in metrics:
        print(f"  Stage Accuracy:    {metrics['stage_accuracy']:.4f}")
        print(f"  Stage F1 (macro):  {metrics['stage_f1_macro']:.4f}")
    print("=" * 70)
    print(f"\nAll reports saved to: {OUT}/")


if __name__ == "__main__":
    main()