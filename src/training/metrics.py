"""
Metrics tracking for training and validation.

Tracks:
    - Binary: accuracy, F1, precision, recall, AUC
    - Stage:  per-stage accuracy (only on pneumonia samples)
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BinaryMetricsTracker:
    """Accumulates predictions over an epoch, then computes metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.y_prob = []

    @torch.no_grad()
    def update(self, logits, targets):
        """
        logits:  [B, 2]  raw model outputs
        targets: [B]     ground truth labels (0 or 1)
        """
        probs = torch.softmax(logits, dim=1)[:, 1]     # P(pneumonia)
        preds = logits.argmax(dim=1)

        self.y_true.extend(targets.cpu().numpy().tolist())
        self.y_pred.extend(preds.cpu().numpy().tolist())
        self.y_prob.extend(probs.cpu().numpy().tolist())

    def compute(self) -> dict:
        y = np.array(self.y_true)
        p = np.array(self.y_pred)
        pr = np.array(self.y_prob)

        metrics = {
            "accuracy": accuracy_score(y, p),
            "f1": f1_score(y, p, zero_division=0),
            "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0),
        }

        # AUC only makes sense if both classes are present
        if len(np.unique(y)) > 1:
            metrics["auc"] = roc_auc_score(y, pr)
        else:
            metrics["auc"] = 0.0

        return metrics


class StageMetricsTracker:
    """Tracks stage prediction accuracy on pneumonia samples only."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    @torch.no_grad()
    def update(self, stage_logits, stage_targets, pneumonia_mask):
        """
        stage_logits:    [B, 3]
        stage_targets:   [B]
        pneumonia_mask:  [B] bool tensor — True for pneumonia samples
        """
        if not pneumonia_mask.any():
            return

        preds = stage_logits[pneumonia_mask].argmax(dim=1)
        targets = stage_targets[pneumonia_mask]

        self.y_true.extend(targets.cpu().numpy().tolist())
        self.y_pred.extend(preds.cpu().numpy().tolist())

    def compute(self) -> dict:
        if len(self.y_true) == 0:
            return {"stage_accuracy": 0.0, "stage_f1": 0.0}

        y = np.array(self.y_true)
        p = np.array(self.y_pred)

        return {
            "stage_accuracy": accuracy_score(y, p),
            "stage_f1": f1_score(y, p, average="macro", zero_division=0),
        }


# ---------------- QUICK TEST ----------------
if __name__ == "__main__":
    print("Testing metrics trackers...")

    # ---- Binary ----
    print("\n1. Binary Metrics test:")
    tracker = BinaryMetricsTracker()

    # Simulate 2 batches
    batch1_logits = torch.tensor([[2.0, 0.5], [0.5, 2.0], [3.0, 0.1], [1.0, 0.2]])
    batch1_targets = torch.tensor([0, 1, 0, 0])
    tracker.update(batch1_logits, batch1_targets)

    batch2_logits = torch.tensor([[0.1, 3.0], [2.5, 0.2]])
    batch2_targets = torch.tensor([1, 0])
    tracker.update(batch2_logits, batch2_targets)

    metrics = tracker.compute()
    for k, v in metrics.items():
        print(f"   {k:10s}: {v:.4f}")

    # ---- Stage ----
    print("\n2. Stage Metrics test:")
    stage_tracker = StageMetricsTracker()

    stage_logits = torch.tensor([
        [2.0, 0.5, 0.1],   # predicts MILD
        [0.2, 2.5, 0.3],   # predicts MODERATE
        [0.1, 0.2, 3.0],   # predicts SEVERE
        [3.0, 0.1, 0.1],   # predicts MILD (will be masked out — normal)
    ])
    stage_targets = torch.tensor([0, 1, 2, 0])
    pneumonia_mask = torch.tensor([True, True, True, False])

    stage_tracker.update(stage_logits, stage_targets, pneumonia_mask)
    stage_metrics = stage_tracker.compute()
    for k, v in stage_metrics.items():
        print(f"   {k:15s}: {v:.4f}")

    print("\n[OK] Metrics work!")