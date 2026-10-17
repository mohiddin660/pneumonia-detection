"""Loss functions for pneumonia detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss — focuses training on hard examples."""

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class MultiTaskLoss(nn.Module):
    """Combined loss: binary (Focal) + stage (CE, only on pneumonia)."""

    def __init__(self, class_weights=None, stage_weight: float = 0.3):
        super().__init__()
        self.binary_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        self.stage_loss = nn.CrossEntropyLoss()
        self.stage_weight = stage_weight

    def forward(self, binary_logits, stage_logits, binary_targets, stage_targets):
        l_binary = self.binary_loss(binary_logits, binary_targets)

        pneumonia_mask = binary_targets == 1
        if pneumonia_mask.any():
            l_stage = self.stage_loss(
                stage_logits[pneumonia_mask],
                stage_targets[pneumonia_mask],
            )
        else:
            l_stage = torch.tensor(0.0, device=binary_logits.device)

        total = l_binary + self.stage_weight * l_stage
        return total, l_binary.item(), l_stage.item()