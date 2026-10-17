"""
Pneumonia classifier using EfficientNet-B0 backbone with multi-task heads.

Two heads:
    1. Binary classification: NORMAL vs PNEUMONIA
    2. Stage classification: MILD / MODERATE / SEVERE
"""
import torch
import torch.nn as nn
import timm


class PneumoniaClassifier(nn.Module):
    """
    Multi-task classifier:
        - backbone: EfficientNet-B0 (pretrained on ImageNet)
        - binary_head: 2 outputs (NORMAL, PNEUMONIA)
        - stage_head:  3 outputs (MILD, MODERATE, SEVERE)
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 2,
        num_stages: int = 3,
        dropout: float = 0.3,
        multi_task: bool = True,
    ):
        super().__init__()
        self.multi_task = multi_task

        # Pretrained backbone, no classification head
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features

        self.dropout = nn.Dropout(dropout)

        # Binary head: NORMAL vs PNEUMONIA
        self.binary_head = nn.Linear(feat_dim, num_classes)

        # Stage head: MILD / MODERATE / SEVERE
        if multi_task:
            self.stage_head = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_stages),
            )

    def forward(self, x):
        """
        Returns:
            binary_logits: [batch, 2]
            stage_logits:  [batch, 3]  (only if multi_task=True)
        """
        feats = self.backbone(x)
        feats = self.dropout(feats)

        binary_logits = self.binary_head(feats)

        if self.multi_task:
            stage_logits = self.stage_head(feats)
            return binary_logits, stage_logits

        return binary_logits


# ---------------- QUICK TEST ----------------
if __name__ == "__main__":
    print("Building model...")
    model = PneumoniaClassifier(multi_task=True)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Dummy forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        binary_logits, stage_logits = model(x)

    print(f"\nInput shape:         {x.shape}")
    print(f"Binary logits shape: {binary_logits.shape}  (expect [2, 2])")
    print(f"Stage logits shape:  {stage_logits.shape}   (expect [2, 3])")
    print("\n[OK] Model works!")