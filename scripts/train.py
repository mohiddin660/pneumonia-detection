"""
Main training entry point.

Run:  python -m scripts.train
"""
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils.class_weight import compute_class_weight

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.data.dataset import RSNAPneumoniaDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.classifier import PneumoniaClassifier
from src.models.losses import MultiTaskLoss
from src.training.trainer import Trainer

log = get_logger("main", "logs/main.log")


def main():
    # ---------- 1. Config ----------
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])

    # ---------- 2. Device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 70)
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB")
    log.info("=" * 70)

    # ---------- 3. Transforms ----------
    size = cfg["data"]["img_size"]
    train_tf = get_train_transforms(size)
    eval_tf = get_eval_transforms(size)

    # ---------- 4. Datasets ----------
    log.info("Loading datasets...")
    train_ds = RSNAPneumoniaDataset("data/splits/train.csv", transform=train_tf)
    val_ds = RSNAPneumoniaDataset("data/splits/val.csv", transform=eval_tf)

    max_samples = cfg["data"].get("max_samples", None)
    if max_samples and max_samples < len(train_ds):
        rng = np.random.default_rng(cfg["project"]["seed"])
        idx = rng.choice(len(train_ds), size=max_samples, replace=False)
        train_ds = Subset(train_ds, idx)
        log.info(f"Subsampled train to {max_samples} samples")

    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ---------- 5. DataLoaders ----------
    bs = cfg["training"]["batch_size"]
    nw = cfg["data"]["num_workers"]

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0),
    )

    # ---------- 6. Class weights (SOFTENED to prevent overfitting) ----------
    log.info("Computing class weights...")
    if isinstance(train_ds, Subset):
        base_df = train_ds.dataset.df
        indices = train_ds.indices
        labels = base_df.iloc[indices]["Target"].values
    else:
        labels = train_ds.df["Target"].values
    labels = np.array(labels)

    class_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=labels
    )
    # ---- SOFTEN the weights using square root ----
    # Before: NORMAL=0.645, PNEUMONIA=2.219 (too aggressive → overfitting)
    # After:  NORMAL=0.803, PNEUMONIA=1.490 (softer → better generalization)
    class_weights = np.sqrt(class_weights)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float, device=device)
    log.info(f"Class weights (softened): NORMAL={class_weights[0]:.3f}, PNEUMONIA={class_weights[1]:.3f}")

    # ---------- 7. Model ----------
    log.info("Building model...")
    model = PneumoniaClassifier(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        num_stages=3,
        dropout=cfg["model"]["dropout"],
        multi_task=cfg["model"]["multi_task"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")

    # ---------- 8. Loss, optimizer, scheduler ----------
    stage_weight = cfg["training"].get("stage_weight", 0.3)
    criterion = MultiTaskLoss(
        class_weights=class_weights_t,
        stage_weight=stage_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=1e-6
    )

    # ---------- 9. TensorBoard ----------
    writer = SummaryWriter(cfg["paths"]["log_dir"])

    # ---------- 10. Trainer ----------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        cfg=cfg,
        writer=writer,
        ckpt_dir=cfg["paths"]["checkpoint_dir"],
    )

    # ---------- 11. Train! ----------
    log.info("Starting training...")
    best_auc = trainer.fit()

    log.info("")
    log.info("=" * 70)
    log.info(f"[DONE] Best val AUC: {best_auc:.4f}")
    log.info(f"Model saved at: {cfg['paths']['checkpoint_dir']}best.pt")
    log.info("=" * 70)

    writer.close()


if __name__ == "__main__":
    main()