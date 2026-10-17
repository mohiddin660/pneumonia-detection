"""
Training loop for pneumonia detection.

Features:
    - Train + validation per epoch
    - Mixed precision (faster on GPU)
    - Gradient clipping (prevents explosion)
    - Best-model checkpointing
    - Early stopping
    - TensorBoard logging
"""
import time
from pathlib import Path

import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.utils.logger import get_logger
from src.training.metrics import BinaryMetricsTracker, StageMetricsTracker

log = get_logger("trainer", "logs/trainer.log")


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        cfg,
        writer=None,
        ckpt_dir="models/",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg
        self.writer = writer

        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision (only on CUDA)
        use_amp = (
            cfg["training"]["mixed_precision"] and device.type == "cuda"
        )
        self.scaler = GradScaler("cuda", enabled=use_amp)

        self.best_auc = 0.0
        self.patience_counter = 0
        self.patience = cfg["training"]["early_stopping_patience"]

    # ---------------- TRAIN ----------------
    def _train_epoch(self, epoch: int):
        self.model.train()

        binary_tracker = BinaryMetricsTracker()
        stage_tracker = StageMetricsTracker()

        running_loss = 0.0
        n_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch:02d}")
        for imgs, binary_labels, stage_labels in pbar:
            imgs = imgs.to(self.device, non_blocking=True)
            binary_labels = binary_labels.to(self.device, non_blocking=True)
            stage_labels = stage_labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.scaler.is_enabled()):
                binary_logits, stage_logits = self.model(imgs)
                loss, l_bin, l_stage = self.criterion(
                    binary_logits, stage_logits, binary_labels, stage_labels
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

            binary_tracker.update(binary_logits, binary_labels)
            pneumonia_mask = binary_labels == 1
            stage_tracker.update(stage_logits, stage_labels, pneumonia_mask)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(n_samples, 1)
        metrics = {**binary_tracker.compute(), **stage_tracker.compute()}
        return avg_loss, metrics

    # ---------------- VALIDATE ----------------
    @torch.no_grad()
    def _val_epoch(self, epoch: int):
        self.model.eval()

        binary_tracker = BinaryMetricsTracker()
        stage_tracker = StageMetricsTracker()

        running_loss = 0.0
        n_samples = 0

        pbar = tqdm(self.val_loader, desc=f"Val   E{epoch:02d}")
        for imgs, binary_labels, stage_labels in pbar:
            imgs = imgs.to(self.device, non_blocking=True)
            binary_labels = binary_labels.to(self.device, non_blocking=True)
            stage_labels = stage_labels.to(self.device, non_blocking=True)

            binary_logits, stage_logits = self.model(imgs)
            loss, _, _ = self.criterion(
                binary_logits, stage_logits, binary_labels, stage_labels
            )

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

            binary_tracker.update(binary_logits, binary_labels)
            pneumonia_mask = binary_labels == 1
            stage_tracker.update(stage_logits, stage_labels, pneumonia_mask)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(n_samples, 1)
        metrics = {**binary_tracker.compute(), **stage_tracker.compute()}
        return avg_loss, metrics

    # ---------------- FIT ----------------
    def fit(self):
        epochs = self.cfg["training"]["epochs"]
        log.info("=" * 70)
        log.info(f"Starting training for {epochs} epochs")
        log.info(f"Device: {self.device} | Patience: {self.patience}")
        log.info("=" * 70)

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_m = self._train_epoch(epoch)
            val_loss, val_m = self._val_epoch(epoch)

            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    pass

            elapsed = time.time() - t0

            log.info(
                f"E{epoch:02d} | "
                f"tr_loss={train_loss:.4f} tr_acc={train_m['accuracy']:.4f} tr_f1={train_m['f1']:.4f} | "
                f"vl_loss={val_loss:.4f} vl_acc={val_m['accuracy']:.4f} vl_f1={val_m['f1']:.4f} "
                f"vl_auc={val_m['auc']:.4f} | "
                f"stage_acc={val_m['stage_accuracy']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # TensorBoard
            if self.writer is not None:
                for k, v in train_m.items():
                    self.writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in val_m.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)
                self.writer.add_scalar("train/loss", train_loss, epoch)
                self.writer.add_scalar("val/loss", val_loss, epoch)

            # Checkpoint best
            current_auc = val_m["auc"]
            if current_auc > self.best_auc:
                self.best_auc = current_auc
                self.patience_counter = 0
                ckpt_path = self.ckpt_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "best_auc": self.best_auc,
                        "config": self.cfg,
                    },
                    ckpt_path,
                )
                log.info(f"   [BEST] New best AUC = {self.best_auc:.4f} -> saved {ckpt_path}")
            else:
                self.patience_counter += 1
                log.info(
                    f"   [no improvement] patience {self.patience_counter}/{self.patience}"
                )
                if self.patience_counter >= self.patience:
                    log.info(f"[EARLY STOP] Stopping at epoch {epoch}")
                    break

        log.info("=" * 70)
        log.info(f"[DONE] Best val AUC: {self.best_auc:.4f}")
        log.info("=" * 70)
        return self.best_auc


# ---------------- QUICK TEST ----------------
if __name__ == "__main__":
    print("Trainer module loaded. Run training with: python -m scripts.train")