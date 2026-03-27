"""
train.py — Training harness for piano AMT.

Implements:
  OnsetsFramesLoss: 4-head loss (onset BCE + frame BCE + offset BCE + velocity MSE).
  Trainer:          Full training loop with Adam, ReduceLROnPlateau, checkpoint saving.
  DummyModel:       Zero-output model for testing harness before adding the real model.
  main():           CLI entry-point with argparse.

Papers:
  Hawthorne 2018a §3.2: Adam lr=6e-4, gradient clip max_norm=3.0, pos_weight.
  Hawthorne 2018a §3.2: weighted BCE for onset/frame/offset; masked MSE for velocity.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.constants import N_KEYS, N_MELS, MAX_SEGMENT_FRAMES
from src.dataloader import get_dataloader


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class OnsetsFramesLoss(nn.Module):
    """
    4-head loss for the Onsets and Frames piano transcription model.

    Heads:
      onset   — BCEWithLogitsLoss with pos_weight (imbalanced labels).
      frame   — BCEWithLogitsLoss with pos_weight.
      offset  — BCEWithLogitsLoss with pos_weight.
      velocity— MSE, computed ONLY at positions where onset > 0.5 (masked).

    The velocity head uses masked MSE so gradient is only computed where a
    note actually begins, following Hawthorne 2018a §3.2.

    Args:
        pos_weight: Positive class weight for BCE losses (default 5.0).
                    Compensates for class imbalance (most frames have no notes).

    Paper: Hawthorne 2018a §3.2 — weighted BCE loss, velocity masked MSE.
    """

    def __init__(self, pos_weight: float = 5.0) -> None:
        super().__init__()
        pw = torch.tensor(pos_weight)
        self.bce_onset  = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.bce_frame  = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.bce_offset = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.mse        = nn.MSELoss(reduction="none")

    def forward(
        self,
        pred:   Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined 4-head loss.

        Args:
            pred:   Dict with keys "onset", "frame", "offset", "velocity".
                    Each value is a logit tensor of shape (B, T, 88).
            target: Dict with same keys. Values are ground-truth labels in [0,1].

        Returns:
            Dict with keys:
              "total"    — sum of all head losses (scalar Tensor, differentiable).
              "onset"    — onset BCE loss value (float, for logging).
              "frame"    — frame BCE loss value (float, for logging).
              "offset"   — offset BCE loss value (float, for logging).
              "velocity" — masked velocity MSE value (float, for logging).

        Shape:
            pred/target tensors: (B, T, N_KEYS) = (B, T, 88)
        """
        loss_onset  = self.bce_onset( pred["onset"],   target["onset"])
        loss_frame  = self.bce_frame( pred["frame"],   target["frame"])
        loss_offset = self.bce_offset(pred["offset"],  target["offset"])

        # Velocity: masked MSE — only where onset > 0.5
        mask = (target["onset"] > 0.5).float()   # (B, T, 88)
        n_active = mask.sum().clamp(min=1.0)      # avoid division by zero

        vel_pred   = torch.sigmoid(pred["velocity"]) if pred["velocity"].requires_grad \
                     else pred["velocity"]
        vel_mse    = self.mse(vel_pred, target["velocity"])  # (B, T, 88)
        loss_vel   = (vel_mse * mask).sum() / n_active

        total = loss_onset + loss_frame + loss_offset + loss_vel

        return {
            "total":    total,
            "onset":    loss_onset.item(),
            "frame":    loss_frame.item(),
            "offset":   loss_offset.item(),
            "velocity": loss_vel.item(),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Full training loop: forward → loss → backward → step → log → checkpoint.

    Args:
        model:           PyTorch model with forward(mel) → Dict[str, Tensor].
        train_loader:    DataLoader for training split.
        val_loader:      DataLoader for validation split.
        device:          torch.device for computation.
        lr:              Adam learning rate (default 6e-4 — Hawthorne 2018a §3.2).
        pos_weight:      BCE positive class weight (default 5.0).
        max_grad_norm:   Gradient clipping norm (default 3.0 — Hawthorne 2018a §3.2).
        checkpoint_dir:  Directory to save checkpoints.
        log_every:       Log training loss every N global steps.

    Papers:
        Hawthorne 2018a §3.2: Adam lr=6e-4, gradient clip max_norm=3.0.
    """

    def __init__(
        self,
        model:           nn.Module,
        train_loader:    DataLoader,
        val_loader:      DataLoader,
        device:          torch.device,
        lr:              float = 6e-4,
        pos_weight:      float = 5.0,
        max_grad_norm:   float = 3.0,
        checkpoint_dir:  str = "checkpoints",
        log_every:       int = 50,
    ) -> None:
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.max_grad_norm  = max_grad_norm
        self.log_every      = log_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = OnsetsFramesLoss(pos_weight=pos_weight)
        self.optimizer = Adam(model.parameters(), lr=lr)

        # ReduceLROnPlateau: halve LR if val loss doesn't improve for 3 epochs
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        self.global_step = 0
        self.best_val_loss: float = float("inf")

    # -----------------------------------------------------------------------

    def _move_batch(self, batch: Dict[str, object]) -> Dict[str, object]:
        """Move all Tensor values in batch to self.device."""
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device, non_blocking=True)
            else:
                moved[k] = v
        return moved

    # -----------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        """
        Run one training epoch over train_loader.

        Logs per-head losses every self.log_every steps.

        Args:
            epoch: Current epoch number (1-indexed).

        Returns:
            Mean total loss over the epoch.
        """
        self.model.train()
        total_loss  = 0.0
        n_batches   = 0
        t0          = time.time()

        for batch in self.train_loader:
            batch = self._move_batch(batch)

            # Forward
            pred = self.model(batch["mel"])

            # Loss
            losses = self.criterion(pred, batch)

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()

            # Gradient clip — Hawthorne 2018a §3.2
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )

            self.optimizer.step()
            self.global_step += 1

            total_loss += losses["total"].item()
            n_batches  += 1

            if self.global_step % self.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  [epoch {epoch:3d}  step {self.global_step:6d}]  "
                    f"total={losses['total'].item():.4f}  "
                    f"onset={losses['onset']:.4f}  "
                    f"frame={losses['frame']:.4f}  "
                    f"offset={losses['offset']:.4f}  "
                    f"vel={losses['velocity']:.4f}  "
                    f"({elapsed:.1f}s)"
                )
                t0 = time.time()

        return total_loss / max(n_batches, 1)

    # -----------------------------------------------------------------------

    def validate(self, epoch: int) -> float:
        """
        Run validation over val_loader.

        Calls scheduler.step(mean_val_loss) after evaluation.

        Args:
            epoch: Current epoch number.

        Returns:
            Mean total validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch  = self._move_batch(batch)
                pred   = self.model(batch["mel"])
                losses = self.criterion(pred, batch)
                total_loss += losses["total"].item()
                n_batches  += 1

        mean_loss = total_loss / max(n_batches, 1)
        self.scheduler.step(mean_loss)
        return mean_loss

    # -----------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Save model and optimiser state to a checkpoint file.

        Filename format: epoch_NNN_valloss_X.XXXX.pt

        Args:
            epoch:    Current epoch.
            val_loss: Validation loss (used in filename for easy sorting).
        """
        fname = self.checkpoint_dir / f"epoch_{epoch:03d}_valloss_{val_loss:.4f}.pt"
        torch.save(
            {
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch":           epoch,
                "val_loss":        val_loss,
                "global_step":     self.global_step,
            },
            fname,
        )
        print(f"  Saved checkpoint: {fname}")

    # -----------------------------------------------------------------------

    def fit(self, epochs: int = 30) -> None:
        """
        Full training loop: train_epoch → validate → checkpoint if best val.

        Args:
            epochs: Total number of training epochs (default 30).
        """
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")

            train_loss = self.train_epoch(epoch)
            val_loss   = self.validate(epoch)

            print(
                f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                print(f"  *** New best val loss: {val_loss:.4f} ***")


# ---------------------------------------------------------------------------
# Dummy model (for testing harness before real model is integrated)
# ---------------------------------------------------------------------------

class _DummyModel(nn.Module):
    """
    Zero-output placeholder model for end-to-end harness testing.

    Returns a dict of zeros with the correct output shapes so the entire
    training loop (loss, backward, optimizer) can be validated before
    the real OnsetsAndFrames model is added.

    Input:  mel Tensor (B, 229, T)
    Output: Dict with "onset", "frame", "offset", "velocity" — each (B, T, 88).
    """

    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, T = mel.shape
        zeros   = torch.zeros(B, T, N_KEYS, device=mel.device)
        return {
            "onset":    zeros,
            "frame":    zeros.clone(),
            "offset":   zeros.clone(),
            "velocity": zeros.clone(),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Command-line entry point for training.

    Usage:
        python train.py --maestro_root /path/to/maestro-v3.0.0 \\
                        --cache_dir /path/to/cache \\
                        --batch_size 4 --epochs 30

    Uses DummyModel for end-to-end pipeline validation.  Replace with the
    real OnsetsAndFrames model once the model module is implemented.
    """
    parser = argparse.ArgumentParser(
        description="Piano AMT training harness (Onsets and Frames)"
    )
    parser.add_argument(
        "--maestro_root", required=True, type=str,
        help="Root directory of MAESTRO v3 dataset (contains *.csv)"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Directory for NPZ cache (defaults to maestro_root/cache)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size (default 4; jongwook ref uses 8)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs (default 30)"
    )
    parser.add_argument(
        "--lr", type=float, default=6e-4,
        help="Adam learning rate (default 6e-4 — Hawthorne 2018a §3.2)"
    )
    parser.add_argument(
        "--max_files", type=int, default=None,
        help="Limit dataset to N files per split (for debugging)"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="DataLoader worker processes (default 2)"
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # DataLoaders
    print("\nBuilding DataLoaders...")
    train_loader = get_dataloader(
        maestro_root=args.maestro_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        max_files=args.max_files,
        use_augmentation=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = get_dataloader(
        maestro_root=args.maestro_root,
        split="validation",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        max_files=args.max_files,
        use_augmentation=False,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")

    # Model — replace _DummyModel with real OnsetsAndFrames once available
    model = _DummyModel()
    print(f"\nModel: {model.__class__.__name__}")

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        pos_weight=5.0,
        max_grad_norm=3.0,
        checkpoint_dir=args.checkpoint_dir,
        log_every=50,
    )

    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    main()
