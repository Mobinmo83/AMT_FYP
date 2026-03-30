"""
models/onsets_frames/train.py — Training harness for OnsetsAndFrames.

Design:
  - Every run gets its own directory under runs/<run_name>/
    containing: checkpoints/, plots/, metrics.json, config.json
  - Loss curves (train + val per epoch) are saved as PNG after every epoch
    so training progress is always visible for the report.
  - All epoch losses are stored in metrics.json and updated after every epoch
    so the file is always up-to-date even if Colab disconnects.
  - The model uses post-sigmoid outputs, so BCELoss is used (not BCEWithLogits).
  - Per-parameter gradient clipping matches jongwook's implementation.
  - ReduceLROnPlateau scheduler halves LR after 3 epochs without improvement.
  - EVERY epoch is checkpointed (latest.pt + best.pt) for Colab crash safety.
  - Full resume support: model, optimizer, scheduler, epoch, best_val_loss,
    global_step, and metrics history are all restored.

Usage (CLI):
    python -m models.onsets_frames.train \\
        --run_name  of_baseline_full \\
        --maestro_root /content/drive/MyDrive/piano_amt/maestro-v3.0.0 \\
        --cache_dir    /content/drive/MyDrive/piano_amt/cache \\
        --runs_dir     /content/drive/MyDrive/piano_amt/runs \\
        --batch_size   8 \\
        --epochs       50

Resume after disconnect:
    python -m models.onsets_frames.train \\
        --run_name  of_baseline_full \\
        --maestro_root /content/drive/MyDrive/piano_amt/maestro-v3.0.0 \\
        --cache_dir    /content/drive/MyDrive/piano_amt/cache \\
        --runs_dir     /content/drive/MyDrive/piano_amt/runs \\
        --resume       /content/drive/MyDrive/piano_amt/runs/of_baseline_full/checkpoints/latest.pt \\
        --epochs       50

Papers:
  Hawthorne 2018a §3.2: Adam lr=6e-4, grad clip norm 3.0, masked MSE velocity.
  jongwook/onsets-and-frames: per-param grad clip, batch_size=8.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path bootstrap — allow running as script from any working directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent   # piano_amt/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.onsets_frames.model import OnsetsAndFrames
from src.constants import N_KEYS, N_MELS
from src.dataloader import get_dataloader


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class OnsetsFramesLoss(nn.Module):
    """
    4-head BCE + masked MSE loss for post-sigmoid model outputs.

    Because OnsetsAndFrames applies Sigmoid inside forward(), outputs are
    probabilities in [0,1].  We therefore use BCELoss (not BCEWithLogitsLoss).

    Heads:
      onset   — BCELoss with pos_weight  (class imbalance: few notes active)
      frame   — BCELoss with pos_weight
      offset  — BCELoss with pos_weight
      velocity— MSE computed ONLY at frames where onset target > 0.5

    Args:
        pos_weight: Scalar weight on positive class for BCE heads.
                    Default 5.0 — Hawthorne 2018a §3.2.

    Paper: Hawthorne 2018a §3.2 — weighted BCE, velocity masked MSE.
    """

    def __init__(self, pos_weight: float = 5.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.mse = nn.MSELoss(reduction="none")

    def _bce(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Weighted BCE: pos_weight on positive class."""
        # Manual weighted BCE so we can apply a scalar pos_weight
        # without registering a buffer every forward call.
        loss = -(
            self.pos_weight * target * torch.log(pred.clamp(min=1e-7))
            + (1.0 - target) * torch.log((1.0 - pred).clamp(min=1e-7))
        )
        return loss.mean()

    def forward(
        self,
        pred:   Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred:   Dict with "onset","frame","offset","velocity" — each (B,T,88)
                    Values are post-sigmoid probabilities [0,1].
            target: Dict with same keys — ground truth labels in [0,1].

        Returns:
            Dict with:
              "total"    — differentiable scalar sum of all head losses
              "onset"    — float (for logging)
              "frame"    — float
              "offset"   — float
              "velocity" — float
        """
        loss_onset  = self._bce(pred["onset"],  target["onset"])
        loss_frame  = self._bce(pred["frame"],  target["frame"])
        loss_offset = self._bce(pred["offset"], target["offset"])

        # Velocity: masked MSE at onset positions only
        mask     = (target["onset"] > 0.5).float()
        n_active = mask.sum().clamp(min=1.0)
        vel_mse  = self.mse(pred["velocity"], target["velocity"])  # (B,T,88)
        loss_vel = (vel_mse * mask).sum() / n_active

        total = loss_onset + loss_frame + loss_offset + loss_vel

        return {
            "total":    total,
            "onset":    loss_onset.item(),
            "frame":    loss_frame.item(),
            "offset":   loss_offset.item(),
            "velocity": loss_vel.item(),
        }


# ---------------------------------------------------------------------------
# Run-directory manager
# ---------------------------------------------------------------------------

class RunDirectory:
    """
    Manages the directory structure for one training run.

    Layout:
        <runs_dir>/<run_name>/
            checkpoints/      ← .pt files (latest.pt + best.pt + per-epoch)
            plots/            ← PNG loss curves saved every epoch
            metrics.json      ← updated after every epoch (survives resume)
            config.json       ← saved once at run start
    """

    def __init__(self, runs_dir: str | Path, run_name: str) -> None:
        self.root = Path(runs_dir) / run_name
        self.checkpoints = self.root / "checkpoints"
        self.plots       = self.root / "plots"

        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoints.mkdir(exist_ok=True)
        self.plots.mkdir(exist_ok=True)

        self.metrics_path = self.root / "metrics.json"
        self.config_path  = self.root / "config.json"

        # Load existing history if resuming, otherwise start fresh
        if self.metrics_path.exists():
            with open(self.metrics_path, "r") as f:
                self._history: Dict[str, List] = json.load(f)
            print(f"  Loaded existing metrics ({len(self._history['epoch'])} epochs)")
        else:
            self._history: Dict[str, List] = {
                "epoch":      [],
                "train_loss": [],
                "val_loss":   [],
                "train_onset":  [], "train_frame":  [], "train_offset":  [], "train_vel":  [],
                "val_onset":    [], "val_frame":    [], "val_offset":    [], "val_vel":    [],
                "lr":         [],
            }

    def save_config(self, cfg: dict) -> None:
        with open(self.config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  Config saved → {self.config_path}")

    def log_epoch(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses:   Dict[str, float],
        lr: float,
    ) -> None:
        """Append one epoch to history and flush to metrics.json."""
        h = self._history
        h["epoch"].append(epoch)
        h["train_loss"].append(train_losses["total"])
        h["val_loss"].append(val_losses["total"])
        h["train_onset"].append(train_losses["onset"])
        h["train_frame"].append(train_losses["frame"])
        h["train_offset"].append(train_losses["offset"])
        h["train_vel"].append(train_losses["velocity"])
        h["val_onset"].append(val_losses["onset"])
        h["val_frame"].append(val_losses["frame"])
        h["val_offset"].append(val_losses["offset"])
        h["val_vel"].append(val_losses["velocity"])
        h["lr"].append(lr)
        with open(self.metrics_path, "w") as f:
            json.dump(h, f, indent=2)

    def save_loss_curves(self, epoch: int) -> None:
        """Save total + per-head loss curves as PNG. Called after every epoch."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        h  = self._history
        ep = h["epoch"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Training curves — epoch {epoch}", fontsize=13)

        # Total loss
        axes[0, 0].plot(ep, h["train_loss"], label="train")
        axes[0, 0].plot(ep, h["val_loss"],   label="val")
        axes[0, 0].set_title("Total loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Per-head
        for ax, key, title in [
            (axes[0, 1], "onset",  "Onset BCE"),
            (axes[0, 2], "frame",  "Frame BCE"),
            (axes[1, 0], "offset", "Offset BCE"),
            (axes[1, 1], "vel",    "Velocity MSE"),
        ]:
            ax.plot(ep, h[f"train_{key}"], label="train")
            ax.plot(ep, h[f"val_{key}"],   label="val")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Learning rate
        axes[1, 2].plot(ep, h["lr"], color="green")
        axes[1, 2].set_title("Learning rate")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = self.plots / f"loss_curves_epoch{epoch:03d}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Loss curve saved → {out_path}")

    def checkpoint_path(self, epoch: int, val_loss: float) -> Path:
        return self.checkpoints / f"epoch_{epoch:03d}_valloss_{val_loss:.4f}.pt"

    def best_checkpoint_path(self) -> Path:
        return self.checkpoints / "best.pt"

    def latest_checkpoint_path(self) -> Path:
        return self.checkpoints / "latest.pt"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Full training loop with per-epoch logging, loss-curve saving,
    and Drive-safe checkpointing.

    GPU performance features:
      - cuDNN benchmark mode (auto-tunes conv algorithms for fixed input sizes)
      - Automatic Mixed Precision (AMP) with GradScaler for ~1.5–2× speedup
      - torch.compile() on PyTorch 2.0+ for kernel fusion
      - Non-blocking CPU→GPU transfers with pin_memory
      - set_to_none=True for zero_grad (avoids memset)
      - tf32 on Ampere+ GPUs (A100/H100) for ~3× faster matmul

    Checkpointing strategy (Colab crash-safe):
      - latest.pt  — saved EVERY epoch (always resumable)
      - best.pt    — saved when val_loss improves (best model for eval)
      - epoch_NNN_valloss_X.XXXX.pt — kept for each best (audit trail)

    Args:
        model:        OnsetsAndFrames (or any compatible nn.Module).
        train_loader: DataLoader for training split.
        val_loader:   DataLoader for validation split.
        device:       torch.device.
        run_dir:      RunDirectory instance managing output paths.
        lr:           Adam learning rate (default 6e-4 — Hawthorne 2018a §3.2).
        pos_weight:   BCE positive class weight (default 5.0).
        max_grad_norm: Per-parameter gradient clip norm (default 3.0).
        log_every:    Print per-step log every N global steps.
        use_amp:      Enable automatic mixed precision (default True for CUDA).
        use_compile:  Enable torch.compile (default True for PyTorch 2.0+).
    """

    def __init__(
        self,
        model:          nn.Module,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        device:         torch.device,
        run_dir:        RunDirectory,
        lr:             float = 6e-4,
        pos_weight:     float = 5.0,
        max_grad_norm:  float = 3.0,
        log_every:      int   = 50,
        use_amp:        bool  = True,
        use_compile:    bool  = True,
    ) -> None:
        self.device         = device
        self.run_dir        = run_dir
        self.max_grad_norm  = max_grad_norm
        self.log_every      = log_every

        # ------------------------------------------------------------------
        # GPU performance setup
        # ------------------------------------------------------------------
        if device.type == "cuda":
            # cuDNN benchmark: auto-selects fastest conv algorithm for fixed
            # input sizes (our mel is always B×229×640 during training).
            # First batch is ~10% slower (benchmarking), all subsequent are faster.
            torch.backends.cudnn.benchmark = True
            print(f"  cuDNN benchmark: enabled")

            # TF32 on Ampere+ GPUs (A100, H100): 3× faster matmul/conv
            # with negligible precision loss. No effect on T4 (Turing arch).
            if torch.cuda.get_device_capability(0)[0] >= 8:  # Ampere = 8.x
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print(f"  TF32: enabled (Ampere+ GPU detected)")
            else:
                print(f"  TF32: not available (GPU compute capability "
                      f"{torch.cuda.get_device_capability(0)})")

        # Move model to device
        self.model = model.to(device)

        # ------------------------------------------------------------------
        # AMP (Automatic Mixed Precision)
        # Runs forward pass in float16 (2× less VRAM, faster compute),
        # keeps master weights in float32 for stability.
        # GradScaler prevents float16 underflow during backward.
        # Safe for all our ops: Conv2d, LSTM, Linear, Sigmoid, BCE, MSE.
        # BatchNorm auto-stays fp32 under autocast.
        # ------------------------------------------------------------------
        self.use_amp = use_amp and (device.type == "cuda")
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
            print(f"  AMP: enabled (float16 forward, float32 weights)")
        else:
            self.scaler = None
            print(f"  AMP: disabled")

        # ------------------------------------------------------------------
        # torch.compile (PyTorch 2.0+)
        # Fuses ops, eliminates Python overhead, reduces kernel launches.
        # ~10-30% faster after warmup. Falls back gracefully if unsupported.
        # ------------------------------------------------------------------
        self.used_compile = False
        if use_compile and device.type == "cuda":
            torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
            if torch_version >= (2, 0):
                try:
                    self.model = torch.compile(self.model)
                    self.used_compile = True
                    print(f"  torch.compile: enabled (kernel fusion)")
                except Exception as e:
                    print(f"  torch.compile: failed ({e}), continuing without")
            else:
                print(f"  torch.compile: requires PyTorch 2.0+ "
                      f"(have {torch.__version__})")

        # ------------------------------------------------------------------
        # Optimizer, scheduler, loss
        # ------------------------------------------------------------------
        self.criterion  = OnsetsFramesLoss(pos_weight=pos_weight)
        self.optimizer  = Adam(model.parameters(), lr=lr)
        self.scheduler  = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.global_step    = 0
        self.best_val_loss: float = float("inf")

    # -----------------------------------------------------------------------

    def _move(self, batch: dict) -> dict:
        """Transfer batch tensors to GPU with non-blocking async copy.
        Works with pin_memory=True in DataLoader for overlap."""
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    # -----------------------------------------------------------------------

    def _per_param_clip(self) -> None:
        """
        Per-parameter gradient clipping — jongwook improvement.
        Each parameter's gradient norm is clipped independently to max_grad_norm.
        This is stricter than global norm clipping and stabilises the large model.

        When using AMP, gradients have already been unscaled by scaler.unscale_()
        before this is called, so clipping operates on the true gradient magnitudes.
        """
        for p in self.model.parameters():
            if p.grad is not None:
                nn.utils.clip_grad_norm_([p], max_norm=self.max_grad_norm)

    # -----------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        One full pass over train_loader with AMP support.

        Returns:
            Dict with keys "total","onset","frame","offset","velocity"
            — mean values over the epoch.
        """
        self.model.train()
        totals: Dict[str, float] = {
            "total": 0., "onset": 0., "frame": 0., "offset": 0., "velocity": 0.
        }
        n_batches = 0
        t0 = time.time()

        for batch in self.train_loader:
            batch = self._move(batch)

            # --- Forward pass (possibly in float16 under AMP) ---
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred   = self.model(batch["mel"])
                losses = self.criterion(pred, batch)

            # --- Backward pass ---
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                # Scale loss to prevent float16 gradient underflow
                self.scaler.scale(losses["total"]).backward()
                # Unscale gradients BEFORE clipping so clip thresholds are correct
                self.scaler.unscale_(self.optimizer)
                self._per_param_clip()
                # Step optimizer (skips step if gradients contained inf/nan)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["total"].backward()
                self._per_param_clip()
                self.optimizer.step()

            self.global_step += 1
            n_batches += 1
            for k in totals:
                totals[k] += losses[k].item() if hasattr(losses[k], "item") else losses[k]

            if self.global_step % self.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  [ep {epoch:03d}  step {self.global_step:6d}]"
                    f"  total={losses['total'].item():.4f}"
                    f"  onset={losses['onset']:.4f}"
                    f"  frame={losses['frame']:.4f}"
                    f"  offset={losses['offset']:.4f}"
                    f"  vel={losses['velocity']:.4f}"
                    f"  ({elapsed:.1f}s)"
                )
                t0 = time.time()

        n = max(n_batches, 1)
        return {k: v / n for k, v in totals.items()}

    # -----------------------------------------------------------------------

    def validate(self, epoch: int) -> Dict[str, float]:
        """One full pass over val_loader with AMP. Calls scheduler.step()."""
        self.model.eval()
        totals: Dict[str, float] = {
            "total": 0., "onset": 0., "frame": 0., "offset": 0., "velocity": 0.
        }
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move(batch)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    pred   = self.model(batch["mel"])
                    losses = self.criterion(pred, batch)
                n_batches += 1
                for k in totals:
                    totals[k] += losses[k].item() if hasattr(losses[k], "item") else losses[k]

        n = max(n_batches, 1)
        means = {k: v / n for k, v in totals.items()}
        self.scheduler.step(means["total"])
        return means

    # -----------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        """
        Save checkpoint to Drive.

        Always saves latest.pt (for resume after disconnect).
        If is_best, also saves best.pt and a named epoch checkpoint.

        Checkpoint contents:
          model_state     — nn.Module state dict (weights, always fp32)
          optimizer_state — Adam state (momentum buffers, step counts)
          scheduler_state — ReduceLROnPlateau state (num_bad_epochs, etc.)
          scaler_state    — AMP GradScaler state (scale factor, growth tracker)
          epoch           — int, last completed epoch
          val_loss        — float, validation loss at this epoch
          global_step     — int, total training steps completed
          best_val_loss   — float, best val loss seen so far
          use_amp         — bool, whether AMP was used (for resume compatibility)
        """
        # If model was torch.compile'd, get the underlying module's state_dict
        model_to_save = self.model
        if hasattr(self.model, '_orig_mod'):
            model_to_save = self.model._orig_mod

        ckpt = {
            "model_state":     model_to_save.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "epoch":           epoch,
            "val_loss":        val_loss,
            "global_step":     self.global_step,
            "best_val_loss":   self.best_val_loss,
            "use_amp":         self.use_amp,
        }

        # Save AMP scaler state for proper resume
        if self.scaler is not None:
            ckpt["scaler_state"] = self.scaler.state_dict()

        # Always save latest.pt — the crash-safety net
        latest_path = self.run_dir.latest_checkpoint_path()
        torch.save(ckpt, latest_path)
        print(f"  latest.pt saved → {latest_path}")

        if is_best:
            # Save named checkpoint for audit trail
            named_path = self.run_dir.checkpoint_path(epoch, val_loss)
            torch.save(ckpt, named_path)

            # Overwrite best.pt
            best_path = self.run_dir.best_checkpoint_path()
            torch.save(ckpt, best_path)
            print(f"  best.pt updated → {best_path}")
            print(f"  Named checkpoint → {named_path}")

    # -----------------------------------------------------------------------

    def fit(self, epochs: int = 30, start_epoch: int = 1) -> None:
        """
        Full training loop: train → validate → log → checkpoint → plot.

        Args:
            epochs:      Total epochs to train (absolute, not additional).
            start_epoch: First epoch number (>1 when resuming).
        """
        print(f"\n{'='*60}")
        print(f"Starting training: epochs {start_epoch}→{epochs}")
        print(f"Run directory: {self.run_dir.root}")
        print(f"Best val loss so far: {self.best_val_loss:.4f}")
        print(f"Global step: {self.global_step}")
        print(f"{'='*60}\n")

        for epoch in range(start_epoch, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} ---")
            t_epoch = time.time()

            train_losses = self.train_epoch(epoch)
            val_losses   = self.validate(epoch)
            current_lr   = self.optimizer.param_groups[0]["lr"]

            # Log to metrics.json (flushed to Drive immediately)
            self.run_dir.log_epoch(epoch, train_losses, val_losses, current_lr)

            elapsed = time.time() - t_epoch
            print(
                f"  train_loss={train_losses['total']:.4f}  "
                f"val_loss={val_losses['total']:.4f}  "
                f"lr={current_lr:.2e}  "
                f"({elapsed:.0f}s)"
            )

            # Save loss curves PNG every epoch
            self.run_dir.save_loss_curves(epoch)

            # Determine if this is a new best
            is_best = val_losses["total"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses["total"]
                print(f"  *** New best val loss: {val_losses['total']:.4f} ***")

            # Save checkpoint EVERY epoch (latest.pt always, best.pt if improved)
            self.save_checkpoint(epoch, val_losses["total"], is_best=is_best)

            # Estimate time remaining
            epochs_left = epochs - epoch
            if epochs_left > 0:
                eta_min = (elapsed * epochs_left) / 60
                print(f"  ETA: ~{eta_min:.0f} min ({epochs_left} epochs left)")

        print(f"\n{'='*60}")
        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")
        print(f"All outputs in: {self.run_dir.root}")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train OnsetsAndFrames on MAESTRO."
    )
    parser.add_argument("--run_name",     required=True,  help="e.g. of_baseline_full")
    parser.add_argument("--maestro_root", required=True)
    parser.add_argument("--cache_dir",    default=None,
                        help="NPZ cache dir (default: maestro_root/cache)")
    parser.add_argument("--runs_dir",     default=None,
                        help="Parent dir for all runs (default: Drive path)")
    parser.add_argument("--max_files",    type=int, default=None,
                        help="Limit files per split (None = all)")
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--epochs",       type=int, default=30)
    parser.add_argument("--lr",           type=float, default=6e-4)
    parser.add_argument("--pos_weight",   type=float, default=5.0)
    parser.add_argument("--max_grad_norm",type=float, default=3.0)
    parser.add_argument("--model_complexity", type=int, default=48,
                        help="Scales hidden dim: size=complexity*16 (default 48→26M)")
    parser.add_argument("--num_workers",  type=int, default=2)
    parser.add_argument("--log_every",    type=int, default=50)
    parser.add_argument("--resume",       default=None,
                        help="Path to checkpoint .pt to resume from (use latest.pt)")
    parser.add_argument("--no_amp",       action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--no_compile",   action="store_true",
                        help="Disable torch.compile")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        compute  = torch.cuda.get_device_capability(0)
        print(f"GPU:    {gpu_name}")
        print(f"VRAM:   {gpu_mem:.1f} GB")
        print(f"Compute capability: {compute[0]}.{compute[1]}")
        if compute[0] >= 8:
            print(f"Architecture: Ampere+ (TF32 + BF16 available)")
        elif compute[0] >= 7:
            if compute[1] >= 5:
                print(f"Architecture: Turing (FP16 AMP available)")
            else:
                print(f"Architecture: Volta (FP16 AMP available)")
        # Recommended batch sizes
        if gpu_mem >= 70:
            rec_bs = "16–32"
        elif gpu_mem >= 35:
            rec_bs = "16"
        elif gpu_mem >= 14:
            rec_bs = "8"
        else:
            rec_bs = "4"
        print(f"Recommended batch_size: {rec_bs}")

    # Default runs_dir to Drive (NOT local /content/ which is wiped on disconnect)
    if args.runs_dir is None:
        args.runs_dir = str(Path(args.maestro_root).parent / "runs")
        print(f"  runs_dir defaulting to: {args.runs_dir}")

    # Run directory
    run_dir = RunDirectory(args.runs_dir, args.run_name)

    # Save config
    cfg = vars(args).copy()
    cfg["device"] = str(device)
    run_dir.save_config(cfg)

    # DataLoaders
    cache_dir = args.cache_dir or str(Path(args.maestro_root) / "cache")
    print("\nBuilding DataLoaders...")
    train_loader = get_dataloader(
        maestro_root=args.maestro_root, split="train",
        batch_size=args.batch_size, num_workers=args.num_workers,
        cache_dir=cache_dir, max_files=args.max_files,
        use_augmentation=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = get_dataloader(
        maestro_root=args.maestro_root, split="validation",
        batch_size=args.batch_size, num_workers=args.num_workers,
        cache_dir=cache_dir, max_files=args.max_files,
        use_augmentation=False,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    # Model
    model = OnsetsAndFrames(model_complexity=args.model_complexity)
    print(f"\nModel: OnsetsAndFrames  complexity={args.model_complexity}")
    print(f"Parameters: {model.count_parameters():,}")

    # Trainer (before resume so we can restore optimizer/scheduler into it)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
        lr=args.lr,
        pos_weight=args.pos_weight,
        max_grad_norm=args.max_grad_norm,
        log_every=args.log_every,
        use_amp=not args.no_amp,
        use_compile=not args.no_compile,
    )

    # Resume from checkpoint — restore EVERYTHING
    start_epoch = 1
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            print(f"WARNING: Resume checkpoint not found: {ckpt_path}")
            print("         Starting from scratch.")
        else:
            print(f"\nResuming from: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location=device)

            # Restore model weights (handle torch.compile wrapper)
            model_to_load = trainer.model
            if hasattr(trainer.model, '_orig_mod'):
                model_to_load = trainer.model._orig_mod
            model_to_load.load_state_dict(ckpt["model_state"])

            # Restore optimizer state (momentum buffers, step counts)
            if "optimizer_state" in ckpt:
                trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
                print("  Restored optimizer state (momentum, step counts)")

            # Restore scheduler state (num_bad_epochs, best, etc.)
            if "scheduler_state" in ckpt:
                trainer.scheduler.load_state_dict(ckpt["scheduler_state"])
                print("  Restored scheduler state (patience counter, best)")

            # Restore AMP scaler state (scale factor, growth tracker)
            if "scaler_state" in ckpt and trainer.scaler is not None:
                trainer.scaler.load_state_dict(ckpt["scaler_state"])
                print("  Restored AMP scaler state")

            # Restore training position
            start_epoch = ckpt["epoch"] + 1
            trainer.global_step = ckpt.get("global_step", 0)
            trainer.best_val_loss = ckpt.get("best_val_loss",
                                              ckpt.get("val_loss", float("inf")))

            print(f"  Resuming at epoch {start_epoch}")
            print(f"  Global step: {trainer.global_step}")
            print(f"  Best val loss: {trainer.best_val_loss:.4f}")
            print(f"  Current LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")

    # Train
    trainer.fit(epochs=args.epochs, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
