"""
models/onsets_frames/train.py — Training harness for OnsetsAndFrames.

Design:
  - Every run gets its own directory under runs/<run_name>/
    containing: checkpoints/, plots/, metrics.json, config.json, timing_summary.json
  - Loss curves (train + val per epoch) are saved as PNG after every epoch
    so training progress is always visible for the report.
  - All epoch losses + timing data are stored in metrics.json and updated after
    every epoch so the file is always up-to-date even if Colab disconnects.
  - The model uses post-sigmoid outputs, so BCELoss is used (not BCEWithLogits).
  - Per-parameter gradient clipping matches jongwook's implementation.
  - ReduceLROnPlateau scheduler halves LR after 3 epochs without improvement.
  - EVERY epoch is checkpointed (latest.pt + best.pt) for Colab crash safety.
  - Full resume support: model, optimizer, scheduler, epoch, best_val_loss,
    global_step, timing history, and metrics history are all restored.

Loss function:
  onset/frame/offset — weighted manual BCE (pos_weight configurable per head)
  velocity           — masked MSE at onset positions only

  Default pos_weight=1.0 (plain BCE) matches jongwook at full-dataset scale.
  For small subsets (≤400 files), set pos_weight_onset/offset=25.0,
  pos_weight_frame=6.0 to prevent near-zero collapse caused by
  ~460:1 onset label sparsity on MAESTRO. Values are set via CONFIG
  in the notebook and passed through Trainer → OnsetsFramesLoss.

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
  jongwook/onsets-and-frames: per-param grad clip, batch_size=8, plain BCE.
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
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
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
    4-head weighted BCE + masked MSE loss for post-sigmoid model outputs.

    Loss design:
      onset/offset — weighted BCE, pos_weight=25.0
                     Onset labels are ~460:1 negative:positive on MAESTRO.
                     pos_weight=25.0 (~5% of true ratio) prevents near-zero
                     collapse on small subsets while keeping precision bounded.

      frame        — weighted BCE, pos_weight=8.0
                     Frame labels are ~33:1 — less sparse than onsets.
                     pos_weight=8.0 (~25% of true ratio) is appropriate.

      velocity     — masked MSE at onset positions only.

    Scale note:
      jongwook and Magenta use plain BCE (pos_weight=1.0) because they
      train on the full MAESTRO dataset for 500,000+ steps, accumulating
      sufficient positive gradient signal through volume alone.
      For subsets ≤400 files, pos_weight is required to prevent collapse.

    Papers:
      Hawthorne 2018a §3.2: weighted onset loss, masked MSE velocity.
      jongwook MAESTRO paper: disabled weighting at full-dataset scale only.
    """

    def __init__(
            self,
            pos_weight_onset:  float = 1.0 ,
            pos_weight_frame:  float = 1.0,
            pos_weight_offset: float = 1.0,
        ) -> None:
            super().__init__()
            self.mse = nn.MSELoss(reduction="none")
            self.pos_weight_onset  = pos_weight_onset
            self.pos_weight_frame  = pos_weight_frame
            self.pos_weight_offset = pos_weight_offset
    def forward(
        self,
        pred:   Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        device = pred["onset"].device

        # Per-head pos_weight based on measured MAESTRO label sparsity:
        #   onset/offset: ~460:1 → pos_weight=25.0
        #   frame:         ~33:1 → pos_weight=8.0
        pos_w_onset  = torch.tensor(self.pos_weight_onset,  device=device)
        pos_w_frame  = torch.tensor(self.pos_weight_frame,  device=device)
        pos_w_offset = torch.tensor(self.pos_weight_offset, device=device)

        def _bce(p, t, pw):
            # Manual weighted BCE — AMP-safe via .float() cast.
            # Equivalent to F.binary_cross_entropy with pos_weight
            # but bypasses the AMP guard on F.binary_cross_entropy.
            p = p.float().clamp(1e-7, 1 - 1e-7)
            t = t.float()
            return -(pw * t * torch.log(p) + (1 - t) * torch.log(1 - p)).mean()

        loss_onset  = _bce(pred["onset"],  target["onset"],  pos_w_onset)
        loss_frame  = _bce(pred["frame"],  target["frame"],  pos_w_frame)
        loss_offset = _bce(pred["offset"], target["offset"], pos_w_offset)

        # Velocity: masked MSE at onset positions only
        mask     = (target["onset"] > 0.5).float()
        n_active = mask.sum().clamp(min=1.0)
        vel_mse  = self.mse(pred["velocity"].float(), target["velocity"].float())
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
            timing_summary.json ← timing statistics (updated after training)
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
        self.timing_path  = self.root / "timing_summary.json"
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
                "epoch_time_seconds": [],
            }

    def save_config(self, cfg: dict) -> None:
        with open(self.config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    def log_epoch(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses:   Dict[str, float],
        lr: float,
        epoch_time: float,
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
        h["epoch_time_seconds"].append(epoch_time)
        with open(self.metrics_path, "w") as f:
            json.dump(h, f, indent=2)

    def save_timing_summary(self, total_training_time: float) -> None:
        """Save comprehensive timing statistics to timing_summary.json."""
        h = self._history
        epoch_times = h.get("epoch_time_seconds", [])

        if not epoch_times:
            return

        summary = {
            "total_training_time_hours": total_training_time / 3600,
            "total_training_time_seconds": total_training_time,
            "total_epochs_trained": len(epoch_times),
            "mean_epoch_time_seconds": sum(epoch_times) / len(epoch_times),
            "min_epoch_time_seconds": min(epoch_times),
            "max_epoch_time_seconds": max(epoch_times),
            "first_epoch_time_seconds": epoch_times[0] if epoch_times else 0,
            "last_epoch_time_seconds": epoch_times[-1] if epoch_times else 0,
            "per_epoch_times_seconds": epoch_times,
        }

        with open(self.timing_path, "w") as f:
            json.dump(summary, f, indent=2)

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
    timing instrumentation, and Drive-safe checkpointing.

    GPU performance features:
      - cuDNN benchmark mode
      - Automatic Mixed Precision (AMP) with GradScaler
      - torch.compile() on PyTorch 2.0+
      - Non-blocking CPU→GPU transfers with pin_memory
      - set_to_none=True for zero_grad
      - tf32 on Ampere+ GPUs

    Checkpointing strategy (Colab crash-safe):
      - latest.pt  — saved EVERY epoch (always resumable)
      - best.pt    — saved when val_loss improves (best model for eval)
      - epoch_NNN_valloss_X.XXXX.pt — kept for each best (audit trail)

    Timing:
      - epoch_time_seconds saved per epoch in metrics.json
      - total_training_time_hours saved in timing_summary.json
      - All timing data saved in checkpoints for resume support
    """

    def __init__(
        self,
        model:          nn.Module,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        device:         torch.device,
        run_dir:        RunDirectory,
        lr:             float = 6e-4,
        max_grad_norm:  float = 3.0,
        log_every:      int   = 50,
        use_amp:        bool  = True,
        use_compile:    bool  = True,
        pos_weight_onset:  float = 1.0,   # ADD
        pos_weight_frame:  float = 1.0,    # ADD
        pos_weight_offset: float = 1.0,   # ADD
    ) -> None:
        self.device         = device
        self.run_dir        = run_dir
        self.max_grad_norm  = max_grad_norm
        self.log_every      = log_every

        # ------------------------------------------------------------------
        # GPU performance setup
        # ------------------------------------------------------------------
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

            if torch.cuda.get_device_capability(0)[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # Move model to device
        self.model = model.to(device)

        # ------------------------------------------------------------------
        # AMP (Automatic Mixed Precision)
        # ------------------------------------------------------------------
        self.use_amp = use_amp and (device.type == "cuda")
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # ------------------------------------------------------------------
        # torch.compile (PyTorch 2.0+)
        # ------------------------------------------------------------------
        self.used_compile = False
        if use_compile and device.type == "cuda":
            torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
            if torch_version >= (2, 0):
                try:
                    self.model = torch.compile(self.model)
                    self.used_compile = True
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # Optimizer, scheduler, loss
        # ------------------------------------------------------------------
        self.criterion = OnsetsFramesLoss(
            pos_weight_onset=pos_weight_onset,
            pos_weight_frame=pos_weight_frame,
            pos_weight_offset=pos_weight_offset,
        )
        self.optimizer  = Adam(model.parameters(), lr=lr)
        self.scheduler = StepLR(
            self.optimizer,
            step_size=10000,   # decay every 10k gradient steps
            gamma=0.98,        # multiply LR by 0.98 each time
        )
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.global_step    = 0
        self.best_val_loss: float = float("inf")
        self.cumulative_training_time: float = 0.0  # total seconds across all sessions

    # -----------------------------------------------------------------------

    def _move(self, batch: dict) -> dict:
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    # -----------------------------------------------------------------------

    def _per_param_clip(self) -> None:
        for p in self.model.parameters():
            if p.grad is not None:
                nn.utils.clip_grad_norm_([p], max_norm=self.max_grad_norm)

    # -----------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {
            "total": 0., "onset": 0., "frame": 0., "offset": 0., "velocity": 0.
        }
        n_batches = 0
        t0 = time.time()

        for batch in self.train_loader:
            batch = self._move(batch)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred   = self.model(batch["mel"])
            losses = self.criterion(pred, batch)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                self.scaler.scale(losses["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                self._per_param_clip()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["total"].backward()
                self._per_param_clip()
                self.optimizer.step()
            
            self.scheduler.step()

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
        return means

    # -----------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
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
            "cumulative_training_time": self.cumulative_training_time,
        }

        if self.scaler is not None:
            ckpt["scaler_state"] = self.scaler.state_dict()

        # Always save latest.pt
        latest_path = self.run_dir.latest_checkpoint_path()
        torch.save(ckpt, latest_path)

        if epoch % 5 == 0:
            periodic_path = self.run_dir.checkpoints / f"epoch_{epoch:03d}.pt"
            torch.save(ckpt, periodic_path)

        if is_best:
            named_path = self.run_dir.checkpoint_path(epoch, val_loss)
            torch.save(ckpt, named_path)
            best_path = self.run_dir.best_checkpoint_path()
            torch.save(ckpt, best_path)

    # -----------------------------------------------------------------------

    def fit(self, epochs: int = 30, start_epoch: int = 1) -> None:
        print(f"\n{'='*60}")
        print(f"Starting training: epochs {start_epoch}→{epochs}")
        print(f"Run directory: {self.run_dir.root}")
        print(f"Best val loss so far: {self.best_val_loss:.4f}")
        print(f"Cumulative training time: {self.cumulative_training_time:.1f}s")
        print(f"{'='*60}\n")

        session_start_time = time.time()
        best_epoch = None

        for epoch in range(start_epoch, epochs + 1):
            t_epoch_start = time.time()

            print(f"--- Epoch {epoch}/{epochs} ---")

            train_losses = self.train_epoch(epoch)
            val_losses   = self.validate(epoch)
            current_lr   = self.optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - t_epoch_start
            self.cumulative_training_time += epoch_time

            # Log to metrics.json
            self.run_dir.log_epoch(epoch, train_losses, val_losses, current_lr, epoch_time)

            # Determine if this is a new best
            is_best = val_losses["total"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses["total"]
                best_epoch = epoch

            # Save loss curves PNG
            self.run_dir.save_loss_curves(epoch)

            # Save checkpoint
            self.save_checkpoint(epoch, val_losses["total"], is_best=is_best)

            # Concise per-epoch output
            best_marker = " ★ NEW BEST" if is_best else ""
            epochs_left = epochs - epoch
            eta_str = f"  ETA: ~{(epoch_time * epochs_left) / 60:.0f}min" if epochs_left > 0 else ""
            print(
                f"  train_loss={train_losses['total']:.4f}  val_loss={val_losses['total']:.4f}  "
                f"lr={current_lr:.2e}  time={epoch_time:.0f}s{eta_str}{best_marker}\n"
            )

        # --- End of training: comprehensive summary ---
        total_session_time = time.time() - session_start_time
        self.run_dir.save_timing_summary(self.cumulative_training_time)

        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Run directory     : {self.run_dir.root}")
        print(f"  Epochs trained    : {start_epoch}→{epochs} ({epochs - start_epoch + 1} epochs)")
        print(f"  Best val loss     : {self.best_val_loss:.4f}" +
              (f" (epoch {best_epoch})" if best_epoch else ""))
        print(f"  Session time      : {total_session_time:.1f} seconds")
        print(f"  Total train time  : {self.cumulative_training_time:.1f} seconds")
        epoch_times = self.run_dir._history.get("epoch_time_seconds", [])
        if epoch_times:
            avg_epoch = sum(epoch_times) / len(epoch_times)
            print(f"  Avg epoch time    : {avg_epoch:.1f} seconds")
        print()

        # Verify saved files
        print(f"  Saved files:")
        for label, path in [
            ("Config",           self.run_dir.config_path),
            ("Metrics",          self.run_dir.metrics_path),
            ("Timing summary",   self.run_dir.timing_path),
            ("Latest checkpoint", self.run_dir.latest_checkpoint_path()),
            ("Best checkpoint",  self.run_dir.best_checkpoint_path()),
        ]:
            exists = path.exists()
            status = "✓" if exists else "✗ MISSING"
            print(f"    {status} {label}: {path}")

        # List all epoch checkpoints
        epoch_ckpts = sorted(self.run_dir.checkpoints.glob("epoch_*.pt"))
        if epoch_ckpts:
            print(f"    ✓ Epoch checkpoints ({len(epoch_ckpts)}):")
            for ck in epoch_ckpts:
                print(f"      - {ck.name}")

        # List loss curve plots
        plots = sorted(self.run_dir.plots.glob("loss_curves_*.png"))
        if plots:
            print(f"    ✓ Loss curve plots ({len(plots)}) saved in: {self.run_dir.plots}")

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
    parser.add_argument("--pos_weight_onset",  type=float, default=1.0,
                    help="BCE pos_weight for onset head (default 1.0 = plain BCE)")
    parser.add_argument("--pos_weight_frame",  type=float, default=1.0,
                        help="BCE pos_weight for frame head (default 1.0 = plain BCE)")
    parser.add_argument("--pos_weight_offset", type=float, default=1.0,
                        help="BCE pos_weight for offset head (default 1.0 = plain BCE)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        compute  = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu_name} | VRAM: {gpu_mem:.1f} GB | Compute: {compute[0]}.{compute[1]}")

    # Default runs_dir to Drive
    if args.runs_dir is None:
        args.runs_dir = str(Path(args.maestro_root).parent / "runs")

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
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    model = OnsetsAndFrames(model_complexity=args.model_complexity)
    print(f"Model: OnsetsAndFrames (complexity={args.model_complexity}, params={model.count_parameters():,})")

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        log_every=args.log_every,
        use_amp=not args.no_amp,
        use_compile=not args.no_compile,
        pos_weight_onset  = args.pos_weight_onset,   
        pos_weight_frame  = args.pos_weight_frame,   
        pos_weight_offset = args.pos_weight_offset,  
    )

    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            print(f"WARNING: Resume checkpoint not found: {ckpt_path}")
            print("         Starting from scratch.")
        else:
            print(f"\nResuming from: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location=device)

            model_to_load = trainer.model
            if hasattr(trainer.model, '_orig_mod'):
                model_to_load = trainer.model._orig_mod
            model_to_load.load_state_dict(ckpt["model_state"])

            if "optimizer_state" in ckpt:
                trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                trainer.scheduler.load_state_dict(ckpt["scheduler_state"])
            if "scaler_state" in ckpt and trainer.scaler is not None:
                trainer.scaler.load_state_dict(ckpt["scaler_state"])

            start_epoch = ckpt["epoch"] + 1
            trainer.global_step = ckpt.get("global_step", 0)
            trainer.best_val_loss = ckpt.get("best_val_loss",
                                              ckpt.get("val_loss", float("inf")))
            trainer.cumulative_training_time = ckpt.get("cumulative_training_time", 0.0)

            print(f"  Epoch: {start_epoch} | Step: {trainer.global_step} | "
                  f"Best val: {trainer.best_val_loss:.4f} | "
                  f"Cumulative time: {trainer.cumulative_training_time / 3600:.2f}h")

    # Train
    trainer.fit(epochs=args.epochs, start_epoch=start_epoch)


if __name__ == "__main__":
    main()