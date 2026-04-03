"""
models/onsets_frames/model.py — OnsetsAndFrames neural network.

Architecture: jongwook/onsets-and-frames PyTorch implementation, adapted for
this pipeline's tensor conventions (mel input (B,229,T), outputs (B,T,88)).

This file is a faithful reproduction of jongwook/onsets-and-frames
transcriber.py with the following adaptations for this pipeline:
  - Input mel shape is (B, N_MELS, T) = (B, 229, T), transposed internally
    to match jongwook's (B, T, F) convention before entering ConvStack.
  - Output is a Dict[str, Tensor] instead of a 5-tuple, for compatibility
    with OnsetsFramesLoss in train.py.

jongwook improvements over Hawthorne 2018a paper (all 5 reproduced here):
  1. Offset stack  — separate CNN+BiLSTM+Linear+Sigmoid head.
  2. Gradient stop — onset AND offset detached at combined junction.
  3. Increased capacity — model_complexity=48 → model_size=768 → ~26M params.
  4. Per-parameter gradient clipping — handled in train.py.
  5. HTK-style mel with fmin=30 — handled in audio.py.

ConvStack (jongwook transcriber.py lines 12–42):
  Conv2d(1, ch1, 3×3, pad=1) → BN → ReLU          ch1 = model_size // 16
  Conv2d(ch1, ch1, 3×3, pad=1) → BN → ReLU
  MaxPool2d(1, 2) → Dropout(0.25)                   pools freq axis
  Conv2d(ch1, ch2, 3×3, pad=1) → BN → ReLU         ch2 = model_size // 8
  MaxPool2d(1, 2) → Dropout(0.25)                   pools freq axis again
  Linear(ch2 * (F//4), model_size) → Dropout(0.5)

Stack layout (jongwook transcriber.py lines 53–83):
  onset_stack    : ConvStack → BiLSTM → Linear → Sigmoid
  offset_stack   : ConvStack → BiLSTM → Linear → Sigmoid
  frame_stack    : ConvStack → Linear → Sigmoid   (no BiLSTM)
  combined_stack : BiLSTM → Linear → Sigmoid
                   input = cat(onset.detach(), offset.detach(), frame_acoustic)
                   dim = output_features × 3
  velocity_stack : ConvStack → Linear              (NO sigmoid — raw MSE target)

Papers:
  Hawthorne et al. 2018a §3 — base architecture.
  jongwook/onsets-and-frames — all 5 improvements.
  Kim, Kwon & Nam 2025 D3RM — offset head necessity.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.constants import N_KEYS, N_MELS


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class ConvStack(nn.Module):
    """
    Convolutional feature extractor — exact reproduction of jongwook's ConvStack.

    jongwook convention: input (B, 1, T, F) — time × freq as a 2D image.
    This pipeline provides mel as (B, F, T) = (B, 229, T).
    The transpose (B, F, T) → (B, T, F) happens in forward() before view().

    Architecture (jongwook transcriber.py lines 14–42):
      Conv2d(1, ch1, 3, pad=1) → BN → ReLU            ch1 = output_features // 16
      Conv2d(ch1, ch1, 3, pad=1) → BN → ReLU
      MaxPool2d(1, 2) → Dropout(0.25)                   halve freq
      Conv2d(ch1, ch2, 3, pad=1) → BN → ReLU           ch2 = output_features // 8
      MaxPool2d(1, 2) → Dropout(0.25)                   halve freq again
      Flatten → Linear(ch2 * F//4, output_features) → Dropout(0.5)

    With model_complexity=48 (output_features=768):
      ch1 = 768 // 16 = 48
      ch2 = 768 // 8  = 96
      FC input = 96 × (229 // 4) = 96 × 57 = 5472

    Input  shape: (B, N_MELS, T) — this pipeline's mel convention
    Output shape: (B, T, output_features)
    """

    def __init__(self, input_features: int, output_features: int) -> None:
        super().__init__()

        ch1 = output_features // 16   # 48 at complexity=48
        ch2 = output_features // 8    # 96 at complexity=48

        # jongwook: input is (B, 1, T, F) — MaxPool2d(1,2) pools along F
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, ch1, (3, 3), padding=1),
            nn.BatchNorm2d(ch1),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(ch1, ch1, (3, 3), padding=1),
            nn.BatchNorm2d(ch1),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),       # (B, ch1, T, F//2)
            nn.Dropout(0.25),
            nn.Conv2d(ch1, ch2, (3, 3), padding=1),
            nn.BatchNorm2d(ch2),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),       # (B, ch2, T, F//4)
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear(ch2 * (input_features // 4), output_features),
            nn.Dropout(0.5),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, N_MELS, T) — log-mel spectrogram batch (this pipeline's
                 convention: freq-first).

        Returns:
            (B, T, output_features)
        """
        # Transpose to jongwook convention: (B, F, T) → (B, T, F)
        x = mel.transpose(1, 2)             # (B, T, F)
        # Add channel dim: (B, T, F) → (B, 1, T, F)  — jongwook's view()
        x = x.unsqueeze(1)                  # (B, 1, T, F)

        x = self.cnn(x)                      # (B, ch2, T, F//4)

        # jongwook: x.transpose(1, 2).flatten(-2) → (B, T, ch2 * F//4)
        x = x.transpose(1, 2)               # (B, T, ch2, F//4)
        x = x.flatten(-2)                   # (B, T, ch2 * F//4)
        # x = x.transpose(1, 2).contiguous()  # (B, T, ch2, F//4)  ← now contiguous
        # x = x.flatten(-2)                   # (B, T, ch2 * F//4)

        x = self.fc(x)                       # (B, T, output_features)
        return x


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM sequence model.

    jongwook lstm.py: BiLSTM(input_size, hidden_size) where hidden_size is
    per-direction, so total output = hidden_size × 2.

    In the main model, it is called as:
        sequence_model = lambda in_size, out_size: BiLSTM(in_size, out_size // 2)
    so hidden_size = out_size // 2 per direction → concat = out_size total.

    Args:
        input_size:  Feature dim of each time step.
        output_size: Hidden size PER DIRECTION.  Total output = output_size × 2.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=output_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_size)
        Returns:
            (B, T, output_size * 2)
        """
        out, _ = self.lstm(x)
        return out  #.contiguous()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class OnsetsAndFrames(nn.Module):
    """
    Onsets and Frames piano transcription model.

    Exact reproduction of jongwook/onsets-and-frames transcriber.py with
    all 5 MAESTRO-paper improvements, adapted for this pipeline's I/O format.

    Args:
        input_features:   Number of mel bins (default N_MELS=229).
        output_features:  Number of piano keys (default N_KEYS=88).
        model_complexity: Scales hidden size: model_size = complexity × 16.
                          Default 48 → model_size=768 → ~26M parameters.

    Forward:
        Input:  mel (B, N_MELS, T) — (B, 229, T) log-mel spectrogram
        Output: Dict with keys onset, frame, offset, velocity
                onset/frame/offset: (B, T, 88) post-sigmoid [0,1]
                velocity:           (B, T, 88) raw (no sigmoid) — MSE target
    """

    def __init__(
        self,
        input_features:   int = N_MELS,
        output_features:  int = N_KEYS,
        model_complexity: int = 48,
    ) -> None:
        super().__init__()

        model_size = model_complexity * 16   # 768 at default

        # jongwook: sequence_model = lambda in, out: BiLSTM(in, out // 2)
        def _seq(in_size: int, out_size: int) -> BiLSTM:
            return BiLSTM(in_size, out_size // 2)

        # ------------------------------------------------------------------
        # Onset stack: ConvStack → BiLSTM → Linear → Sigmoid
        # (jongwook transcriber.py line 57)
        # ------------------------------------------------------------------
        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            _seq(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # Offset stack: same topology as onset — separate weights
        # (jongwook improvement 1: offset head)
        # (jongwook transcriber.py line 63)
        # ------------------------------------------------------------------
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            _seq(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # Frame acoustic stack: ConvStack → Linear → Sigmoid  (no BiLSTM)
        # (jongwook transcriber.py line 69)
        # ------------------------------------------------------------------
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # Combined stack: BiLSTM → Linear → Sigmoid
        # Input: cat(onset.detach(), offset.detach(), frame_acoustic)
        #        dim = output_features × 3
        # (jongwook transcriber.py line 75: output_features * 3)
        # (jongwook improvement 2: gradient stopping on BOTH onset + offset)
        # ------------------------------------------------------------------
        self.combined_stack = nn.Sequential(
            _seq(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # Velocity stack: ConvStack → Linear  (NO Sigmoid)
        # jongwook transcriber.py line 80: no sigmoid on velocity.
        # Raw output trained with masked MSE against velocity/128 targets.
        # ------------------------------------------------------------------
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
        )

    # -----------------------------------------------------------------------

    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass — reproduces jongwook transcriber.py lines 85–94.

        Args:
            mel: FloatTensor (B, N_MELS, T) — log-mel spectrogram batch.
                 This pipeline convention is (B, 229, T); ConvStack handles
                 the transpose to jongwook's (B, T, F) internally.

        Returns:
            Dict[str, Tensor] with keys:
              "onset"    : (B, T, 88) — onset probabilities    [0,1]  (post-sigmoid)
              "frame"    : (B, T, 88) — combined frame probs   [0,1]  (post-sigmoid)
              "offset"   : (B, T, 88) — offset probabilities   [0,1]  (post-sigmoid)
              "velocity" : (B, T, 88) — velocity predictions   (raw, no sigmoid)

        Note:
            onset/frame/offset are post-sigmoid → train.py uses BCELoss.
            velocity has no sigmoid → train.py uses masked MSE directly.
            The decode.py clips velocity to [1, 127] after scaling by 128,
            so out-of-range raw values are handled safely.
        """
        # --- Onset ---
        onset_pred = self.onset_stack(mel)            # (B, T, 88)

        # --- Offset ---
        offset_pred = self.offset_stack(mel)          # (B, T, 88)

        # --- Frame acoustic (from mel only, no BiLSTM) ---
        activation_pred = self.frame_stack(mel)       # (B, T, 88)

        # --- Combined frame ---
        # jongwook line 92: cat([onset.detach(), offset.detach(), activation])
        # Both onset and offset gradients are stopped (improvement 2).
        combined_input = torch.cat(
            [onset_pred.detach(), offset_pred.detach(), activation_pred],
            dim=-1,
        )                                              # (B, T, 88*3)
        frame_pred = self.combined_stack(combined_input)  # (B, T, 88)

        # --- Velocity (raw, no sigmoid) ---
        velocity_pred = self.velocity_stack(mel)      # (B, T, 88)

        return {
            "onset":    onset_pred,
            "frame":    frame_pred,
            "offset":   offset_pred,
            "velocity": velocity_pred,
        }

    # -----------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
