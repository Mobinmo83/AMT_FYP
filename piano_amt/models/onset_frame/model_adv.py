"""
models/onsets_frames/model_adv.py — Adversarial variant of OnsetsAndFrames.

PLACEHOLDER — to be implemented in the next phase.

Planned design:
  - Same CNN+BiLSTM architecture as OnsetsAndFrames in model.py.
  - Additional adversarial training objective (domain-adversarial or
    pitch-adversarial discriminator) applied during training only.
  - Imports OnsetsAndFrames from model.py and wraps it with the adversarial
    discriminator so the base architecture is not duplicated.
  - train.py will detect model_adv via a --model flag and pass
    use_adversarial=True to the Trainer.

Usage (planned):
    from models.onsets_frames.model_adv import OnsetsAndFramesAdv
    model = OnsetsAndFramesAdv(model_complexity=48)
"""

# Not yet implemented — see model.py for the base architecture.
