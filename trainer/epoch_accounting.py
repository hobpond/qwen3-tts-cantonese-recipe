# SPDX-License-Identifier: Apache-2.0
"""Per-epoch loss bookkeeping for sft_12hz_lora_mlflow.py, kept free of torch.

WHY A SEPARATE FILE. BUG-TRAIN-001: the per-batch accounting sat indented under the
epoch-level STEPTIMES block, so it ran once per epoch against the last micro-batch.
"Epoch N finished | Avg Loss" was one batch, and MLflow `train_loss` got at most one
point per epoch. The optimisation was never affected; the record of it was.

The fix is a move, not a rewrite. It lives here so it can be tested without torch,
accelerate or qwen_tts, none of which the review container has. A deterministic
fixture then shows the counts and means, and a structural test pins where the trainer
calls it.

Values may be Python floats or 0-d tensors: float() reads either, the same conversion
the trainer already used (`loss.item()`, `float(talker_loss)`).
"""
from __future__ import annotations


class EpochAccounting:
    """Sums and a batch count for one epoch. Create one per epoch."""

    def __init__(self) -> None:
        self.loss = 0.0
        self.talker = 0.0
        self.sub = 0.0
        self.sub_buggy = 0.0
        self.batches = 0

    def add(self, loss, talker, sub, sub_buggy) -> float:
        """Record ONE micro-batch. Returns the batch loss as a float, for logging."""
        current = float(loss)
        self.loss += current
        self.talker += float(talker)
        self.sub += float(sub)
        self.sub_buggy += float(sub_buggy)
        self.batches += 1
        return current

    def means(self) -> dict | None:
        """Per-batch means, or None for an epoch that recorded nothing."""
        if not self.batches:
            return None
        n = self.batches
        return {"loss": self.loss / n, "talker": self.talker / n,
                "sub": self.sub / n, "sub_buggy": self.sub_buggy / n}
