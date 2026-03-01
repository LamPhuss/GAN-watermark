"""
kgw_discriminator.py — Trainable Discriminator wrapping KGW z-score.

Two modes:
  1. Z-score Discriminator (non-trainable): Uses raw z-score from KGW hash.
     Reward = sigmoid((z - threshold) / temperature).
     This gives smooth reward signal directly from watermark strength.

  2. LSTM Discriminator (trainable): Like UPV detector but trained on
     (real_watermarked=1, fake/natural=0). Can learn patterns beyond z-score.

For GAN pipeline, mode 1 is preferred because:
  - Z-score is well-calibrated (known statistical properties)
  - No risk of D being "too strong" since reward is smooth
  - Spoofer can directly target green tokens
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from watermark.kgw_watermark import KGWDetector


class KGWDiscriminator(nn.Module):
    """
    Discriminator for GAN pipeline using KGW z-score.

    Provides reward signal for REINFORCE:
      reward = sigmoid((z_score - z_center) / temperature)

    z_center and temperature can be tuned:
      - z_center=2.0, temp=2.0: gentle (reward>0.5 when z>2)
      - z_center=4.0, temp=1.0: strict (reward>0.5 only when z>4)
    """

    def __init__(
        self,
        vocab_size: int,
        gamma: float = 0.25,
        delta: float = 2.0,
        context_width: int = 4,
        hash_key: int = 15485863,
        z_threshold: float = 4.0,
        z_center: float = 2.0,
        temperature: float = 2.0,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.z_center = z_center
        self.temperature = temperature
        self.z_threshold = z_threshold

        self.detector = KGWDetector(
            vocab_size=vocab_size,
            gamma=gamma, delta=delta,
            context_width=context_width,
            hash_key=hash_key,
            z_threshold=z_threshold,
            device=device,
        )

        # Dummy parameter so optimizer doesn't complain
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        token_ids: torch.LongTensor,
        lengths: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward for batch of sequences.

        Args:
            token_ids: (batch, seq_len) token IDs
            lengths: (batch,) actual lengths (excluding padding)

        Returns:
            (batch,) reward in [0, 1]
        """
        batch_size = token_ids.size(0)
        rewards = torch.zeros(batch_size, device=self.device)

        for b in range(batch_size):
            seq_len = lengths[b].item() if lengths is not None else token_ids.size(1)
            ids = token_ids[b, :seq_len]

            if len(ids) < self.detector.context_width + 1:
                rewards[b] = 0.0
                continue

            z = self.detector.get_z_score(ids)
            # Smooth reward: sigmoid mapping
            rewards[b] = torch.sigmoid(
                torch.tensor((z - self.z_center) / self.temperature, device=self.device)
            )

        return rewards

    def get_reward(
        self,
        token_ids: torch.LongTensor,
        prompt_lengths: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Get reward for MC Search. Extracts generated part only.

        Args:
            token_ids: (batch, seq_len) full sequences (prompt + generated)
            prompt_lengths: (batch,) prompt lengths

        Returns:
            (batch,) reward in [0, 1]
        """
        batch_size = token_ids.size(0)
        rewards = torch.zeros(batch_size, device=self.device)

        for b in range(batch_size):
            if prompt_lengths is not None:
                gen_ids = token_ids[b, prompt_lengths[b].item():]
            else:
                gen_ids = token_ids[b]

            if len(gen_ids) < self.detector.context_width + 1:
                rewards[b] = 0.0
                continue

            z = self.detector.get_z_score(gen_ids)
            rewards[b] = torch.sigmoid(
                torch.tensor((z - self.z_center) / self.temperature, device=self.device)
            )

        return rewards

    def detect_batch(self, token_ids: torch.LongTensor, lengths=None) -> List[Dict]:
        """Detect watermark for batch — returns full detection dicts."""
        results = []
        batch_size = token_ids.size(0)
        for b in range(batch_size):
            seq_len = lengths[b].item() if lengths is not None else token_ids.size(1)
            ids = token_ids[b, :seq_len]
            results.append(self.detector.detect(ids))
        return results

    def get_trainable_params(self):
        """No trainable params — z-score is deterministic."""
        return []

    def count_params(self):
        return {"total": 0, "trainable": 0, "frozen": 0}
