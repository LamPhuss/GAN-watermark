# ============================================================
# upv_discriminator.py
# Description: Wrapper around the original UPV Detector for GAN
#   adversarial training. Makes the UPV detector ROBUST against
#   spoofing attacks by training it to distinguish:
#     - Real watermarked text (label=1)
#     - Fake/spoofed watermarked text (label=0)
#     - Natural text (label=0)
#
# Architecture:
#   token_ids → binary encoding → UPVSubNet (FROZEN) → LSTM → FC → sigmoid
#
# The binary_classifier (UPVSubNet) acts as the "shared embedding"
# from the UPV paper and MUST stay frozen during adversarial training
# to maintain compatibility with the UPV Generator.
# ============================================================

import torch
import torch.nn as nn
from typing import Optional, List

from upv.network_model import UPVDetector


class UPVDiscriminatorWrapper(nn.Module):
    """
    Wraps the original UPV Detector for use as GAN Discriminator.

    The UPV Detector takes binary-encoded tokens (batch, seq_len, bit_number)
    and outputs watermark probability (batch, 1).

    This wrapper handles:
      1. Token ID → binary vector conversion
      2. Freezing the shared embedding (binary_classifier / UPVSubNet)
      3. Providing get_reward() for RL policy gradient
      4. Exposing only trainable params (LSTM + FC) for optimizer
    """

    def __init__(
        self,
        bit_number: int = 16,
        detector_weights_path: Optional[str] = None,
        freeze_embedding: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.bit_number = bit_number
        self.device_str = device

        # Load UPV Detector (original architecture)
        self.detector = UPVDetector(bit_number=bit_number)

        if detector_weights_path is not None:
            print(f"[UPVDiscWrapper] Loading detector weights: {detector_weights_path}")
            state = torch.load(detector_weights_path, map_location="cpu")
            self.detector.load_state_dict(state)
            print("[UPVDiscWrapper] Weights loaded successfully.")
        else:
            print("[UPVDiscWrapper] WARNING: No weights loaded — random init.")

        if freeze_embedding:
            self.freeze_shared_embedding()

    # ── Freeze / Unfreeze ──

    def freeze_shared_embedding(self) -> None:
        """
        Freeze the binary_classifier (UPVSubNet) — the shared embedding.

        This is CRITICAL: the UPV paper shows that fine-tuning the shared
        embedding drops F1 by 11.1%. During adversarial training, only
        the LSTM and FC layers should be updated.
        """
        for param in self.detector.binary_classifier.parameters():
            param.requires_grad = False
        print("[UPVDiscWrapper] Shared embedding (binary_classifier) FROZEN.")

    def unfreeze_shared_embedding(self) -> None:
        """Unfreeze shared embedding (use ONLY during pre-training if needed)."""
        for param in self.detector.binary_classifier.parameters():
            param.requires_grad = True
        print("[UPVDiscWrapper] Shared embedding UNFROZEN.")

    def freeze_all(self) -> None:
        """Freeze all parameters (for pure inference)."""
        for param in self.detector.parameters():
            param.requires_grad = False

    # ── Token → Binary conversion ──

    def _tokens_to_binary(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        Convert token IDs to binary vectors.

        Args:
            token_ids: (batch, seq_len) token IDs

        Returns:
            (batch, seq_len, bit_number) float tensor of binary vectors

        Note: Token IDs are taken modulo 2^bit_number to fit in
        the binary representation. With bit_number=16, supports
        vocab up to 65536 (OPT-1.3B has 50265 — fits).
        """
        max_val = 2 ** self.bit_number
        # Clamp to valid range
        clamped = token_ids % max_val  # (batch, seq_len)

        batch_size, seq_len = clamped.shape
        binary = torch.zeros(
            batch_size, seq_len, self.bit_number,
            device=clamped.device, dtype=torch.float32,
        )

        # Vectorized binary conversion
        for bit_pos in range(self.bit_number):
            # Extract bit at position (MSB first to match int_to_bin_list)
            shift = self.bit_number - 1 - bit_pos
            binary[:, :, bit_pos] = ((clamped >> shift) & 1).float()

        return binary

    # ── Forward ──

    def forward(
        self,
        token_ids: torch.LongTensor,
        lengths: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: token_ids → binary → UPVDetector → probability.

        Args:
            token_ids: (batch, seq_len) token IDs
            lengths:   (batch,) actual lengths — used to mask padding.
                       If None, processes full sequences.

        Returns:
            (batch, 1) probability that each sequence is REAL watermarked
        """
        # Convert to binary encoding
        binary = self._tokens_to_binary(token_ids)  # (batch, seq, bit_number)

        # Mask padding positions to zero (if lengths provided)
        if lengths is not None:
            mask = torch.arange(
                binary.size(1), device=binary.device
            ).unsqueeze(0) < lengths.unsqueeze(1)  # (batch, seq)
            binary = binary * mask.unsqueeze(-1).float()

        # Forward through UPV Detector
        prob = self.detector(binary)  # (batch, 1)
        return prob

    # ── Reward for RL ──

    def get_reward(
        self,
        token_ids: torch.LongTensor,
        lengths: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Get reward signal for policy gradient.

        Returns D(x) — probability that D thinks x is REAL watermarked.
        Higher = Attacker successfully fooled D.

        Args:
            token_ids: (batch, seq_len) token IDs
            lengths:   (batch,) sequence lengths

        Returns:
            (batch,) reward values in [0, 1]
        """
        with torch.no_grad():
            prob = self.forward(token_ids, lengths)  # (batch, 1)
        return prob.squeeze(-1)  # (batch,)

    # ── Trainable params ──

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get only trainable parameters (LSTM + FC, NOT binary_classifier)."""
        return [p for p in self.detector.parameters() if p.requires_grad]

    def count_params(self) -> dict:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.detector.parameters())
        trainable = sum(p.numel() for p in self.detector.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}
