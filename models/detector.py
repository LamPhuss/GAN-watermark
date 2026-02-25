# ============================================================
# detector.py
# Description: LSTM-based Discriminator (D_φ) for the GAN
#   Distinguishes real watermark (from UPV Generator) vs 
#   fake watermark (from Attacker)
# ============================================================

import torch
import torch.nn as nn
from typing import Optional


class SharedEmbedding(nn.Module):
    """
    Shared embedding layer that converts token IDs to dense vectors.
    This layer can be frozen during adversarial training to prevent
    the discriminator from "cheating" via embedding drift.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) token IDs
        Returns:
            (batch, seq_len, embedding_dim) embedded token vectors
        """
        return self.embedding(token_ids)


class WatermarkDiscriminator(nn.Module):
    """
    LSTM-based Discriminator (D_φ).
    
    Architecture:
        token_ids -> SharedEmbedding (frozen) -> LSTM -> FC -> Sigmoid
    
    Input:  Sequence of token IDs (batch, seq_len)
    Output: Probability that the sequence is REAL watermarked text [0, 1]
    
    The discriminator operates at the sequence level:
    it takes the final LSTM hidden state and passes it through
    fully-connected layers to produce a single probability.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        freeze_embedding: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        
        # Shared embedding (frozen during adversarial training)
        self.shared_embedding = SharedEmbedding(vocab_size, embedding_dim)
        if freeze_embedding:
            self.freeze_embedding()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=False,
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def freeze_embedding(self) -> None:
        """Freeze the shared embedding layer parameters."""
        for param in self.shared_embedding.parameters():
            param.requires_grad = False

    def unfreeze_embedding(self) -> None:
        """Unfreeze the shared embedding layer parameters."""
        for param in self.shared_embedding.parameters():
            param.requires_grad = True

    def forward(
        self, 
        token_ids: torch.LongTensor,
        lengths: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            token_ids: (batch, seq_len) token IDs
            lengths: (batch,) actual sequence lengths (for packing)
        
        Returns:
            (batch, 1) probability that each sequence is REAL watermarked
        """
        batch_size = token_ids.size(0)
        
        # Embed tokens
        embedded = self.shared_embedding(token_ids)  # (batch, seq_len, emb_dim)
        
        # Pack sequences if lengths provided (for variable-length sequences)
        if lengths is not None:
            # Sort by length (descending) for pack_padded_sequence
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            embedded_sorted = embedded[sort_idx]
            
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded_sorted, 
                lengths_sorted.cpu().clamp(min=1),  # Ensure min length 1
                batch_first=True, 
                enforce_sorted=True,
            )
            lstm_out_packed, (h_n, c_n) = self.lstm(packed)
            
            # Unsort
            _, unsort_idx = sort_idx.sort()
            h_n = h_n[:, unsort_idx, :]
        else:
            lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Take the final hidden state from the last LSTM layer
        final_hidden = h_n[-1]  # (batch, hidden_dim)
        
        # Classification
        out = self.dropout(final_hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        prob = self.sigmoid(out)  # (batch, 1) values in [0, 1]
        
        return prob

    def get_trainable_params(self):
        """Get only trainable parameters (excluding frozen embedding)."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_reward(
        self,
        token_ids: torch.LongTensor,
        lengths: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Get reward signal for the Attacker's policy gradient.
        
        The reward is D(x) = probability that D thinks x is REAL.
        Higher reward means the Attacker successfully fooled D.
        
        Args:
            token_ids: (batch, seq_len) generated token IDs from Attacker
            lengths: (batch,) sequence lengths
        
        Returns:
            (batch,) reward values in [0, 1]
        """
        with torch.no_grad():
            prob = self.forward(token_ids, lengths)  # (batch, 1)
        return prob.squeeze(-1)  # (batch,)
