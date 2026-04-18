"""
positional_encoding.py
----------------------
Sinusoidal positional encoding (Vaswani et al., 2017).

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

The matrix is computed once for a maximum sequence length and cached.
At runtime, call get(seq_len) to retrieve a (seq_len, d_model) slice.

No learnable parameters — this module has no save/load surface.
"""

import numpy as np


class PositionalEncoding:
    """
    Pre-computed sinusoidal positional encoding table.

    Parameters
    ----------
    d_model  : int — embedding dimension (must match the model)
    max_len  : int — maximum sequence length to pre-compute (default 512)
    """

    def __init__(self, d_model: int, max_len: int = 512):
        self.d_model = d_model
        self.max_len = max_len
        self._pe = self._build(max_len, d_model)   # (max_len, d_model)

    # ──────────────────────────────────────────
    # Build
    # ──────────────────────────────────────────

    @staticmethod
    def _build(max_len: int, d_model: int) -> np.ndarray:
        pos = np.arange(max_len)[:, None]                          # (L, 1)
        i   = np.arange(d_model)[None, :]                         # (1, D)
        angles = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
        pe = np.where(i % 2 == 0, np.sin(angles), np.cos(angles))
        return pe.astype(np.float32)                               # (L, D)

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def get(self, seq_len: int) -> np.ndarray:
        """
        Return positional encoding for positions 0 … seq_len-1.

        Parameters
        ----------
        seq_len : int — must be ≤ max_len

        Returns
        -------
        pe : (seq_len, d_model) float32
        """
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_len={self.max_len}. "
                "Increase max_len when constructing PositionalEncoding."
            )
        return self._pe[:seq_len]   # view — no copy
