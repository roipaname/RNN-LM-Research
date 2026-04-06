"""
embedding.py
------------
EmbeddingLayer: maps integer token indices to dense embedding vectors.

Forward:  tokens (T,) -> embeddings (T, embed_dim)
Backward: dE (T, embed_dim) -> accumulates gradient into dE_matrix
"""

import numpy as np


class EmbeddingLayer:
    """
    Learnable character embedding table.

    Attributes
    ----------
    vocab_size : int
        Number of unique tokens (characters + special tokens).
    embed_dim : int
        Dimensionality of each embedding vector.
    E : ndarray of shape (vocab_size, embed_dim)
        The embedding matrix. Initialised with small random values.
    """

    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 0):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        rng = np.random.default_rng(seed)
        # Scale: sqrt(1 / embed_dim) — keeps initial norms reasonable
        scale = np.sqrt(1.0 / embed_dim)
        self.E: np.ndarray = rng.normal(0.0, scale, (vocab_size, embed_dim))

        # Cache for backward pass
        self._last_tokens: np.ndarray | None = None

        # Gradient accumulator (reset each step by the caller)
        self.dE: np.ndarray = np.zeros_like(self.E)

    # ──────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        tokens : ndarray of shape (T,) — integer indices

        Returns
        -------
        embeddings : ndarray of shape (T, embed_dim)
        """
        self._last_tokens = tokens  # cache for backward
        return self.E[tokens]       # fancy indexing — no copy on read

    # ──────────────────────────────────────────
    # Backward
    # ──────────────────────────────────────────

    def backward(self, d_out: np.ndarray) -> None:
        """
        Accumulate gradients into self.dE.

        Parameters
        ----------
        d_out : ndarray of shape (T, embed_dim)
            Upstream gradient w.r.t. embeddings.
        """
        if self._last_tokens is None:
            raise RuntimeError("backward() called before forward().")
        # np.add.at handles repeated indices correctly (unlike +=)
        np.add.at(self.dE, self._last_tokens, d_out)

    # ──────────────────────────────────────────
    # Parameter / gradient access
    # ──────────────────────────────────────────

    def params(self) -> dict[str, np.ndarray]:
        return {"E": self.E}

    def grads(self) -> dict[str, np.ndarray]:
        return {"E": self.dE}

    def zero_grad(self) -> None:
        self.dE[:] = 0.0

    # ──────────────────────────────────────────
    # Serialisation helpers
    # ──────────────────────────────────────────

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"E": self.E.copy()}

    def load_state_dict(self, d: dict[str, np.ndarray]) -> None:
        self.E = d["E"].copy()
