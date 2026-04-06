"""
sampling.py
-----------
Sampling strategies for inference-time token selection.

Supports:
  - Greedy decoding      (temperature = 0)
  - Temperature scaling  (temperature > 0)
  - Top-k filtering      (top_k > 0)
"""

import numpy as np


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    rng: np.random.Generator | None = None,
) -> int:
    """
    Sample a single token index from logits.

    Parameters
    ----------
    logits      : (V,) — raw log-probabilities (before final softmax)
    temperature : float — scaling factor
                  0  → greedy argmax
                  <1 → sharper distribution (more conservative)
                  >1 → flatter distribution (more creative)
    top_k       : int — keep only the top-k highest logits before sampling
                  0  → no truncation
    rng         : numpy Generator; uses global RNG if None

    Returns
    -------
    token_idx : int
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── Greedy ────────────────────────────────────────────────────
    if temperature == 0.0:
        return int(np.argmax(logits))

    # ── Temperature scaling ───────────────────────────────────────
    logits = logits / temperature

    # ── Top-k filtering ───────────────────────────────────────────
    if top_k > 0:
        top_k = min(top_k, len(logits))
        threshold = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits >= threshold, logits, -np.inf)

    # ── Softmax → sample ──────────────────────────────────────────
    logits -= logits.max()           # numerical stability
    probs = np.exp(logits)
    probs /= probs.sum()

    return int(rng.choice(len(probs), p=probs))


def top_k_probs(
    logits: np.ndarray,
    temperature: float = 1.0,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the top-k token indices and their probabilities
    (used by the Streamlit UI for the probability bar chart).

    Parameters
    ----------
    logits      : (V,)
    temperature : float
    k           : int

    Returns
    -------
    top_indices : (k,) — sorted descending by probability
    top_probs   : (k,) — corresponding probabilities
    """
    if temperature > 0:
        scaled = logits / temperature
    else:
        scaled = logits.copy()

    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()

    idx = np.argsort(probs)[::-1][:k]
    return idx, probs[idx]
