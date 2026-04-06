"""
attention.py
------------
Bahdanau (additive) attention (Bahdanau et al., 2015).

Given the full sequence of GRU hidden states H = [h_1, ..., h_T]:

    e_{t,s} = v_a · tanh(W_a · h_s + U_a · h_T + b_a)   alignment score
    α_{t,s} = softmax(e_{t,s})                            attention weights
    c_t     = Σ_s α_{t,s} · h_s                           context vector
    h_attn  = tanh(W_c · [h_T, c_t] + b_c)               attended state

Here we use a simplified single-query variant: the query is always h_T
(the final hidden state), and we attend over all past hidden states.
This is the canonical setup for language model generation where we want
to summarise the sequence before projecting to logits.

Forward:  H (T, H) -> (h_attn (H,), alpha (T,))
Backward: d_h_attn (H,) -> dH (T, H)
"""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    x_shifted = x - np.max(x)
    e = np.exp(x_shifted)
    return e / e.sum()


class BahdanauAttention:
    """
    Additive attention module.

    Parameters
    ----------
    hidden_dim : int  — H
    attn_dim   : int  — internal attention projection dimension (default H // 2)
    seed       : int
    """

    def __init__(self, hidden_dim: int, attn_dim: int | None = None, seed: int = 0):
        self.H = hidden_dim
        self.A = attn_dim if attn_dim is not None else hidden_dim // 2

        rng = np.random.default_rng(seed)

        def xavier(r, c):
            lim = np.sqrt(6.0 / (r + c))
            return rng.uniform(-lim, lim, (r, c))

        # Alignment parameters
        self.W_a = xavier(self.A, self.H)   # query projection
        self.U_a = xavier(self.A, self.H)   # key projection
        self.V_a = xavier(self.A, 1)        # score projection  (A -> scalar)
        self.b_a = np.zeros(self.A)

        # Output projection: [h_T, c_t] -> h_attn
        self.W_c = xavier(self.H, 2 * self.H)
        self.b_c = np.zeros(self.H)

        # Gradient buffers
        self.dW_a = np.zeros_like(self.W_a)
        self.dU_a = np.zeros_like(self.U_a)
        self.dV_a = np.zeros_like(self.V_a)
        self.db_a = np.zeros_like(self.b_a)
        self.dW_c = np.zeros_like(self.W_c)
        self.db_c = np.zeros_like(self.b_c)

        # Cache
        self.cache: dict = {}

    # ──────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────

    def forward(self, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        H : (T, H_dim) — all GRU hidden states

        Returns
        -------
        h_attn : (H_dim,) — attended hidden state
        alpha  : (T,)     — attention weights (for visualisation)
        """
        T = H.shape[0]
        h_T = H[-1]             # query: final hidden state  (H,)

        # ── Alignment scores ──────────────────
        # keys:  U_a · h_s  for each s  → (T, A)
        keys = H @ self.U_a.T                               # (T, A)
        # query: W_a · h_T  broadcast    → (A,) -> (1, A)
        query = (self.W_a @ h_T + self.b_a).reshape(1, self.A)  # (1, A)
        # tanh combined
        e_pre = query + keys                                # (T, A)  broadcast
        e_tan = np.tanh(e_pre)                              # (T, A)
        # score: V_a · tanh(...)
        scores = e_tan @ self.V_a                           # (T, 1)
        scores = scores.squeeze(-1)                         # (T,)

        # ── Attention weights ─────────────────
        alpha = softmax(scores)                             # (T,)

        # ── Context vector ────────────────────
        c_t = alpha @ H                                     # (H,)  weighted sum

        # ── Output projection ─────────────────
        concat = np.concatenate([h_T, c_t])                # (2H,)
        h_attn = np.tanh(self.W_c @ concat + self.b_c)    # (H,)

        # Cache for backward
        self.cache = dict(
            H=H, h_T=h_T, keys=keys, query=query,
            e_pre=e_pre, e_tan=e_tan, scores=scores,
            alpha=alpha, c_t=c_t, concat=concat, h_attn=h_attn,
        )
        return h_attn, alpha

    # ──────────────────────────────────────────
    # Backward
    # ──────────────────────────────────────────

    def backward(self, d_h_attn: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        d_h_attn : (H,) — upstream gradient w.r.t. h_attn

        Returns
        -------
        dH : (T, H) — gradient w.r.t. all GRU hidden states
        """
        c = self.cache
        H, h_T = c["H"], c["h_T"]
        e_tan, alpha, c_t = c["e_tan"], c["alpha"], c["c_t"]
        concat = c["concat"]
        T = H.shape[0]

        # ── Output projection backward ────────
        d_tanh_wc = d_h_attn * (1.0 - c["h_attn"] ** 2)       # (H,)
        self.dW_c += np.outer(d_tanh_wc, concat)
        self.db_c += d_tanh_wc
        d_concat = self.W_c.T @ d_tanh_wc                      # (2H,)
        d_h_T_from_concat = d_concat[: self.H]                 # (H,) partial
        d_c_t = d_concat[self.H :]                             # (H,)

        # ── Context vector backward ───────────
        # c_t = alpha @ H  =>  dH += outer(alpha, d_c_t), d_alpha = H @ d_c_t
        dH = np.outer(alpha, d_c_t)                            # (T, H)
        d_alpha = H @ d_c_t                                    # (T,)

        # ── Softmax backward ──────────────────
        # d_scores = alpha * (d_alpha - alpha·d_alpha)
        dot = alpha @ d_alpha
        d_scores = alpha * (d_alpha - dot)                     # (T,)

        # ── Score → V_a backward ─────────────
        # scores = e_tan @ V_a  →  (T,)
        # d_e_tan = d_scores ⊗ V_a.T   (T, A) via outer
        d_e_tan = np.outer(d_scores, self.V_a.squeeze())       # (T, A)
        self.dV_a += (e_tan.T @ d_scores.reshape(-1, 1))      # (A, 1)

        # ── tanh backward ─────────────────────
        d_e_pre = d_e_tan * (1.0 - e_tan ** 2)                # (T, A)

        # ── Keys (U_a) backward ───────────────
        # keys = H @ U_a.T
        self.dU_a += d_e_pre.T @ H                             # (A, H)
        dH += d_e_pre @ self.U_a                               # (T, H)

        # ── Query (W_a) backward ─────────────
        # query = W_a @ h_T + b_a, broadcast over T
        d_query = d_e_pre.sum(axis=0)                          # (A,)
        self.dW_a += np.outer(d_query, h_T)
        self.db_a += d_query
        d_h_T_from_query = self.W_a.T @ d_query               # (H,)

        # ── Accumulate gradient into h_T (last row of H) ──
        d_h_T = d_h_T_from_concat + d_h_T_from_query
        dH[-1] += d_h_T

        return dH   # (T, H)

    # ──────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────

    def params(self) -> dict[str, np.ndarray]:
        return dict(
            W_a=self.W_a, U_a=self.U_a, V_a=self.V_a, b_a=self.b_a,
            W_c=self.W_c, b_c=self.b_c,
        )

    def grads(self) -> dict[str, np.ndarray]:
        return dict(
            W_a=self.dW_a, U_a=self.dU_a, V_a=self.dV_a, b_a=self.db_a,
            W_c=self.dW_c, b_c=self.db_c,
        )

    def zero_grad(self) -> None:
        for arr in (self.dW_a, self.dU_a, self.dV_a,
                    self.db_a, self.dW_c, self.db_c):
            arr[:] = 0.0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params().items()}

    def load_state_dict(self, d: dict[str, np.ndarray]) -> None:
        self.W_a = d["W_a"].copy(); self.U_a = d["U_a"].copy()
        self.V_a = d["V_a"].copy(); self.b_a = d["b_a"].copy()
        self.W_c = d["W_c"].copy(); self.b_c = d["b_c"].copy()
