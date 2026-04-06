"""
adam.py
-------
Adam optimiser (Kingma and Ba, 2015).

Implements the parameter update rule:
    m_t = β1 * m_{t-1} + (1 - β1) * g_t
    v_t = β2 * v_{t-1} + (1 - β2) * g_t²
    m̂_t = m_t / (1 - β1^t)          (bias correction)
    v̂_t = v_t / (1 - β2^t)
    θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)

Additionally supports gradient clipping (by global norm) before the update.
"""

import numpy as np


class Adam:
    """
    Parameters
    ----------
    lr     : float — learning rate (default 1e-3)
    beta1  : float — 1st moment decay (default 0.9)
    beta2  : float — 2nd moment decay (default 0.999)
    epsilon: float — numerical stability (default 1e-8)
    clip   : float — max gradient norm; None to disable clipping
    """

    def __init__(
        self,
        lr: float     = 1e-3,
        beta1: float  = 0.9,
        beta2: float  = 0.999,
        epsilon: float = 1e-8,
        clip: float | None = 5.0,
    ):
        self.lr      = lr
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self.clip    = clip

        self.t: int = 0                        # step counter
        self.m: dict[str, np.ndarray] = {}     # 1st moments
        self.v: dict[str, np.ndarray] = {}     # 2nd moments

    # ──────────────────────────────────────────
    # Gradient clipping (by global L2 norm)
    # ──────────────────────────────────────────

    @staticmethod
    def _global_norm(grads: dict[str, np.ndarray]) -> float:
        total = sum(np.sum(g ** 2) for g in grads.values())
        return float(np.sqrt(total))

    def _clip_grads(
        self, grads: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        if self.clip is None:
            return grads
        norm = self._global_norm(grads)
        if norm > self.clip:
            scale = self.clip / (norm + 1e-12)
            return {k: g * scale for k, g in grads.items()}
        return grads

    # ──────────────────────────────────────────
    # Update step
    # ──────────────────────────────────────────

    def step(
        self,
        params: dict[str, np.ndarray],
        grads: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Apply one Adam update step in-place on `params`.

        Parameters
        ----------
        params : dict name → parameter ndarray
        grads  : dict name → gradient ndarray  (same keys as params)

        Returns
        -------
        params : updated in-place (also returned for convenience)
        """
        self.t += 1
        grads = self._clip_grads(grads)

        b1t = self.beta1 ** self.t
        b2t = self.beta2 ** self.t

        for key, g in grads.items():
            if key not in params:
                continue  # skip grad keys that have no matching param

            # Initialise moment buffers on first encounter
            if key not in self.m:
                self.m[key] = np.zeros_like(g)
                self.v[key] = np.zeros_like(g)

            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * g ** 2

            m_hat = self.m[key] / (1.0 - b1t)
            v_hat = self.v[key] / (1.0 - b2t)

            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params
