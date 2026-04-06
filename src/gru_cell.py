"""
gru_cell.py
-----------
A single GRU cell implemented entirely in NumPy.

Equations (Cho et al., 2014):
    z_t = σ(W_z · [h_{t-1}, x_t] + b_z)          update gate
    r_t = σ(W_r · [h_{t-1}, x_t] + b_r)          reset gate
    ĥ_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h) candidate hidden state
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ĥ_t        output hidden state

Forward:  (x_t, h_{t-1}) -> h_t
Backward: dh_t -> (dx_t, dh_{t-1}, weight gradients)
"""

import numpy as np


# ──────────────────────────────────────────────
# Activation helpers
# ──────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def d_sigmoid(s: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given its OUTPUT value s."""
    return s * (1.0 - s)


def d_tanh(t: np.ndarray) -> np.ndarray:
    """Derivative of tanh given its OUTPUT value t."""
    return 1.0 - t ** 2


class GRUCell:
    """
    Single GRU cell.

    Parameters
    ----------
    input_dim : int   — dimensionality of x_t
    hidden_dim : int  — dimensionality of h_t  (H in the design doc)
    seed : int
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 0):
        self.input_dim = input_dim
        self.H = hidden_dim

        rng = np.random.default_rng(seed)
        D, H = input_dim, hidden_dim

        # Xavier / Glorot uniform initialisation
        def xavier(rows, cols):
            lim = np.sqrt(6.0 / (rows + cols))
            return rng.uniform(-lim, lim, (rows, cols))

        # Weight matrices: concatenated form [h_{t-1}, x_t] has dim H+D
        self.W_z = xavier(H, H + D)   # update gate
        self.W_r = xavier(H, H + D)   # reset gate
        self.W_h = xavier(H, H + D)   # candidate state

        # Biases
        self.b_z = np.zeros(H)
        self.b_r = np.zeros(H)
        self.b_h = np.zeros(H)

        # Gradient buffers
        self.dW_z = np.zeros_like(self.W_z)
        self.dW_r = np.zeros_like(self.W_r)
        self.dW_h = np.zeros_like(self.W_h)
        self.db_z = np.zeros_like(self.b_z)
        self.db_r = np.zeros_like(self.b_r)
        self.db_h = np.zeros_like(self.b_h)

        # Cache for backward pass
        self.cache: dict = {}

    # ──────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x      : (D,)  — input at current timestep
        h_prev : (H,)  — hidden state from previous timestep

        Returns
        -------
        h_t : (H,)
        """
        xh = np.concatenate([h_prev, x])   # (H+D,)

        z = sigmoid(self.W_z @ xh + self.b_z)            # (H,)
        r = sigmoid(self.W_r @ xh + self.b_r)            # (H,)
        rh = r * h_prev                                  # (H,)
        xrh = np.concatenate([rh, x])                   # (H+D,)
        h_cand = np.tanh(self.W_h @ xrh + self.b_h)    # (H,)
        h_t = (1.0 - z) * h_prev + z * h_cand          # (H,)

        # Cache everything needed for backward
        self.cache = dict(
            x=x, h_prev=h_prev, xh=xh,
            z=z, r=r, rh=rh, xrh=xrh,
            h_cand=h_cand, h_t=h_t
        )
        return h_t

    # ──────────────────────────────────────────
    # Backward
    # ──────────────────────────────────────────

    def backward(
        self, dh_t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Backpropagate through one GRU cell.

        Parameters
        ----------
        dh_t : (H,) — upstream gradient w.r.t. h_t

        Returns
        -------
        dx     : (D,) — gradient w.r.t. x_t
        dh_prev: (H,) — gradient w.r.t. h_{t-1}
        """
        c = self.cache
        x, h_prev = c["x"], c["h_prev"]
        z, r, rh  = c["z"], c["r"], c["rh"]
        xrh, h_cand = c["xrh"], c["h_cand"]

        H = self.H

        # ── h_t = (1-z)*h_prev + z*h_cand ──
        dh_cand = dh_t * z                                    # (H,)
        dz      = dh_t * (h_cand - h_prev)                   # (H,)
        dh_prev = dh_t * (1.0 - z)                           # (H,) ← partial

        # ── h_cand = tanh(W_h · xrh + b_h) ──
        d_h_cand_pre = dh_cand * d_tanh(h_cand)              # (H,)
        self.dW_h   += np.outer(d_h_cand_pre, xrh)
        self.db_h   += d_h_cand_pre
        d_xrh        = self.W_h.T @ d_h_cand_pre             # (H+D,)
        d_rh         = d_xrh[:H]
        dx_from_h    = d_xrh[H:]                             # (D,) partial

        # ── rh = r * h_prev ──
        dr      = d_rh * h_prev                              # (H,)
        dh_prev += d_rh * r                                  # accumulate

        # ── z gate ──
        dz_pre = dz * d_sigmoid(z)                           # (H,)
        self.dW_z += np.outer(dz_pre, c["xh"])
        self.db_z += dz_pre
        d_xh_from_z = self.W_z.T @ dz_pre                   # (H+D,)

        # ── r gate ──
        dr_pre = dr * d_sigmoid(r)                           # (H,)
        self.dW_r += np.outer(dr_pre, c["xh"])
        self.db_r += dr_pre
        d_xh_from_r = self.W_r.T @ dr_pre                   # (H+D,)

        # ── combine gradients through the [h_prev, x] concatenation ──
        d_xh = d_xh_from_z + d_xh_from_r                   # (H+D,)
        dh_prev += d_xh[:H]
        dx = d_xh[H:] + dx_from_h                           # (D,)

        return dx, dh_prev

    # ──────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────

    def params(self) -> dict[str, np.ndarray]:
        return dict(
            W_z=self.W_z, b_z=self.b_z,
            W_r=self.W_r, b_r=self.b_r,
            W_h=self.W_h, b_h=self.b_h,
        )

    def grads(self) -> dict[str, np.ndarray]:
        return dict(
            W_z=self.dW_z, b_z=self.db_z,
            W_r=self.dW_r, b_r=self.db_r,
            W_h=self.dW_h, b_h=self.db_h,
        )

    def zero_grad(self) -> None:
        for arr in (self.dW_z, self.dW_r, self.dW_h,
                    self.db_z, self.db_r, self.db_h):
            arr[:] = 0.0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params().items()}

    def load_state_dict(self, d: dict[str, np.ndarray]) -> None:
        self.W_z = d["W_z"].copy(); self.b_z = d["b_z"].copy()
        self.W_r = d["W_r"].copy(); self.b_r = d["b_r"].copy()
        self.W_h = d["W_h"].copy(); self.b_h = d["b_h"].copy()
