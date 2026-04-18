"""
transformer_block.py
--------------------
A single GPT-style Transformer decoder block implemented in pure NumPy.

Each block contains two sub-layers with Pre-LayerNorm residual connections:

    x → LayerNorm → MultiHeadCausalSelfAttention → + (residual) → x'
    x' → LayerNorm → PositionWiseFFN             → + (residual) → out

Components
----------
  MultiHeadCausalSelfAttention
    • Fused QKV projection  (3·D, D)  — one matmul instead of three
    • Causal (lower-triangular) mask — prevents attending to future tokens
    • Scaled dot-product attention:  softmax(QK^T / √d_k) · V
    • Output projection  (D, D)

  PositionWiseFFN
    • W1 (ffn_dim, D)  + GELU activation
    • W2 (D, ffn_dim)

  LayerNorm  — applied before each sub-layer (Pre-LN, GPT-2 style)

Forward:  x (T, D) → out (T, D)
Backward: d_out (T, D) → dx (T, D)  +  accumulates weight gradients
"""

import numpy as np
import math


# ── Activation helpers ────────────────────────────────────────────────────────

def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU approximation (Hendrycks & Gimpel, 2016)."""
    return 0.5 * x * (1.0 + np.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    ))


def _d_gelu(x: np.ndarray) -> np.ndarray:
    """Derivative of the GELU approximation w.r.t. its input x."""
    c = math.sqrt(2.0 / math.pi)
    tanh_arg = c * (x + 0.044715 * x ** 3)
    tanh_val = np.tanh(tanh_arg)
    d_tanh_arg = c * (1.0 + 3 * 0.044715 * x ** 2)
    sech2 = 1.0 - tanh_val ** 2
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_tanh_arg


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`."""
    x_s = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x_s)
    return e / e.sum(axis=axis, keepdims=True)


def _layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Layer normalisation over the last axis.

    Returns
    -------
    y    : (T, D) — normalised output
    mean : (T, 1)
    rstd : (T, 1) — 1 / sqrt(var + eps)
    """
    mean = x.mean(axis=-1, keepdims=True)          # (T, 1)
    var  = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)               # (T, 1)
    x_hat = (x - mean) * rstd                     # (T, D)
    return gamma * x_hat + beta, mean, rstd, x_hat


def _d_layer_norm(
    d_y: np.ndarray,
    x_hat: np.ndarray,
    rstd: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass through LayerNorm.

    Returns
    -------
    dx     : (T, D)
    d_gamma: (D,)
    d_beta : (D,)
    """
    T, D = d_y.shape
    d_gamma = (d_y * x_hat).sum(axis=0)           # (D,)
    d_beta  = d_y.sum(axis=0)                     # (D,)
    d_x_hat = d_y * gamma                         # (T, D)
    # Full LN backward (Ioffe & Szegedy)
    dx = rstd * (
        d_x_hat
        - d_x_hat.mean(axis=-1, keepdims=True)
        - x_hat * (d_x_hat * x_hat).mean(axis=-1, keepdims=True)
    )
    return dx, d_gamma, d_beta


# ── Causal mask (static; built once per seq_len) ──────────────────────────────

def build_causal_mask(seq_len: int) -> np.ndarray:
    """
    Return an additive causal mask of shape (seq_len, seq_len).
    Upper-triangle entries are -1e9 (masked); lower-triangle / diagonal are 0.
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    return np.where(mask == 0, np.float32(-1e9), np.float32(0.0))


# ── Xavier initialisation ─────────────────────────────────────────────────────

def _xavier(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    lim = math.sqrt(6.0 / (rows + cols))
    return rng.uniform(-lim, lim, (rows, cols)).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  TransformerBlock
# ═════════════════════════════════════════════════════════════════════════════

class TransformerBlock:
    """
    One GPT-style decoder block (Pre-LN, causal self-attention + FFN).

    Parameters
    ----------
    d_model     : int   — embedding / model dimension
    num_heads   : int   — number of attention heads (must divide d_model)
    ffn_dim     : int   — inner dimension of the FFN (typically 4 × d_model)
    dropout_rate: float — applied to attention weights and FFN hidden layer
    seed        : int
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout_rate: float = 0.1,
        seed: int = 0,
    ):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.D = d_model
        self.H = num_heads
        self.d_k = d_model // num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

        rng = np.random.default_rng(seed)

        # ── Attention weights ─────────────────────────────────────
        # Fused QKV: one (3D, D) matrix instead of three separate ones.
        self.W_qkv   = _xavier(rng, 3 * d_model, d_model)
        self.b_qkv   = np.zeros(3 * d_model, dtype=np.float32)
        self.W_o     = _xavier(rng, d_model, d_model)
        self.b_o     = np.zeros(d_model, dtype=np.float32)

        # ── FFN weights ───────────────────────────────────────────
        self.W1 = _xavier(rng, ffn_dim, d_model)
        self.b1 = np.zeros(ffn_dim, dtype=np.float32)
        self.W2 = _xavier(rng, d_model, ffn_dim)
        self.b2 = np.zeros(d_model, dtype=np.float32)

        # ── LayerNorm params (Pre-LN before each sub-layer) ───────
        self.ln1_gamma = np.ones(d_model,  dtype=np.float32)
        self.ln1_beta  = np.zeros(d_model, dtype=np.float32)
        self.ln2_gamma = np.ones(d_model,  dtype=np.float32)
        self.ln2_beta  = np.zeros(d_model, dtype=np.float32)

        # ── Gradient buffers (zero-initialised) ───────────────────
        self.dW_qkv   = np.zeros_like(self.W_qkv)
        self.db_qkv   = np.zeros_like(self.b_qkv)
        self.dW_o     = np.zeros_like(self.W_o)
        self.db_o     = np.zeros_like(self.b_o)
        self.dW1      = np.zeros_like(self.W1)
        self.db1      = np.zeros_like(self.b1)
        self.dW2      = np.zeros_like(self.W2)
        self.db2      = np.zeros_like(self.b2)
        self.dln1_gamma = np.zeros_like(self.ln1_gamma)
        self.dln1_beta  = np.zeros_like(self.ln1_beta)
        self.dln2_gamma = np.zeros_like(self.ln2_gamma)
        self.dln2_beta  = np.zeros_like(self.ln2_beta)

        # ── Forward cache ─────────────────────────────────────────
        self._cache: dict = {}

        # ── Dropout RNG ───────────────────────────────────────────
        self._rng = np.random.default_rng(seed + 1000)

    # ──────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────

    def forward(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        training: bool = True,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        x        : (T, D) — input token representations
        mask     : (T, T) — additive causal mask (−1e9 for masked positions)
        training : bool   — enables dropout

        Returns
        -------
        out : (T, D)
        """
        T, D = x.shape

        # ── Attention sub-layer ─────────────────────────────────────
        # Pre-LayerNorm
        xn1, mean1, rstd1, xhat1 = _layer_norm(x, self.ln1_gamma, self.ln1_beta)

        # Fused QKV projection
        qkv   = xn1 @ self.W_qkv.T + self.b_qkv       # (T, 3D)
        Q, K, V = np.split(qkv, 3, axis=-1)            # each (T, D)

        # Reshape to (H, T, d_k)
        Q = Q.reshape(T, self.H, self.d_k).transpose(1, 0, 2)  # (H, T, d_k)
        K = K.reshape(T, self.H, self.d_k).transpose(1, 0, 2)
        V = V.reshape(T, self.H, self.d_k).transpose(1, 0, 2)

        # Scaled dot-product attention
        scale  = math.sqrt(self.d_k)
        scores = Q @ K.transpose(0, 2, 1) / scale + mask       # (H, T, T)
        attn   = _softmax(scores, axis=-1)                      # (H, T, T)

        # Attention dropout
        attn_drop = attn
        attn_mask_drop = None
        if training and self.dropout_rate > 0.0:
            attn_mask_drop = (
                self._rng.random(attn.shape) >= self.dropout_rate
            ).astype(np.float32)
            attn_drop = attn * attn_mask_drop / (1.0 - self.dropout_rate)

        ctx = attn_drop @ V                                     # (H, T, d_k)
        ctx = ctx.transpose(1, 0, 2).reshape(T, D)             # (T, D)
        attn_out = ctx @ self.W_o.T + self.b_o                 # (T, D)

        x_attn = x + attn_out      # residual

        # ── FFN sub-layer ─────────────────────────────────────────
        # Pre-LayerNorm
        xn2, mean2, rstd2, xhat2 = _layer_norm(x_attn, self.ln2_gamma, self.ln2_beta)

        h    = xn2 @ self.W1.T + self.b1                       # (T, ffn_dim)
        h_act = _gelu(h)                                        # (T, ffn_dim)

        # FFN dropout
        ffn_mask = None
        if training and self.dropout_rate > 0.0:
            ffn_mask = (
                self._rng.random(h_act.shape) >= self.dropout_rate
            ).astype(np.float32)
            h_act_drop = h_act * ffn_mask / (1.0 - self.dropout_rate)
        else:
            h_act_drop = h_act

        ffn_out = h_act_drop @ self.W2.T + self.b2             # (T, D)

        out = x_attn + ffn_out     # residual

        # Cache for backward
        self._cache = dict(
            x=x, xn1=xn1, mean1=mean1, rstd1=rstd1, xhat1=xhat1,
            Q=Q, K=K, V=V,
            scores=scores, attn=attn, attn_drop=attn_drop,
            attn_mask_drop=attn_mask_drop,
            ctx=ctx, attn_out=attn_out, x_attn=x_attn,
            xn2=xn2, mean2=mean2, rstd2=rstd2, xhat2=xhat2,
            h=h, h_act=h_act, ffn_mask=ffn_mask, h_act_drop=h_act_drop,
            ffn_out=ffn_out, T=T, D=D,
        )
        return out

    # ──────────────────────────────────────────
    # Backward
    # ──────────────────────────────────────────

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        d_out : (T, D) — upstream gradient w.r.t. block output

        Returns
        -------
        dx : (T, D) — gradient w.r.t. block input x
        """
        c = self._cache
        T, D = c["T"], c["D"]

        # ── FFN residual ─────────────────────────────────────────
        d_x_attn = d_out.copy()   # gradient through residual
        d_ffn_out = d_out

        # FFN output projection backward
        self.dW2  += d_ffn_out.T @ c["h_act_drop"]             # (D, ffn_dim)
        self.db2  += d_ffn_out.sum(axis=0)
        d_h_act_drop = d_ffn_out @ self.W2                      # (T, ffn_dim)

        # FFN dropout backward
        if c["ffn_mask"] is not None:
            d_h_act = d_h_act_drop * c["ffn_mask"] / (1.0 - self.dropout_rate)
        else:
            d_h_act = d_h_act_drop

        # GELU backward
        d_h = d_h_act * _d_gelu(c["h"])                        # (T, ffn_dim)

        # FFN W1 backward
        self.dW1 += d_h.T @ c["xn2"]                           # (ffn_dim, D)
        self.db1 += d_h.sum(axis=0)
        d_xn2 = d_h @ self.W1                                  # (T, D)

        # Pre-LN2 backward
        dx_ln2, dg2, dbeta2 = _d_layer_norm(d_xn2, c["xhat2"], c["rstd2"], self.ln2_gamma)
        self.dln2_gamma += dg2
        self.dln2_beta  += dbeta2
        d_x_attn += dx_ln2

        # ── Attention residual ───────────────────────────────────
        dx = d_x_attn.copy()
        d_attn_out = d_x_attn

        # Output projection backward
        self.dW_o += d_attn_out.T @ c["ctx"]                   # (D, D)
        self.db_o += d_attn_out.sum(axis=0)
        d_ctx = d_attn_out @ self.W_o                          # (T, D)

        # Reshape d_ctx back to (H, T, d_k)
        d_ctx_h = d_ctx.reshape(T, self.H, self.d_k).transpose(1, 0, 2)

        # Attention + dropout backward
        d_attn_drop = d_ctx_h @ c["V"].transpose(0, 2, 1)     # (H, T, T)
        d_V = c["attn_drop"].transpose(0, 2, 1) @ d_ctx_h     # (H, T, d_k)

        if c["attn_mask_drop"] is not None:
            d_attn = d_attn_drop * c["attn_mask_drop"] / (1.0 - self.dropout_rate)
        else:
            d_attn = d_attn_drop

        # Softmax backward: d_scores = attn * (d_attn - sum(d_attn * attn, axis=-1, keepdims=True))
        attn = c["attn"]
        d_scores = attn * (d_attn - (d_attn * attn).sum(axis=-1, keepdims=True))  # (H, T, T)

        scale = math.sqrt(self.d_k)
        d_scores /= scale

        # Q, K backward from scores
        d_Q = d_scores @ c["K"]                                # (H, T, d_k)
        d_K = d_scores.transpose(0, 2, 1) @ c["Q"]            # (H, T, d_k)

        # Reshape all head gradients back to (T, D)
        d_Q = d_Q.transpose(1, 0, 2).reshape(T, D)
        d_K = d_K.transpose(1, 0, 2).reshape(T, D)
        d_V = d_V.transpose(1, 0, 2).reshape(T, D)

        d_qkv = np.concatenate([d_Q, d_K, d_V], axis=-1)      # (T, 3D)

        # Fused QKV weight backward
        self.dW_qkv += d_qkv.T @ c["xn1"]                     # (3D, D)
        self.db_qkv += d_qkv.sum(axis=0)
        d_xn1 = d_qkv @ self.W_qkv                             # (T, D)

        # Pre-LN1 backward
        dx_ln1, dg1, dbeta1 = _d_layer_norm(d_xn1, c["xhat1"], c["rstd1"], self.ln1_gamma)
        self.dln1_gamma += dg1
        self.dln1_beta  += dbeta1
        dx += dx_ln1

        return dx   # (T, D)

    # ──────────────────────────────────────────
    # Parameter / gradient access
    # ──────────────────────────────────────────

    def params(self) -> dict[str, np.ndarray]:
        return {
            "W_qkv": self.W_qkv, "b_qkv": self.b_qkv,
            "W_o":   self.W_o,   "b_o":   self.b_o,
            "W1":    self.W1,    "b1":    self.b1,
            "W2":    self.W2,    "b2":    self.b2,
            "ln1_gamma": self.ln1_gamma, "ln1_beta": self.ln1_beta,
            "ln2_gamma": self.ln2_gamma, "ln2_beta": self.ln2_beta,
        }

    def grads(self) -> dict[str, np.ndarray]:
        return {
            "W_qkv": self.dW_qkv, "b_qkv": self.db_qkv,
            "W_o":   self.dW_o,   "b_o":   self.db_o,
            "W1":    self.dW1,    "b1":    self.db1,
            "W2":    self.dW2,    "b2":    self.db2,
            "ln1_gamma": self.dln1_gamma, "ln1_beta": self.dln1_beta,
            "ln2_gamma": self.dln2_gamma, "ln2_beta": self.dln2_beta,
        }

    def zero_grad(self) -> None:
        for arr in (
            self.dW_qkv, self.db_qkv, self.dW_o, self.db_o,
            self.dW1, self.db1, self.dW2, self.db2,
            self.dln1_gamma, self.dln1_beta,
            self.dln2_gamma, self.dln2_beta,
        ):
            arr[:] = 0.0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params().items()}

    def load_state_dict(self, d: dict[str, np.ndarray]) -> None:
        self.W_qkv = d["W_qkv"].copy(); self.b_qkv = d["b_qkv"].copy()
        self.W_o   = d["W_o"].copy();   self.b_o   = d["b_o"].copy()
        self.W1    = d["W1"].copy();    self.b1    = d["b1"].copy()
        self.W2    = d["W2"].copy();    self.b2    = d["b2"].copy()
        self.ln1_gamma = d["ln1_gamma"].copy(); self.ln1_beta = d["ln1_beta"].copy()
        self.ln2_gamma = d["ln2_gamma"].copy(); self.ln2_beta = d["ln2_beta"].copy()
        # Reset grad buffers after loading
        self.zero_grad()
