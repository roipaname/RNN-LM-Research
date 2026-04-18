"""
transformerlm.py
----------------
TransformerLM: GPT-style causal language model built from scratch in NumPy.

Architecture (forward pass):
    tokens (T,)
      └─ EmbeddingLayer                    → x   (T, D)
      └─ PositionalEncoding                → x   (T, D)   (added in-place)
      └─ N × TransformerBlock              → x   (T, D)
           each block: Pre-LN + Fused-QKV MultiHeadCausalSelfAttn + Pre-LN + GELU-FFN
      └─ Final LayerNorm                   → x   (T, D)
      └─ Weight-tied output projection     → logits (T, V)
           ( logits = x @ embedding_matrix.T )
      └─ label-smoothed cross-entropy      → loss (scalar)

Public API (mirrors the old RNNLM class):
    forward(tokens, targets, training) -> (logits, loss)
    backward(scale)                    -> grads dict
    generate(seed_tokens, vocab, ...)  -> (completion_str, attention_weights)
    zero_grad()
    params()
    save(path)
    load(path)
    from_checkpoint(path)              [staticmethod]
"""

import numpy as np
import os
import math

from .embedding           import EmbeddingLayer
from .positional_encoding import PositionalEncoding
from .transformer_block   import TransformerBlock, build_causal_mask, _layer_norm, _d_layer_norm


# ── Utilities ─────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    smoothing: float = 0.1,
) -> float:
    """
    Label-smoothed cross-entropy loss.

    Parameters
    ----------
    logits   : (T, V) — raw (pre-softmax) scores
    targets  : (T,)   — integer target indices
    smoothing: float  — label smoothing factor (0 = hard targets)

    Returns
    -------
    loss : scalar float
    """
    T, V = logits.shape
    log_p = logits - np.log(np.exp(logits - logits.max(axis=-1, keepdims=True)).sum(
        axis=-1, keepdims=True)) - logits.max(axis=-1, keepdims=True)
    # Numerically stable log-softmax
    log_p = logits - (
        np.log(np.sum(np.exp(logits - logits.max(-1, keepdims=True)), axis=-1, keepdims=True))
        + logits.max(-1, keepdims=True)
    )
    nll    = -log_p[np.arange(T), targets]          # (T,)
    smooth = -log_p.mean(axis=-1)                   # (T,)
    loss   = (1.0 - smoothing) * nll + smoothing * smooth
    return float(loss.mean())


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along last axis."""
    m = logits.max(axis=-1, keepdims=True)
    return logits - m - np.log(np.sum(np.exp(logits - m), axis=-1, keepdims=True))


# ═════════════════════════════════════════════════════════════════════════════
#  TransformerLM
# ═════════════════════════════════════════════════════════════════════════════

class TransformerLM:
    """
    Transformer Language Model.

    Parameters
    ----------
    vocab_size    : int
    embed_dim     : int   — d_model (default 256)
    num_heads     : int   — attention heads (must divide embed_dim, default 8)
    num_layers    : int   — number of transformer blocks (default 4)
    ffn_dim       : int   — FFN inner dimension (default 4 × embed_dim)
    dropout_rate  : float — dropout probability (default 0.1)
    label_smoothing: float — label smoothing for CE loss (default 0.1)
    max_seq_len   : int   — maximum context length (default 512)
    seed          : int
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int         = 256,
        num_heads: int         = 8,
        num_layers: int        = 4,
        ffn_dim: int | None    = None,
        dropout_rate: float    = 0.1,
        label_smoothing: float = 0.1,
        max_seq_len: int       = 512,
        seed: int              = 0,
    ):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.vocab_size      = vocab_size
        self.D               = embed_dim
        self.num_heads       = num_heads
        self.num_layers      = num_layers
        self.ffn_dim         = ffn_dim if ffn_dim is not None else 4 * embed_dim
        self.dropout_rate    = dropout_rate
        self.label_smoothing = label_smoothing
        self.max_seq_len     = max_seq_len

        # ── Sub-modules ────────────────────────────────────────────
        self.embedding = EmbeddingLayer(vocab_size, embed_dim, seed=seed)
        self.pos_enc   = PositionalEncoding(embed_dim, max_len=max_seq_len)

        self.blocks: list[TransformerBlock] = [
            TransformerBlock(
                d_model      = embed_dim,
                num_heads    = num_heads,
                ffn_dim      = self.ffn_dim,
                dropout_rate = dropout_rate,
                seed         = seed + i,
            )
            for i in range(num_layers)
        ]

        # ── Final LayerNorm ────────────────────────────────────────
        self.final_ln_gamma = np.ones(embed_dim,  dtype=np.float32)
        self.final_ln_beta  = np.zeros(embed_dim, dtype=np.float32)
        self.d_final_ln_gamma = np.zeros_like(self.final_ln_gamma)
        self.d_final_ln_beta  = np.zeros_like(self.final_ln_beta)

        # ── Weight-tied output: logits = x @ embedding.E.T ────────
        # (no separate W_out — saves V×D parameters and regularises the model)

        # ── Causal mask cache ──────────────────────────────────────
        self._mask_cache: dict[int, np.ndarray] = {}

        # ── Forward cache ──────────────────────────────────────────
        self._cache: dict = {}

    # ── Helpers ───────────────────────────────────────────────────

    def _get_mask(self, seq_len: int) -> np.ndarray:
        """Return (and cache) the causal mask for this sequence length."""
        if seq_len not in self._mask_cache:
            self._mask_cache[seq_len] = build_causal_mask(seq_len)
        return self._mask_cache[seq_len]

    # ──────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────

    def forward(
        self,
        tokens: np.ndarray,
        targets: np.ndarray | None = None,
        training: bool = True,
    ) -> tuple[np.ndarray, float]:
        """
        Parameters
        ----------
        tokens  : (T,)        — input token indices
        targets : (T,) | None — target token indices (for loss)
        training: bool        — controls dropout

        Returns
        -------
        logits : (T, V)  — raw pre-softmax scores at each position
        loss   : float   — label-smoothed cross-entropy (0.0 if targets is None)
        """
        T = len(tokens)
        if T > self.max_seq_len:
            raise ValueError(
                f"Input length {T} exceeds max_seq_len={self.max_seq_len}. "
                "Truncate the input or increase max_seq_len."
            )

        # ── 1. Embedding + Positional Encoding ────────────────────
        x = self.embedding.forward(tokens).astype(np.float32)  # (T, D)
        x = x + self.pos_enc.get(T)                            # (T, D)  broadcast

        # ── 2. Transformer blocks ──────────────────────────────────
        mask = self._get_mask(T)
        for block in self.blocks:
            x = block.forward(x, mask=mask, training=training)

        # ── 3. Final LayerNorm ─────────────────────────────────────
        x_normed, mean_f, rstd_f, xhat_f = _layer_norm(
            x, self.final_ln_gamma, self.final_ln_beta
        )

        # ── 4. Weight-tied output projection ──────────────────────
        logits = x_normed @ self.embedding.E.T                 # (T, V)

        # ── 5. Loss ────────────────────────────────────────────────
        loss = 0.0
        if targets is not None:
            loss = _cross_entropy(logits, targets, self.label_smoothing)

        # Cache for backward
        self._cache = dict(
            tokens=tokens, targets=targets,
            x_normed=x_normed, mean_f=mean_f, rstd_f=rstd_f, xhat_f=xhat_f,
            logits=logits, T=T,
        )
        return logits, loss

    # ──────────────────────────────────────────
    # Backward (BPTT through Transformer)
    # ──────────────────────────────────────────

    def backward(self, scale: float = 1.0) -> dict[str, np.ndarray]:
        """
        Backpropagate through the full forward graph.

        Parameters
        ----------
        scale : float — loss scaling factor (typically 1.0; set to 1/B for manual batching)

        Returns
        -------
        grads : flat dict mapping parameter name → gradient ndarray
        """
        c = self._cache
        logits, targets = c["logits"], c["targets"]
        T, V = logits.shape
        x_normed = c["x_normed"]

        # ── 5→4: d_loss / d_logits (label-smoothed softmax CE) ────
        log_p   = _log_softmax(logits)                          # (T, V)
        probs   = np.exp(log_p)                                 # (T, V)

        # Hard target gradient
        d_logits = probs.copy()
        d_logits[np.arange(T), targets] -= 1.0
        # Smooth target gradient: -1/V for all positions
        d_logits_smooth = probs - 1.0 / V
        d_logits = (
            (1.0 - self.label_smoothing) * d_logits
            + self.label_smoothing * d_logits_smooth
        )
        d_logits /= T
        d_logits *= scale                                       # (T, V)

        # ── 4: weight-tied output projection backward ──────────────
        # logits = x_normed @ E.T   shape: (T, D) @ (D, V) → (T, V)
        # d_x_normed = d_logits @ E            → (T, D)
        # d_E       += d_logits.T @ x_normed   → (V, D)
        d_x_normed = d_logits @ self.embedding.E               # (T, D)
        # Embedding gradient from output projection (weight tying)
        self.embedding.dE += d_logits.T @ x_normed             # (V, D)

        # ── 3: Final LayerNorm backward ────────────────────────────
        dx, dg_f, db_f = _d_layer_norm(
            d_x_normed, c["xhat_f"], c["rstd_f"], self.final_ln_gamma
        )
        self.d_final_ln_gamma += dg_f
        self.d_final_ln_beta  += db_f

        # ── 2: Transformer blocks backward (reverse order) ─────────
        for block in reversed(self.blocks):
            dx = block.backward(dx)

        # ── 1: Embedding backward ──────────────────────────────────
        # (dx here is d_loss / d_embedding_output, before PE which is constant)
        self.embedding.backward(dx)

        # ── Collect gradients ──────────────────────────────────────
        grads: dict[str, np.ndarray] = {}
        grads["final_ln_gamma"] = self.d_final_ln_gamma
        grads["final_ln_beta"]  = self.d_final_ln_beta
        grads.update({f"emb_{k}": v for k, v in self.embedding.grads().items()})
        for i, block in enumerate(self.blocks):
            grads.update({f"block{i}_{k}": v for k, v in block.grads().items()})
        return grads

    def zero_grad(self) -> None:
        self.d_final_ln_gamma[:] = 0.0
        self.d_final_ln_beta[:]  = 0.0
        self.embedding.zero_grad()
        for block in self.blocks:
            block.zero_grad()

    # ──────────────────────────────────────────
    # Inference / generation
    # ──────────────────────────────────────────

    def generate(
        self,
        seed_tokens: np.ndarray,
        vocab,
        n: int = 200,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> tuple[str, np.ndarray]:
        """
        Auto-regressively generate `n` tokens.

        Parameters
        ----------
        seed_tokens : (T,) integer indices
        vocab       : Vocabulary instance with a .decode() method
        n           : number of new tokens to generate
        temperature : sampling temperature (0 → greedy)
        top_k       : top-k truncation (0 = no truncation)

        Returns
        -------
        completion  : str
        attn_placeholder : np.ndarray of zeros (shape compatibility with old API)
        """
        from .sampling import sample_token

        tokens = list(seed_tokens)

        for _ in range(n):
            # Always cap context to max_seq_len to prevent mask shape error
            ctx = tokens[-self.max_seq_len:]
            inp = np.array(ctx, dtype=np.int32)
            logits, _ = self.forward(inp, targets=None, training=False)
            # Sample from last-position logits
            next_tok = sample_token(logits[-1], temperature, top_k)
            tokens.append(next_tok)

        generated = tokens[len(seed_tokens):]
        # Return a zero attention array for API compatibility with old RNNLM
        alpha_placeholder = np.zeros(len(seed_tokens))
        return vocab.decode(generated), alpha_placeholder

    # ──────────────────────────────────────────
    # Param / grad / state helpers
    # ──────────────────────────────────────────

    def params(self) -> dict[str, np.ndarray]:
        p: dict[str, np.ndarray] = {
            "final_ln_gamma": self.final_ln_gamma,
            "final_ln_beta":  self.final_ln_beta,
        }
        p.update({f"emb_{k}": v for k, v in self.embedding.params().items()})
        for i, block in enumerate(self.blocks):
            p.update({f"block{i}_{k}": v for k, v in block.params().items()})
        return p

    def save(self, path: str) -> None:
        """
        Save model weights + hyperparams to a .npz checkpoint.

        The checkpoint stores all weight arrays plus metadata scalars
        (_vocab_size, _embed_dim, _num_heads, _num_layers, _ffn_dim,
         _max_seq_len) so that from_checkpoint() can reconstruct the
        exact model without any hardcoded config.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {k: v.copy() for k, v in self.params().items()}
        # Hyperparameter metadata
        payload["_vocab_size"]   = np.array(self.vocab_size)
        payload["_embed_dim"]    = np.array(self.D)
        payload["_num_heads"]    = np.array(self.num_heads)
        payload["_num_layers"]   = np.array(self.num_layers)
        payload["_ffn_dim"]      = np.array(self.ffn_dim)
        payload["_max_seq_len"]  = np.array(self.max_seq_len)
        fpath = path if path.endswith(".npz") else path + ".npz"
        np.savez_compressed(fpath, **payload)
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"[TransformerLM] Saved → {fpath}  ({size_mb:.1f} MB, "
              f"{len(payload)} arrays)")

    def load(self, path: str) -> None:
        """
        Load weights from an .npz checkpoint into this model instance.

        Validates that the checkpoint hyperparams match this model's config.
        Raises ValueError if there is a mismatch.
        """
        fpath = path if path.endswith(".npz") else path + ".npz"
        data  = np.load(fpath)

        # ── Hyperparam validation ──────────────────────────────────
        if "_vocab_size" in data:
            ckpt = {
                "vocab_size" : int(data["_vocab_size"]),
                "embed_dim"  : int(data["_embed_dim"]),
                "num_heads"  : int(data["_num_heads"]),
                "num_layers" : int(data["_num_layers"]),
                "ffn_dim"    : int(data["_ffn_dim"]),
            }
            mine = {
                "vocab_size" : self.vocab_size,
                "embed_dim"  : self.D,
                "num_heads"  : self.num_heads,
                "num_layers" : self.num_layers,
                "ffn_dim"    : self.ffn_dim,
            }
            mismatches = {k: (ckpt[k], mine[k]) for k in ckpt if ckpt[k] != mine[k]}
            if mismatches:
                details = ", ".join(f"{k}: ckpt={c} vs model={m}"
                                    for k, (c, m) in mismatches.items())
                raise ValueError(
                    f"[TransformerLM] Checkpoint hyperparams do not match this model. "
                    f"Mismatches: {details}. "
                    "Use TransformerLM.from_checkpoint(path) to reconstruct automatically."
                )

        # ── Route weights to sub-modules ──────────────────────────
        emb_dict = {k[4:]: v for k, v in data.items() if k.startswith("emb_")}
        self.embedding.load_state_dict(emb_dict)

        for i, block in enumerate(self.blocks):
            pfx = f"block{i}_"
            block_dict = {k[len(pfx):]: v for k, v in data.items() if k.startswith(pfx)}
            block.load_state_dict(block_dict)

        self.final_ln_gamma = data["final_ln_gamma"].copy()
        self.final_ln_beta  = data["final_ln_beta"].copy()

        print(f"[TransformerLM] Loaded ← {fpath}  ({len(data)} arrays)")

    @staticmethod
    def from_checkpoint(path: str) -> "TransformerLM":
        """
        Reconstruct a TransformerLM with the exact hyperparams stored in
        a checkpoint, then load the weights.

        Usage
        -----
        model = TransformerLM.from_checkpoint("checkpoints/best_transformer")
        """
        fpath = path if path.endswith(".npz") else path + ".npz"
        data  = np.load(fpath)
        required = ("_vocab_size", "_embed_dim", "_num_heads",
                    "_num_layers", "_ffn_dim")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(
                f"[TransformerLM] Checkpoint is missing metadata keys: {missing}. "
                "Cannot reconstruct model automatically. "
                "Provide hyperparams manually and use model.load(path) instead."
            )
        model = TransformerLM(
            vocab_size   = int(data["_vocab_size"]),
            embed_dim    = int(data["_embed_dim"]),
            num_heads    = int(data["_num_heads"]),
            num_layers   = int(data["_num_layers"]),
            ffn_dim      = int(data["_ffn_dim"]),
            max_seq_len  = int(data.get("_max_seq_len", 512)),
            dropout_rate = 0.0,   # inference — no dropout
        )
        model.load(fpath)
        return model
