"""
rnnlm.py
--------
RNNLM: the central language model class.

Architecture (forward pass):
    tokens (T,)
      └─ EmbeddingLayer          → embeddings (T, d)
      └─ GRULayer (2-layer GRU)  → H (T, H)
      └─ BahdanauAttention       → h_attn (H,), alpha (T,)
      └─ Linear projection       → logits (V,)
      └─ softmax / cross-entropy → loss (scalar)

The model exposes:
  forward(tokens, targets) -> (logits, loss)
  backward(dL)             -> grads dict
  generate(seed, n, T, k)  -> completion string
  save / load              → .npz checkpoints
"""

import numpy as np
import os

from .embedding import EmbeddingLayer
from .gru_layer  import GRULayer
from .attention  import BahdanauAttention


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    """Average cross-entropy loss over a sequence."""
    T = len(targets)
    log_probs = np.log(probs[np.arange(T), targets] + 1e-12)
    return -log_probs.mean()


class RNNLM:
    """
    Recurrent Neural Network Language Model.

    Parameters
    ----------
    vocab_size  : int
    embed_dim   : int  — d in design doc  (default 64)
    hidden_dim  : int  — H in design doc  (default 256)
    num_layers  : int  — GRU stacking depth (default 2)
    keep_prob   : float — dropout keep prob (default 0.8)
    attn_dim    : int  — attention projection dim (default H // 2)
    seed        : int
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int   = 64,
        hidden_dim: int  = 256,
        num_layers: int  = 2,
        keep_prob: float = 0.8,
        attn_dim: int | None = None,
        seed: int = 0,
    ):
        self.vocab_size = vocab_size
        self.H = hidden_dim
        rng = np.random.default_rng(seed)

        # ── Sub-modules ────────────────────────────────────────────
        self.embedding = EmbeddingLayer(vocab_size, embed_dim, seed=seed)
        self.gru       = GRULayer(embed_dim, hidden_dim,
                                  num_layers=num_layers,
                                  keep_prob=keep_prob, seed=seed)
        self.attention = BahdanauAttention(hidden_dim,
                                           attn_dim=attn_dim, seed=seed)

        # ── Output projection: h_attn (H,) → logits (V,) ──────────
        lim = np.sqrt(6.0 / (hidden_dim + vocab_size))
        self.W_out = rng.uniform(-lim, lim, (vocab_size, hidden_dim))
        self.b_out = np.zeros(vocab_size)
        self.dW_out = np.zeros_like(self.W_out)
        self.db_out = np.zeros_like(self.b_out)

        # Persistent hidden state across generation steps
        self.h_prev: np.ndarray | None = None

        # Cache for backward pass
        self._cache: dict = {}

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
        probs : (T, V)  — softmax probabilities at each step
        loss  : float   — cross-entropy (0.0 if targets is None)
        """
        T = len(tokens)

        # ── 1. Embedding ───────────────────────────────────────────
        emb = self.embedding.forward(tokens)          # (T, d)

        # ── 2. GRU ────────────────────────────────────────────────
        H_states = self.gru.forward(emb, training=training)  # (T, H)

        # ── 3. Attention ──────────────────────────────────────────
        h_attn, alpha = self.attention.forward(H_states)     # (H,), (T,)

        # ── 4. Output projection ──────────────────────────────────
        # We project h_attn once and use it for ALL T positions.
        # (This is the standard LM head in character-level models.)
        logits_single = self.W_out @ h_attn + self.b_out     # (V,)
        # Expand to (T, V) so callers always receive per-step probabilities
        logits = np.tile(logits_single, (T, 1))              # (T, V)
        probs  = np.apply_along_axis(softmax, 1, logits)     # (T, V)

        # ── 5. Loss ───────────────────────────────────────────────
        loss = 0.0
        if targets is not None:
            loss = cross_entropy(probs, targets)

        # Cache everything for backward
        self._cache = dict(
            tokens=tokens, targets=targets,
            emb=emb, H_states=H_states,
            h_attn=h_attn, alpha=alpha,
            logits_single=logits_single, probs=probs,
            T=T,
        )
        return probs, loss

    # ──────────────────────────────────────────
    # Backward (BPTT)
    # ──────────────────────────────────────────

    def backward(self, scale: float = 1.0) -> dict[str, np.ndarray]:
        """
        Backpropagate through the entire forward graph.

        Parameters
        ----------
        scale : float — loss scaling factor (1 / batch_size typical)

        Returns
        -------
        grads : flat dict mapping param name → gradient ndarray
        """
        c = self._cache
        probs, targets = c["probs"], c["targets"]
        T = c["T"]

        # ── 5→4: dL/d_logits (softmax + cross-entropy combined) ──
        # For each timestep: d_loss/d_logit_i = probs_i - 1{i==target}
        # We average over T, then sum across positions (all use h_attn once)
        d_logits = probs.copy()                               # (T, V)
        d_logits[np.arange(T), targets] -= 1.0
        d_logits /= T
        d_logits *= scale

        # Sum across timesteps (since logits_single was broadcast)
        d_logits_single = d_logits.sum(axis=0)               # (V,)

        # ── 4: output projection backward ────────────────────────
        h_attn = c["h_attn"]
        self.dW_out += np.outer(d_logits_single, h_attn)
        self.db_out += d_logits_single
        d_h_attn = self.W_out.T @ d_logits_single            # (H,)

        # ── 3: attention backward ─────────────────────────────────
        dH_states = self.attention.backward(d_h_attn)        # (T, H)

        # ── 2: GRU backward ───────────────────────────────────────
        d_emb = self.gru.backward(dH_states)                 # (T, d)

        # ── 1: embedding backward ─────────────────────────────────
        self.embedding.backward(d_emb)

        # ── Collect all gradients ─────────────────────────────────
        grads: dict[str, np.ndarray] = {}
        grads["W_out"] = self.dW_out
        grads["b_out"] = self.db_out
        grads.update({f"emb_{k}": v for k, v in self.embedding.grads().items()})
        grads.update({f"gru_{k}": v for k, v in self.gru.grads().items()})
        grads.update({f"attn_{k}": v for k, v in self.attention.grads().items()})
        return grads

    def zero_grad(self) -> None:
        self.dW_out[:] = 0.0
        self.db_out[:] = 0.0
        self.embedding.zero_grad()
        self.gru.zero_grad()
        self.attention.zero_grad()

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
        Generate `n` characters starting from `seed_tokens`.

        Parameters
        ----------
        seed_tokens : (T,) integer indices
        vocab       : Vocabulary instance (for decode)
        n           : number of tokens to generate
        temperature : sampling temperature (0 → greedy)
        top_k       : keep only top-k logits (0 → no truncation)

        Returns
        -------
        completion : str
        alpha      : (T, ) attention weights from last step
        """
        from .sampling import sample_token

        tokens = list(seed_tokens)
        alpha_last = np.zeros(len(seed_tokens))

        for _ in range(n):
            inp = np.array(tokens[-min(len(tokens), 512):], dtype=np.int32)
            probs, _ = self.forward(inp, targets=None, training=False)
            alpha_last = self._cache["alpha"]

            # Use the last timestep's probabilities
            logits = np.log(probs[-1] + 1e-12)
            next_tok = sample_token(logits, temperature, top_k)
            tokens.append(next_tok)

        generated = tokens[len(seed_tokens):]
        return vocab.decode(generated), alpha_last

    # ──────────────────────────────────────────
    # Param / grad / state helpers
    # ──────────────────────────────────────────

    def params(self) -> dict[str, np.ndarray]:
        p = {"W_out": self.W_out, "b_out": self.b_out}
        p.update({f"emb_{k}": v for k, v in self.embedding.params().items()})
        p.update({f"gru_{k}": v for k, v in self.gru.params().items()})
        p.update({f"attn_{k}": v for k, v in self.attention.params().items()})
        return p

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **self.params())
        print(f"[RNNLM] Saved checkpoint → {path}")

    def load(self, path: str) -> None:
        data = np.load(path + ".npz" if not path.endswith(".npz") else path)
        # Route each array back to its module
        emb_dict  = {k[4:]: v for k, v in data.items() if k.startswith("emb_")}
        gru_dict  = {k[4:]: v for k, v in data.items() if k.startswith("gru_")}
        attn_dict = {k[5:]: v for k, v in data.items() if k.startswith("attn_")}

        self.embedding.load_state_dict(emb_dict)
        self.gru.load_state_dict(gru_dict)
        self.attention.load_state_dict(attn_dict)
        self.W_out = data["W_out"].copy()
        self.b_out = data["b_out"].copy()
        print(f"[RNNLM] Loaded checkpoint ← {path}")
