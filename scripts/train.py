"""
train.py
--------
Training script for the RNN Language Model.

By default the script RESUMES from the best existing checkpoint if one is
found. The checkpoint's stored hyperparams (_vocab_size, _embed_dim,
_hidden_dim, _num_layers) are used to rebuild the model and DataLoader
automatically — no need to re-specify them on the command line.

Pass --scratch to ignore any checkpoint and train from random weights.

Usage:
    python -m scripts.train --mode char          # resume char-level (default)
    python -m scripts.train --mode word          # resume word-level
    python -m scripts.train --mode char --scratch
    python -m scripts.train --ablation no_attention
    python -m scripts.train --ablation single_layer
    python -m scripts.train --ablation no_dropout
    python -m scripts.train --ablation vanilla_rnn

CLI flags (only needed when training from --scratch or first run):
    --mode         str   "char" or "word"  (default "char")
    --scratch      flag  ignore checkpoint; train from random weights
    --epochs       int   (default 200)
    --lr           float (default 1e-3)
    --hidden       int   (default 256)   ← overridden by checkpoint if resuming
    --embed        int   (default 64)    ← overridden by checkpoint if resuming
    --seq_len      int   (default 100)
    --batch_size   int   (default 64)
    --steps        int   steps per epoch (default 200)
    --patience     int   early-stopping patience (default 20)
    --data_dir     str   (default data/raw/topics)
    --checkpoint   str   override checkpoint path
"""

import argparse
import csv
import os
import time

import numpy as np

from src.adam        import Adam
from src.data_loader import DataLoader
from src.rnnlm       import RNNLM
from src.settings    import BEST_MODEL_LETTER, BEST_MODEL_WORD


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers  (format-compatible with training notebooks)
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(params: dict, path: str) -> None:
    """Save a flat param dict to a .npz file."""
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **{k: np.array(v) for k, v in params.items()})
    print(f"  Saved  -> {path}")


def load_checkpoint(path: str) -> dict:
    """Load a .npz checkpoint into a plain NumPy param dict."""
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    data   = np.load(path)
    params = {k: np.array(v) for k, v in data.items()}
    print(f"  Loaded <- {path}  ({len(params)} arrays)")
    return params


def read_checkpoint_hparams(path: str) -> dict | None:
    """
    Read hyperparams from a checkpoint.

    First tries the explicit scalar keys (_vocab_size etc.) written by
    train.py.  If those are absent (e.g. checkpoint saved by a notebook),
    infers the dimensions from the weight array shapes:

        W_out  : (vocab_size, hidden_dim)
        emb_E  : (vocab_size, embed_dim)
        gru_*  : first weight shape reveals num_layers indirectly

    Returns None only if the file does not exist.
    Keys: vocab_size, embed_dim, hidden_dim, num_layers
    """
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    if not os.path.exists(path):
        return None

    data = np.load(path)

    # ── Prefer explicit scalar keys (written by train.py) ────────
    if "_vocab_size" in data:
        return {
            "vocab_size": int(data["_vocab_size"]),
            "embed_dim":  int(data["_embed_dim"]),
            "hidden_dim": int(data["_hidden_dim"]),
            "num_layers": int(data["_num_layers"]),
        }

    # ── Infer from weight shapes (notebook-saved checkpoints) ────
    # W_out shape: (vocab_size, hidden_dim)
    # emb_E shape: (vocab_size, embed_dim)
    # Count gru layers by counting distinct layer indices in keys
    keys = list(data.keys())

    if "W_out" not in data or "emb_E" not in data:
        print("[train] WARNING: checkpoint has no recognisable weight keys.")
        return None

    vocab_size, hidden_dim = data["W_out"].shape
    _,          embed_dim  = data["emb_E"].shape

    # Layer count: gru keys look like "gru_cells_0_Wz" or "gru_0_Wz"
    # Count unique layer indices present in key names
    import re as _re
    layer_indices = set()
    for k in keys:
        m = _re.search(r"(?:cells_|layer_?)(\d+)", k)
        if m:
            layer_indices.add(int(m.group(1)))
    num_layers = len(layer_indices) if layer_indices else 2  # default 2

    print(
        f"[train] Inferred hparams from weight shapes — "
        f"vocab={vocab_size}, embed={embed_dim}, "
        f"hidden={hidden_dim}, layers={num_layers}"
    )
    return {
        "vocab_size": vocab_size,
        "embed_dim":  embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }


def apply_checkpoint(model: RNNLM, params: dict) -> None:
    """Push a loaded param dict back into a model instance."""
    emb_dict  = {k[4:]: v for k, v in params.items() if k.startswith("emb_")}
    gru_dict  = {k[4:]: v for k, v in params.items() if k.startswith("gru_")}
    attn_dict = {k[5:]: v for k, v in params.items() if k.startswith("attn_")}
    model.embedding.load_state_dict(emb_dict)
    model.gru.load_state_dict(gru_dict)
    model.attention.load_state_dict(attn_dict)
    model.W_out = params["W_out"].copy()
    model.b_out = params["b_out"].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(vocab_size: int, embed_dim: int, hidden_dim: int,
                num_layers: int, ablation: str | None) -> RNNLM:
    """Construct an RNNLM (or ablation variant) with explicit dimensions."""
    kwargs = dict(
        vocab_size = vocab_size,
        embed_dim  = embed_dim,
        hidden_dim = hidden_dim,
        num_layers = num_layers,
        keep_prob  = 0.8,
        attn_dim   = None,
    )

    if ablation == "single_layer":
        kwargs["num_layers"] = 1
        print("[Ablation] single_layer: 1 GRU layer")

    elif ablation == "no_dropout":
        kwargs["keep_prob"] = 1.0
        print("[Ablation] no_dropout: keep_prob = 1.0")

    elif ablation == "no_attention":
        print("[Ablation] no_attention: bypassing BahdanauAttention")
        return NoAttentionRNNLM(**kwargs)

    elif ablation == "vanilla_rnn":
        print("[Ablation] vanilla_rnn: 1-layer GRU, no dropout")
        return VanillaRNNLM(**kwargs)

    return RNNLM(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: RNNLM,
    optimizer: Adam,
    loader: DataLoader,
    steps: int,
) -> float:
    """Run one training epoch; return mean loss."""
    total_loss = 0.0
    for step, (X, Y) in enumerate(loader.iter_epoch("train", steps)):
        model.zero_grad()
        batch_loss = 0.0

        for i in range(len(X)):
            _, loss = model.forward(X[i], Y[i], training=True)
            batch_loss += loss

        batch_loss /= len(X)
        total_loss += batch_loss

        grads  = model.backward(scale=1.0 / len(X))
        params = model.params()
        optimizer.step(params, grads)

        if (step + 1) % 50 == 0:
            print(f"    step {step+1}/{steps}  loss={batch_loss:.4f}")

    return total_loss / steps


# ──────────────────────────────────────────────────────────────────────────────
# Ablation variants
# ──────────────────────────────────────────────────────────────────────────────

class NoAttentionRNNLM(RNNLM):
    """RNNLM that skips BahdanauAttention; uses h_T (last hidden state) directly."""

    def forward(self, tokens, targets=None, training=True):
        from src.rnnlm import cross_entropy, softmax as _softmax
        emb      = self.embedding.forward(tokens)
        H_states = self.gru.forward(emb, training=training)
        h_T      = H_states[-1]

        logits_single = self.W_out @ h_T + self.b_out
        T      = len(tokens)
        logits = np.tile(logits_single, (T, 1))
        probs  = np.apply_along_axis(_softmax, 1, logits)

        loss = cross_entropy(probs, targets) if targets is not None else 0.0

        self._cache = dict(
            tokens=tokens, targets=targets,
            emb=emb, H_states=H_states,
            h_attn=h_T, alpha=np.ones(T) / T,
            logits_single=logits_single, probs=probs, T=T,
        )
        return probs, loss

    def backward(self, scale=1.0):
        c               = self._cache
        probs, targets  = c["probs"], c["targets"]
        T               = c["T"]

        d_logits                    = probs.copy()
        d_logits[np.arange(T), targets] -= 1.0
        d_logits                   /= T
        d_logits                   *= scale
        d_logits_single             = d_logits.sum(axis=0)

        self.dW_out += np.outer(d_logits_single, c["h_attn"])
        self.db_out += d_logits_single
        d_h_T        = self.W_out.T @ d_logits_single

        dH       = np.zeros_like(c["H_states"])
        dH[-1]   = d_h_T
        d_emb    = self.gru.backward(dH)
        self.embedding.backward(d_emb)

        grads = {"W_out": self.dW_out, "b_out": self.db_out}
        grads.update({f"emb_{k}": v for k, v in self.embedding.grads().items()})
        grads.update({f"gru_{k}": v for k, v in self.gru.grads().items()})
        return grads


class VanillaRNNLM(RNNLM):
    """Single-layer GRU, no dropout — ablation baseline."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from src.gru_layer import GRULayer
        self.gru = GRULayer(
            kwargs["embed_dim"], kwargs["hidden_dim"],
            num_layers=1, keep_prob=1.0,
        )
        print("  [VanillaRNNLM] 1-layer GRU, no dropout.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train RNN Language Model")
    parser.add_argument("--mode",       type=str,   default="char",
                        choices=["char", "word"],
                        help="Tokenisation level: 'char' or 'word'")
    parser.add_argument("--scratch",    action="store_true",
                        help="Train from random weights, ignoring any checkpoint.")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden",     type=int,   default=256,
                        help="Hidden dim (overridden by checkpoint when resuming)")
    parser.add_argument("--embed",      type=int,   default=64,
                        help="Embed dim (overridden by checkpoint when resuming)")
    parser.add_argument("--seq_len",    type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--steps",      type=int,   default=200)
    parser.add_argument("--patience",   type=int,   default=20)
    parser.add_argument("--data_dir",   type=str,   default="data/raw/topics")
    parser.add_argument("--checkpoint", type=str,   default=None,
                        help="Override checkpoint path.")
    parser.add_argument("--ablation",   type=str,   default=None,
                        choices=[None, "no_attention", "single_layer",
                                 "no_dropout", "vanilla_rnn"])
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("report",      exist_ok=True)

    # ── Resolve checkpoint path ───────────────────────────────────
    if args.checkpoint is None:
        base_ckpt = (BEST_MODEL_WORD if args.mode == "word"
                     else BEST_MODEL_LETTER)
    else:
        base_ckpt = args.checkpoint

    suffix    = f"_{args.ablation}" if args.ablation else ""
    # Resolve to absolute path so cwd at invocation never matters
    from pathlib import Path as _Path
    ckpt_path = str(_Path(base_ckpt).resolve()).removesuffix(".npz") + suffix
    ckpt_file = ckpt_path + ".npz"

    print(f"\n[train] Checkpoint : {ckpt_file}")
    print(f"[train] File exists : {os.path.exists(ckpt_file)}")

    # ── Read hyperparams from checkpoint (if resuming) ────────────
    resuming     = not args.scratch and os.path.exists(ckpt_file)
    ckpt_hparams = read_checkpoint_hparams(ckpt_file) if resuming else None

    if resuming and ckpt_hparams is not None:
        embed_dim  = ckpt_hparams["embed_dim"]
        hidden_dim = ckpt_hparams["hidden_dim"]
        num_layers = ckpt_hparams["num_layers"]
        ckpt_vocab = ckpt_hparams["vocab_size"]
        # For word mode: cap DataLoader vocab to match checkpoint
        max_vocab  = (ckpt_vocab - 2) if args.mode == "word" else None
        print(
            f"\n=== Resuming from checkpoint: {ckpt_file} ===\n"
            f"    Checkpoint hparams: vocab={ckpt_vocab}, embed={embed_dim}, "
            f"hidden={hidden_dim}, layers={num_layers}"
        )
    else:
        embed_dim  = args.embed
        hidden_dim = args.hidden
        num_layers = 2
        max_vocab  = None
        if args.scratch:
            print("\n=== Starting from scratch (--scratch flag set) ===")
        else:
            print(
                f"\n=== No checkpoint found at '{ckpt_file}' "
                "— starting from scratch ==="
            )

    # ── 1. Data ───────────────────────────────────────────────────
    print(f"\n=== Loading dataset (mode='{args.mode}') ===")
    loader = DataLoader(
        data_dir   = args.data_dir,
        seq_len    = args.seq_len,
        batch_size = args.batch_size,
        mode       = args.mode,
        max_vocab  = max_vocab,
    )

    # Sanity check: vocab must match checkpoint exactly
    if resuming and ckpt_hparams is not None and loader.vocab.size != ckpt_hparams["vocab_size"]:
        raise RuntimeError(
            f"Vocab size mismatch after applying max_vocab cap: "
            f"DataLoader={loader.vocab.size}, checkpoint={ckpt_hparams['vocab_size']}. "
            f"The corpus or data_dir may have changed since the checkpoint was saved."
        )

    # ── 2. Model ──────────────────────────────────────────────────
    print("\n=== Building model ===")
    model = build_model(
        vocab_size = loader.vocab.size,
        embed_dim  = embed_dim,
        hidden_dim = hidden_dim,
        num_layers = num_layers,
        ablation   = args.ablation,
    )
    print(
        f"    mode={args.mode}, vocab={loader.vocab.size}, "
        f"embed={embed_dim}, hidden={hidden_dim}, layers={num_layers}"
    )

    # ── 3. Load weights if resuming ───────────────────────────────
    if resuming:
        params = load_checkpoint(ckpt_file)
        apply_checkpoint(model, params)

    # ── 4. Optimiser ──────────────────────────────────────────────
    optimizer = Adam(lr=args.lr, clip=5.0)

    # ── 5. Training loop ──────────────────────────────────────────
    best_val_ppl     = float("inf")
    patience_counter = 0
    history: list[tuple] = []

    print("\n=== Training ===")
    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, optimizer, loader, args.steps)
        val_ppl    = loader.compute_perplexity(model, "val", steps=30)
        elapsed    = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"loss={train_loss:.4f}  val_ppl={val_ppl:.2f}  "
            f"({elapsed:.0f}s)"
        )
        history.append((epoch, train_loss, val_ppl))

        if val_ppl < best_val_ppl:
            best_val_ppl     = val_ppl
            patience_counter = 0
            # Build full payload: params + hyperparams
            payload = model.params()
            payload["_vocab_size"] = np.array(loader.vocab.size)
            payload["_embed_dim"]  = np.array(embed_dim)
            payload["_hidden_dim"] = np.array(hidden_dim)
            payload["_num_layers"] = np.array(model.gru.num_layers)
            save_checkpoint(payload, ckpt_path)
            print(f"  ✓ New best val_ppl={val_ppl:.2f} — saved → {ckpt_file}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping after epoch {epoch}.")
            break

    # ── 6. Training log ───────────────────────────────────────────
    log_path = f"report/training_log_{args.mode}{suffix}.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_ppl"])
        writer.writerows(history)
    print(f"\nTraining log → {log_path}")

    # ── 7. Test evaluation (reload best weights) ──────────────────
    best_params = load_checkpoint(ckpt_file)
    apply_checkpoint(model, best_params)
    test_ppl = loader.compute_perplexity(model, "test", steps=50)
    print(f"Final test perplexity ({args.mode}-level): {test_ppl:.2f}")

    # ── 8. Ablation log ───────────────────────────────────────────
    if args.ablation:
        abl_path   = "report/ablation_results.csv"
        file_exists = os.path.isfile(abl_path)
        with open(abl_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["variant", "mode", "best_val_ppl", "test_ppl"])
            writer.writerow([args.ablation, args.mode, best_val_ppl, test_ppl])
        print(f"Ablation results → {abl_path}")


if __name__ == "__main__":
    main()