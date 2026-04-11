"""
train.py
--------
Training script for the RNN Language Model.

By default the script RESUMES from the best existing checkpoint if one is
found. Pass --scratch to start from random weights instead.

Usage:
    python train.py                        # resume char-level (default)
    python train.py --scratch              # train char-level from scratch
    python train.py --mode word            # resume word-level model
    python train.py --mode word --scratch  # word-level from scratch
    python train.py --ablation no_attention
    python train.py --ablation single_layer
    python train.py --ablation no_dropout
    python train.py --ablation vanilla_rnn

Hyperparameters (adjust at the top of main() or via CLI flags):
    --mode         str   "char" or "word"  (default "char")
    --scratch      flag  start from random weights, ignoring any checkpoint
    --epochs       int   (default 200)
    --lr           float (default 1e-3)
    --hidden       int   (default 256)
    --embed        int   (default 64)
    --seq_len      int   (default 100)
    --batch_size   int   (default 64)
    --steps        int   (default 200 steps per epoch)
    --patience     int   (default 20 early-stopping patience)
    --data_dir     str   (default data/raw/topics)
    --checkpoint   str   (override checkpoint path)
"""

import argparse
import os
import csv
import time
import numpy as np

from src.data_loader import DataLoader
from src.rnnlm       import RNNLM
from src.adam        import Adam
from src.settings    import BEST_MODEL_WORD, BEST_MODEL_LETTER


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers  (format-compatible with the training notebook)
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(params: dict, path: str) -> None:
    """Save a flat param dict to a .npz file (matches notebook format)."""
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
    data = np.load(path)
    params = {k: np.array(v) for k, v in data.items()}
    print(f"  Loaded <- {path}  ({len(params)} arrays)")
    return params


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
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_model(args, vocab_size: int) -> RNNLM:
    """Construct an RNNLM according to args / ablation flag."""
    ablation = getattr(args, "ablation", None)

    kwargs = dict(
        vocab_size  = vocab_size,
        embed_dim   = args.embed,
        hidden_dim  = args.hidden,
        num_layers  = 2,
        keep_prob   = 0.8,
        attn_dim    = None,
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
        print("[Ablation] vanilla_rnn: using VanillaRNNLM (no gates)")
        return VanillaRNNLM(**kwargs)

    return RNNLM(**kwargs)


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

        grads = model.backward(scale=1.0 / len(X))
        params = model.params()
        optimizer.step(params, grads)

        if (step + 1) % 50 == 0:
            print(f"    step {step+1}/{steps}  loss={batch_loss:.4f}")

    return total_loss / steps


# ──────────────────────────────────────────────────────────────────────────────
# Ablation variants
# ──────────────────────────────────────────────────────────────────────────────

class NoAttentionRNNLM(RNNLM):
    """RNNLM variant that skips BahdanauAttention; uses h_T directly."""

    def forward(self, tokens, targets=None, training=True):
        emb      = self.embedding.forward(tokens)
        H_states = self.gru.forward(emb, training=training)
        h_T      = H_states[-1]

        logits_single = self.W_out @ h_T + self.b_out
        T = len(tokens)
        logits = np.tile(logits_single, (T, 1))

        from src.attention import softmax
        probs = np.apply_along_axis(softmax, 1, logits)

        loss = 0.0
        if targets is not None:
            from src.rnnlm import cross_entropy
            loss = cross_entropy(probs, targets)

        self._cache = dict(
            tokens=tokens, targets=targets,
            emb=emb, H_states=H_states,
            h_attn=h_T, alpha=np.ones(T)/T,
            logits_single=logits_single, probs=probs, T=T,
        )
        return probs, loss

    def backward(self, scale=1.0):
        c = self._cache
        probs, targets = c["probs"], c["targets"]
        T = c["T"]

        d_logits = probs.copy()
        d_logits[np.arange(T), targets] -= 1.0
        d_logits /= T
        d_logits *= scale
        d_logits_single = d_logits.sum(axis=0)

        h_T = c["h_attn"]
        self.dW_out += np.outer(d_logits_single, h_T)
        self.db_out += d_logits_single
        d_h_T = self.W_out.T @ d_logits_single

        dH = np.zeros_like(c["H_states"])
        dH[-1] = d_h_T

        d_emb = self.gru.backward(dH)
        self.embedding.backward(d_emb)

        grads = {"W_out": self.dW_out, "b_out": self.db_out}
        grads.update({f"emb_{k}": v for k, v in self.embedding.grads().items()})
        grads.update({f"gru_{k}": v for k, v in self.gru.grads().items()})
        return grads


class VanillaRNNLM(RNNLM):
    """
    Simplified RNNLM using a single-layer vanilla RNN (no gates, no attention).
    For ablation study only.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from src.gru_layer import GRULayer
        self.gru = GRULayer(
            kwargs["embed_dim"], kwargs["hidden_dim"],
            num_layers=1, keep_prob=1.0
        )
        print("  [VanillaRNNLM] Using 1-layer GRU without dropout as RNN proxy.")


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train RNN Language Model")
    parser.add_argument("--mode",        type=str,   default="char",
                        choices=["char", "word"],
                        help="Tokenisation level: 'char' or 'word'")
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--hidden",      type=int,   default=256)
    parser.add_argument("--embed",       type=int,   default=64)
    parser.add_argument("--seq_len",     type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--patience",    type=int,   default=20)
    parser.add_argument("--data_dir",    type=str,   default="data/raw/topics")
    parser.add_argument("--checkpoint",  type=str,   default=None,
                        help="Override checkpoint path. Defaults to "
                             "checkpoints/best_model_word or best_model_letter.")
    parser.add_argument("--scratch",     action="store_true",
                        help="Train from random weights, ignoring any existing checkpoint.")
    parser.add_argument("--ablation",    type=str,   default=None,
                        choices=[None, "no_attention", "single_layer",
                                 "no_dropout", "vanilla_rnn"])
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("report",      exist_ok=True)

    # ── Resolve checkpoint path from mode ────────────────────────
    if args.checkpoint is None:
        if args.mode == "word":
            ckpt_path = str(BEST_MODEL_WORD).replace(".npz", "")
        else:
            ckpt_path = str(BEST_MODEL_LETTER).replace(".npz", "")
    else:
        ckpt_path = args.checkpoint

    # Append ablation suffix if needed
    suffix = f"_{args.ablation}" if args.ablation else ""
    ckpt_path = ckpt_path + suffix
    ckpt_file = ckpt_path + ".npz"

    # ── 1. Data ─────────────────────────────────────────────────
    print(f"\n=== Loading dataset (mode='{args.mode}') ===")
    loader = DataLoader(
        data_dir   = args.data_dir,
        seq_len    = args.seq_len,
        batch_size = args.batch_size,
        mode       = args.mode,
    )

    # ── 2. Model ─────────────────────────────────────────────────
    print("\n=== Building model ===")
    model = build_model(args, vocab_size=loader.vocab.size)
    print(f"    mode={args.mode}, vocab_size={loader.vocab.size}, "
          f"hidden={args.hidden}, embed={args.embed}")

    # ── Resume or scratch ─────────────────────────────────────────
    if not args.scratch and os.path.exists(ckpt_file):
        print(f"\n=== Resuming from checkpoint: {ckpt_file} ===")
        params = load_checkpoint(ckpt_file)
        apply_checkpoint(model, params)
    elif args.scratch:
        print("\n=== Starting from scratch (--scratch flag set) ===")
    else:
        print(f"\n=== No checkpoint found at '{ckpt_file}' — starting from scratch ===")

    # ── 3. Optimiser ─────────────────────────────────────────────
    optimizer = Adam(lr=args.lr, clip=5.0)

    # ── 4. Training loop ─────────────────────────────────────────
    best_val_ppl = float("inf")
    patience_counter = 0
    history = []   # (epoch, train_loss, val_ppl)

    print("\n=== Training ===")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, loader, args.steps)
        val_ppl    = loader.compute_perplexity(model, "val", steps=30)
        elapsed    = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"loss={train_loss:.4f}  val_ppl={val_ppl:.2f}  "
              f"({elapsed:.0f}s)")

        history.append((epoch, train_loss, val_ppl))

        # ── Checkpointing ─────────────────────────────────────────
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            patience_counter = 0
            save_checkpoint(model.params(), ckpt_path)
            print(f"  ✓ New best val_ppl={val_ppl:.2f} — checkpoint saved → {ckpt_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after epoch {epoch}.")
            break

    # ── 5. Save training log ─────────────────────────────────────
    log_path = f"report/training_log_{args.mode}{suffix}.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_ppl"])
        writer.writerows(history)
    print(f"\nTraining log saved → {log_path}")

    # ── 6. Test evaluation ───────────────────────────────────────
    apply_checkpoint(model, load_checkpoint(ckpt_file))
    test_ppl = loader.compute_perplexity(model, "test", steps=50)
    print(f"\nFinal test perplexity ({args.mode}-level): {test_ppl:.2f}")

    # ── 7. Ablation comparison ───────────────────────────────────
    if args.ablation:
        ablation_path = "report/ablation_results.csv"
        file_exists = os.path.isfile(ablation_path)
        with open(ablation_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["variant", "mode", "best_val_ppl", "test_ppl"])
            writer.writerow([args.ablation, args.mode, best_val_ppl, test_ppl])
        print(f"Ablation results appended → {ablation_path}")


if __name__ == "__main__":
    main()