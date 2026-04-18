"""
train.py
--------
Training script for the Transformer Language Model.

By default the script RESUMES from the best existing checkpoint if one is
found.  The checkpoint's stored hyperparams (_vocab_size, _embed_dim,
_num_heads, _num_layers, _ffn_dim) are used to rebuild the model and
DataLoader automatically.

Pass --scratch to ignore any checkpoint and train from random weights.

Usage:
    python -m scripts.train                  # resume (default)
    python -m scripts.train --scratch        # fresh run
    python -m scripts.train --ablation no_pe
    python -m scripts.train --ablation single_layer
    python -m scripts.train --ablation no_dropout
    python -m scripts.train --ablation small

CLI flags (only needed when training from --scratch or first run):
    --scratch        flag   ignore checkpoint; train from random weights
    --epochs         int    (default 50)
    --lr             float  (default 3e-4)
    --embed          int    d_model       (default 256)
    --heads          int    num_heads     (default 8)
    --layers         int    num_layers    (default 4)
    --ffn_dim        int    FFN inner dim (default 1024)
    --dropout        float  (default 0.10)
    --seq_len        int    (default 64)
    --batch_size     int    (default 32)
    --steps          int    steps per epoch (default 300)
    --val_steps      int    validation steps per epoch (default 20)
    --patience       int    early-stopping patience (default 15)
    --warmup         int    LR warmup steps (default 1000)
    --max_files      int    cap on .txt files loaded (default 500)
    --vocab_cap      int    vocabulary size cap (default 8000)
    --data_dir       str    (default data/raw/topics)
    --checkpoint     str    override checkpoint path
    --ablation       str    one of: no_pe, single_layer, no_dropout, small
"""

import argparse
import csv
import math
import os
import time

import numpy as np

from src.adam          import Adam
from src.data_loader   import DataLoader
from src.transformerlm import TransformerLM
from src.settings      import BEST_MODEL_TRANSFORMER, CHECKPOINTS_DIR


# ══════════════════════════════════════════════════════════════════════════════
# LR Schedule  (cosine decay with linear warm-up — pure NumPy)
# ══════════════════════════════════════════════════════════════════════════════

class WarmupCosineSchedule:
    """
    Linear warm-up for `warmup_steps`, then cosine decay to `end_lr`.

    Call .get(step) to retrieve the current learning rate.
    """

    def __init__(
        self,
        peak_lr:      float,
        warmup_steps: int,
        total_steps:  int,
        end_lr:       float | None = None,
    ):
        self.peak_lr      = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.end_lr       = end_lr if end_lr is not None else peak_lr * 0.05

    def get(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.end_lr + (self.peak_lr - self.end_lr) * cosine


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def read_checkpoint_hparams(path: str) -> dict | None:
    """
    Read hyperparams from a .npz checkpoint.

    Returns a dict with keys:
        vocab_size, embed_dim, num_heads, num_layers, ffn_dim, max_seq_len
    Returns None if the file does not exist or contains no hparam keys.
    """
    fpath = str(path)
    if not fpath.endswith(".npz"):
        fpath += ".npz"
    if not os.path.exists(fpath):
        return None

    data = np.load(fpath)
    if "_vocab_size" not in data:
        print(f"[train] WARNING: checkpoint '{fpath}' has no hparam metadata.")
        return None

    return {
        "vocab_size" : int(data["_vocab_size"]),
        "embed_dim"  : int(data["_embed_dim"]),
        "num_heads"  : int(data["_num_heads"]),
        "num_layers" : int(data["_num_layers"]),
        "ffn_dim"    : int(data["_ffn_dim"]),
        "max_seq_len": int(data.get("_max_seq_len", 512)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Ablation variants
# ══════════════════════════════════════════════════════════════════════════════

class NoPETransformerLM(TransformerLM):
    """Ablation: remove positional encoding (PE is zeroed)."""

    def forward(self, tokens, targets=None, training=True):
        # Bypass positional encoding by patching pos_enc.get temporarily
        _orig = self.pos_enc.get

        def _zero_pe(T):
            return np.zeros((T, self.D), dtype=np.float32)

        self.pos_enc.get = _zero_pe
        result = super().forward(tokens, targets, training)
        self.pos_enc.get = _orig
        return result


def build_model(
    vocab_size: int,
    embed_dim:  int,
    num_heads:  int,
    num_layers: int,
    ffn_dim:    int,
    dropout:    float,
    ablation:   str | None,
    max_seq_len: int = 512,
    seed:       int  = 0,
) -> TransformerLM:
    """Construct a TransformerLM (or ablation variant)."""
    base_kwargs = dict(
        vocab_size    = vocab_size,
        embed_dim     = embed_dim,
        num_heads     = num_heads,
        num_layers    = num_layers,
        ffn_dim       = ffn_dim,
        dropout_rate  = dropout,
        max_seq_len   = max_seq_len,
        seed          = seed,
    )

    if ablation == "single_layer":
        base_kwargs["num_layers"] = 1
        print("[Ablation] single_layer: 1 Transformer block")

    elif ablation == "no_dropout":
        base_kwargs["dropout_rate"] = 0.0
        print("[Ablation] no_dropout: dropout_rate = 0.0")

    elif ablation == "small":
        base_kwargs.update(embed_dim=128, num_heads=4, ffn_dim=256, num_layers=2)
        print("[Ablation] small: d=128, 4H, 2L, ffn=256")

    elif ablation == "no_pe":
        print("[Ablation] no_pe: positional encoding zeroed")
        return NoPETransformerLM(**base_kwargs)

    return TransformerLM(**base_kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:     TransformerLM,
    optimizer: Adam,
    schedule:  WarmupCosineSchedule,
    loader:    DataLoader,
    steps:     int,
    step_offset: int,
) -> tuple[float, int]:
    """
    Run one training epoch.

    Returns
    -------
    mean_loss   : float
    step_offset : int — updated global step count (for LR schedule)
    """
    total_loss = 0.0

    for step_idx, (X, Y) in enumerate(loader.iter_epoch("train", steps)):
        global_step = step_offset + step_idx

        # Update Adam LR from schedule
        optimizer.lr = schedule.get(global_step)

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

        if (step_idx + 1) % 50 == 0:
            cur_lr = schedule.get(global_step)
            ppl    = math.exp(min(batch_loss, 20))
            print(
                f"    step {step_idx+1:4d}/{steps}"
                f"  loss={batch_loss:.4f}"
                f"  ppl={ppl:.1f}"
                f"  lr={cur_lr:.2e}"
            )

    return total_loss / steps, step_offset + steps


def estimate_val_ppl(
    model:  TransformerLM,
    loader: DataLoader,
    steps:  int = 20,
) -> float:
    """Estimate validation perplexity over `steps` random batches."""
    total_loss = 0.0
    count      = 0
    for _ in range(steps):
        X, Y = loader.sample_batch("val")
        for i in range(len(X)):
            _, loss = model.forward(X[i], Y[i], training=False)
            total_loss += loss
            count      += 1
    avg = total_loss / max(count, 1)
    return math.exp(min(avg, 20))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")

    # ── Resume / scratch ──────────────────────────────────────────────────────
    parser.add_argument("--scratch",     action="store_true",
                        help="Train from random weights, ignoring any checkpoint.")

    # ── Architecture (only needed on first run / --scratch) ───────────────────
    parser.add_argument("--embed",       type=int,   default=256,
                        help="d_model  (overridden by checkpoint when resuming)")
    parser.add_argument("--heads",       type=int,   default=8,
                        help="num_heads (overridden by checkpoint when resuming)")
    parser.add_argument("--layers",      type=int,   default=4,
                        help="num_layers (overridden by checkpoint when resuming)")
    parser.add_argument("--ffn_dim",     type=int,   default=1024,
                        help="FFN inner dim (overridden by checkpoint when resuming)")
    parser.add_argument("--dropout",     type=float, default=0.10)

    # ── Training schedule ─────────────────────────────────────────────────────
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--warmup",      type=int,   default=1000,
                        help="Linear LR warmup steps")
    parser.add_argument("--steps",       type=int,   default=300,
                        help="Gradient steps per epoch")
    parser.add_argument("--val_steps",   type=int,   default=20,
                        help="Batches used to estimate val perplexity")
    parser.add_argument("--patience",    type=int,   default=15,
                        help="Early-stopping patience (epochs)")
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--seq_len",     type=int,   default=64)

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--max_files",   type=int,   default=500)
    parser.add_argument("--vocab_cap",   type=int,   default=8000)
    parser.add_argument("--data_dir",    type=str,   default="data/raw/topics")

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint",  type=str,   default=None,
                        help="Override checkpoint path.")
    parser.add_argument("--ablation",    type=str,   default=None,
                        choices=[None, "no_pe", "single_layer",
                                 "no_dropout", "small"])
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("report",      exist_ok=True)

    # ── Resolve checkpoint path ───────────────────────────────────────────────
    from pathlib import Path
    base_ckpt = Path(args.checkpoint) if args.checkpoint else Path(BEST_MODEL_TRANSFORMER)
    suffix    = f"_{args.ablation}" if args.ablation else ""
    ckpt_stem = str(base_ckpt).removesuffix(".npz") + suffix
    ckpt_file = ckpt_stem + ".npz"

    print(f"\n[train] Checkpoint : {ckpt_file}")
    print(f"[train] Exists     : {os.path.exists(ckpt_file)}")

    # ── Decide whether to resume ──────────────────────────────────────────────
    resuming     = not args.scratch and os.path.exists(ckpt_file)
    ckpt_hparams = read_checkpoint_hparams(ckpt_file) if resuming else None

    if resuming and ckpt_hparams:
        embed_dim   = ckpt_hparams["embed_dim"]
        num_heads   = ckpt_hparams["num_heads"]
        num_layers  = ckpt_hparams["num_layers"]
        ffn_dim     = ckpt_hparams["ffn_dim"]
        max_seq_len = ckpt_hparams["max_seq_len"]
        ckpt_vocab  = ckpt_hparams["vocab_size"]
        print(
            f"\n=== Resuming from checkpoint ===\n"
            f"    vocab={ckpt_vocab}  embed={embed_dim}  heads={num_heads}"
            f"  layers={num_layers}  ffn={ffn_dim}"
        )
    else:
        embed_dim   = args.embed
        num_heads   = args.heads
        num_layers  = args.layers
        ffn_dim     = args.ffn_dim
        max_seq_len = args.seq_len * 2   # generous context budget
        ckpt_vocab  = None
        if args.scratch:
            print("\n=== Starting from scratch (--scratch) ===")
        else:
            print(f"\n=== No checkpoint at '{ckpt_file}' — starting from scratch ===")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print(f"\n=== Loading dataset (word-level, max_files={args.max_files}) ===")
    loader = DataLoader(
        data_dir       = args.data_dir,
        seq_len        = args.seq_len,
        batch_size     = args.batch_size,
        max_files      = args.max_files,
        vocab_size_cap = args.vocab_cap,
    )

    # Vocab sanity check when resuming
    if resuming and ckpt_vocab is not None and loader.vocab.size != ckpt_vocab:
        raise RuntimeError(
            f"Vocab size mismatch: DataLoader={loader.vocab.size}, "
            f"checkpoint={ckpt_vocab}. "
            "The corpus or max_files cap may differ from when the checkpoint was saved. "
            "Pass --scratch to train fresh, or restore the original corpus."
        )

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("\n=== Building model ===")
    model = build_model(
        vocab_size  = loader.vocab.size,
        embed_dim   = embed_dim,
        num_heads   = num_heads,
        num_layers  = num_layers,
        ffn_dim     = ffn_dim,
        dropout     = args.dropout,
        ablation    = args.ablation,
        max_seq_len = max_seq_len,
    )
    n_params = sum(v.size for v in model.params().values())
    print(
        f"    vocab={loader.vocab.size}  embed={embed_dim}  heads={num_heads}"
        f"  layers={num_layers}  ffn={ffn_dim}\n"
        f"    Parameters: {n_params:,}"
    )

    # ── 3. Load weights if resuming ───────────────────────────────────────────
    if resuming:
        model.load(ckpt_file)

    # ── 4. Optimiser + LR schedule ────────────────────────────────────────────
    total_steps = args.epochs * args.steps
    schedule    = WarmupCosineSchedule(
        peak_lr      = args.lr,
        warmup_steps = args.warmup,
        total_steps  = total_steps,
    )
    optimizer = Adam(lr=schedule.get(0), clip=1.0)

    # ── 5. Training loop ──────────────────────────────────────────────────────
    best_val_ppl     = float("inf")
    patience_counter = 0
    global_step      = 0
    history: list[tuple] = []

    print(f"\n=== Training: {args.epochs} epochs × {args.steps} steps ==="
          f" (total {total_steps:,} steps)")
    print(f"    batch={args.batch_size}  seq_len={args.seq_len}"
          f"  peak_lr={args.lr}  warmup={args.warmup}")
    print("=" * 65)

    t_total = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        print(f"\nEpoch {epoch:03d}/{args.epochs}")
        train_loss, global_step = train_one_epoch(
            model, optimizer, schedule, loader, args.steps, global_step
        )

        val_ppl = estimate_val_ppl(model, loader, args.val_steps)
        elapsed = time.time() - t0
        total_m = (time.time() - t_total) / 60
        cur_lr  = schedule.get(global_step)

        train_ppl = math.exp(min(train_loss, 20))
        print(
            f"─── Epoch {epoch:03d}  "
            f"train_loss={train_loss:.4f} (ppl={train_ppl:.1f})  "
            f"val_ppl={val_ppl:.2f}  "
            f"lr={cur_lr:.2e}  "
            f"{elapsed:.0f}s  [{total_m:.1f} min total]"
        )
        history.append((epoch, train_loss, train_ppl, val_ppl, cur_lr))

        # ── Save best checkpoint ──────────────────────────────────────────────
        if val_ppl < best_val_ppl:
            best_val_ppl     = val_ppl
            patience_counter = 0
            model.save(ckpt_stem)
            print(f"  ⭐ New best val_ppl={best_val_ppl:.2f}  →  {ckpt_file}")
        else:
            patience_counter += 1
            print(f"  No improvement  ({patience_counter}/{args.patience})")

        # ── Periodic checkpoint every 10 epochs ──────────────────────────────
        if epoch % 10 == 0:
            periodic = ckpt_stem + f"_ep{epoch:03d}"
            model.save(periodic)
            print(f"  💾 Periodic checkpoint → {periodic}.npz")

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_counter >= args.patience:
            print(f"\n⏹  Early stopping triggered after epoch {epoch}.")
            break

    # ── 6. Training log ───────────────────────────────────────────────────────
    log_path = f"report/training_log_transformer{suffix}.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_ppl", "val_ppl", "lr"])
        writer.writerows(history)
    print(f"\nTraining log → {log_path}")

    # ── 7. Test evaluation (reload best weights) ──────────────────────────────
    model.load(ckpt_file)
    test_ppl = loader.compute_perplexity(model, "test", steps=50)
    print(f"Final test perplexity: {test_ppl:.2f}")
    print(f"Best val  perplexity: {best_val_ppl:.2f}")

    # ── 8. Ablation log ───────────────────────────────────────────────────────
    if args.ablation:
        abl_path    = "report/ablation_results.csv"
        file_exists = os.path.isfile(abl_path)
        with open(abl_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["variant", "best_val_ppl", "test_ppl",
                                 "embed_dim", "num_heads", "num_layers", "ffn_dim"])
            writer.writerow([args.ablation, best_val_ppl, test_ppl,
                             embed_dim, num_heads, num_layers, ffn_dim])
        print(f"Ablation results appended → {abl_path}")


if __name__ == "__main__":
    main()