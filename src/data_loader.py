"""
data_loader.py
--------------
Loads the poetry corpus, builds the character or word vocabulary,
encodes all sequences, and provides batch sampling.

Split is done at POEM level (not line level) to prevent data leakage.
Partition: 80% train / 10% validation / 10% test.

Pass mode="char" (default) for the character-level model,
or mode="word" for the word-level model.

Pass max_vocab=N to cap word-level vocab to the N most frequent tokens
(rare words map to <UNK>). This is required when resuming a checkpoint
that was trained with a smaller vocabulary — set max_vocab to
(checkpoint _vocab_size - 2) to exclude the 2 special tokens.
"""

import os
import re
import random
from collections import Counter
from pathlib import Path

import numpy as np

from src.settings import GENRES_DATA_DIR

# ──────────────────────────────────────────────
# Special tokens
# ──────────────────────────────────────────────
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class Vocabulary:
    """Maps characters (or words) to integer indices and back."""

    def __init__(self):
        self.char2idx: dict[str, int] = {}
        self.idx2char: dict[int, str] = {}
        self.mode: str = "char"
        self._built = False

    def build(self, text: str, mode: str = "char",
              max_vocab: int | None = None) -> None:
        """
        Build vocab from a string of all training text.

        Parameters
        ----------
        text      : full training corpus as a single string
        mode      : "char" or "word"
        max_vocab : if set, keep only the top-N most frequent tokens
                    (word mode only). Rare tokens map to <UNK>.
                    Ignored in char mode.
        """
        self.mode = mode

        if mode == "char":
            tokens = sorted(set(text))
        elif mode == "word":
            words = text.split()
            if max_vocab is not None:
                counts = Counter(words)
                tokens = sorted(w for w, _ in counts.most_common(max_vocab))
            else:
                tokens = sorted(set(words))
        else:
            raise ValueError(f"Unknown vocab mode '{mode}'. Use 'char' or 'word'.")

        # Reserve index 0 for PAD, 1 for UNK
        special = [PAD_TOKEN, UNK_TOKEN]
        all_tokens = special + tokens
        self.char2idx = {tok: i for i, tok in enumerate(all_tokens)}
        self.idx2char = {i: tok for tok, i in self.char2idx.items()}
        self._built = True

    @property
    def size(self) -> int:
        return len(self.char2idx)

    def encode(self, text: str) -> list[int]:
        unk = self.char2idx[UNK_TOKEN]
        if self.mode == "char":
            return [self.char2idx.get(ch, unk) for ch in text]
        else:
            return [self.char2idx.get(w, unk) for w in text.split()]

    def decode(self, indices: list[int]) -> str:
        tokens = [self.idx2char.get(i, UNK_TOKEN) for i in indices]
        if self.mode == "char":
            return "".join(tokens)
        else:
            return " ".join(tokens)


class DataLoader:
    """
    Loads .txt files from data_dir, splits poems, builds vocab,
    and provides random batch sampling for training.

    Parameters
    ----------
    data_dir   : path to genre folders containing .txt files
    seq_len    : context window length (in tokens)
    batch_size : number of sequences per batch
    seed       : random seed
    mode       : "char" or "word"
    max_vocab  : cap word-level vocabulary to the top-N most frequent tokens.
                 Set to (checkpoint _vocab_size - 2) when resuming a word-level
                 checkpoint so vocab sizes stay consistent.
                 Ignored in char mode.
    """

    def __init__(
        self,
        data_dir: str | Path = GENRES_DATA_DIR,
        seq_len: int = 100,
        batch_size: int = 64,
        seed: int = 42,
        mode: str = "char",
        max_vocab: int | None = None,
    ):
        self.data_dir  = Path(data_dir)
        self.seq_len   = seq_len
        self.batch_size = batch_size
        self.rng       = random.Random(seed)
        self.np_rng    = np.random.default_rng(seed)
        self.mode      = mode
        self.max_vocab = max_vocab

        self.vocab = Vocabulary()

        self.splits:  dict[str, list[str]]  = {}
        self.encoded: dict[str, np.ndarray] = {}

        self._load_and_split()
        self._build_vocab()
        self._encode_splits()

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _load_poems(self) -> list[str]:
        """Read all .txt files under data_dir/genre/ sub-directories."""
        txt_files: list[Path] = []
        for genre_path in sorted(self.data_dir.iterdir()):
            if genre_path.is_dir():
                txt_files.extend(sorted(genre_path.glob("*.txt")))

        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found under '{self.data_dir}'. "
                "Please place your poetry corpus files there."
            )

        poems: list[str] = []
        for path in txt_files:
            raw = path.read_text(encoding="utf-8", errors="replace")
            blocks = re.split(r"\n{2,}", raw.strip())
            poems.extend(b.strip() for b in blocks if len(b.strip()) > 50)
        return poems

    def _load_and_split(self) -> None:
        poems = self._load_poems()
        self.rng.shuffle(poems)
        n       = len(poems)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)
        self.splits["train"] = poems[:n_train]
        self.splits["val"]   = poems[n_train : n_train + n_val]
        self.splits["test"]  = poems[n_train + n_val :]
        print(
            f"[DataLoader] Poems — train: {len(self.splits['train'])}, "
            f"val: {len(self.splits['val'])}, test: {len(self.splits['test'])}"
        )

    def _build_vocab(self) -> None:
        train_text    = "\n".join(self.splits["train"])
        effective_max = self.max_vocab if self.mode == "word" else None
        self.vocab.build(train_text, mode=self.mode, max_vocab=effective_max)
        cap_note = f", capped at {self.max_vocab}" if effective_max else ""
        print(
            f"[DataLoader] Vocabulary size ({self.mode}-level{cap_note}): "
            f"{self.vocab.size}"
        )

    def _encode_splits(self) -> None:
        for split, poems in self.splits.items():
            text = "\n".join(poems)
            self.encoded[split] = np.array(
                self.vocab.encode(text), dtype=np.int32
            )
            print(
                f"[DataLoader] {split} encoded length: "
                f"{len(self.encoded[split]):,} tokens"
            )

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def sample_batch(
        self, split: str = "train"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample a random batch:
          X: (batch_size, seq_len) — input token indices
          Y: (batch_size, seq_len) — targets (X shifted right by 1)
        """
        data      = self.encoded[split]
        max_start = len(data) - self.seq_len - 1
        if max_start <= 0:
            raise ValueError(
                f"Split '{split}' is too short for seq_len={self.seq_len}."
            )
        starts = self.np_rng.integers(0, max_start, size=self.batch_size)
        X = np.stack([data[s     : s + self.seq_len]     for s in starts])
        Y = np.stack([data[s + 1 : s + self.seq_len + 1] for s in starts])
        return X, Y

    def iter_epoch(self, split: str = "train", steps_per_epoch: int = 200):
        """Yield (X, Y) batches for one epoch."""
        for _ in range(steps_per_epoch):
            yield self.sample_batch(split)

    def compute_perplexity(
        self, model, split: str = "val", steps: int = 50
    ) -> float:
        """
        Estimate perplexity on `split` by averaging cross-entropy
        over `steps` random batches (inference mode — no dropout).
        """
        total_loss = 0.0
        for _ in range(steps):
            X, Y = self.sample_batch(split)
            for i in range(len(X)):
                _, loss = model.forward(X[i], Y[i], training=False)
                total_loss += loss
        avg_loss = total_loss / (steps * self.batch_size)
        return float(np.exp(avg_loss))


if __name__ == "__main__":
    print("🔍 Testing DataLoader...\n")
    for mode in ["char", "word"]:
        print(f"\n{'='*40}\n  Mode: {mode}-level\n{'='*40}")
        loader = DataLoader(mode=mode)
        for split in ["train", "val", "test"]:
            print(f"  {split}: {len(loader.splits[split])} poems")
        print(f"  Vocab: {loader.vocab.size}")
        X, Y = loader.sample_batch("train")
        print(f"  Batch X: {X.shape}, Y: {Y.shape}")
    print("\n✅ DataLoader test complete!")