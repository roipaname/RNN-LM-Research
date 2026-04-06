"""
data_loader.py
--------------
Loads the poetry corpus, builds the character vocabulary,
encodes all sequences, and provides batch sampling.

Split is done at POEM level (not line level) to prevent data leakage.
Partition: 80% train / 10% validation / 10% test.
"""

import os
import re
import random
import numpy as np


# ──────────────────────────────────────────────
# Special tokens
# ──────────────────────────────────────────────
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class Vocabulary:
    """Maps characters to integer indices and back."""

    def __init__(self):
        self.char2idx: dict[str, int] = {}
        self.idx2char: dict[int, str] = {}
        self._built = False

    def build(self, text: str) -> None:
        """Build vocab from a string of all training text."""
        chars = sorted(set(text))
        # Reserve index 0 for PAD, 1 for UNK
        special = [PAD_TOKEN, UNK_TOKEN]
        all_tokens = special + chars
        self.char2idx = {ch: i for i, ch in enumerate(all_tokens)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self._built = True

    @property
    def size(self) -> int:
        return len(self.char2idx)

    def encode(self, text: str) -> list[int]:
        unk = self.char2idx[UNK_TOKEN]
        return [self.char2idx.get(ch, unk) for ch in text]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.idx2char.get(i, UNK_TOKEN) for i in indices)


class DataLoader:
    """
    Loads .txt files from data_dir, splits poems, builds vocab,
    and provides random batch sampling for training.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 100,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.vocab = Vocabulary()

        # Raw poem strings per split
        self.splits: dict[str, list[str]] = {}

        # Encoded flat arrays per split (for fast slicing)
        self.encoded: dict[str, np.ndarray] = {}

        self._load_and_split()
        self._build_vocab()
        self._encode_splits()

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _load_poems(self) -> list[str]:
        """
        Read all .txt files in data_dir.
        Poems are separated by blank lines (standard Gutenberg format).
        """
        poems: list[str] = []
        txt_files=[]
        for genre in os.listdir(self.data_dir):
          genre_path = os.path.join(self.data_dir, genre)

          if not os.path.isdir(genre_path):
             continue  # skip files if any
          for f in os.listdir(genre_path):
                if f.endswith(".txt"):
                    file_path=os.path.join(genre_path,f)
                    txt_files.append(file_path)

        if not txt_files or len(txt_files)==0:
            raise FileNotFoundError(
                f"No .txt files found in '{self.data_dir}'. "
                "Please place your poetry corpus files there."
            )
        for path in sorted(txt_files):
            
            with open(path, encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
            # Split on double newlines to separate poems/stanzas
            blocks = re.split(r"\n{2,}", raw.strip())
            poems.extend([b.strip() for b in blocks if len(b.strip()) > 50])
        return poems

    def _load_and_split(self) -> None:
        poems = self._load_poems()
        self.rng.shuffle(poems)
        n = len(poems)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        self.splits["train"] = poems[:n_train]
        self.splits["val"] = poems[n_train : n_train + n_val]
        self.splits["test"] = poems[n_train + n_val :]
        print(
            f"[DataLoader] Poems — train: {len(self.splits['train'])}, "
            f"val: {len(self.splits['val'])}, test: {len(self.splits['test'])}"
        )

    def _build_vocab(self) -> None:
        train_text = "\n".join(self.splits["train"])
        self.vocab.build(train_text)
        print(f"[DataLoader] Vocabulary size: {self.vocab.size}")

    def _encode_splits(self) -> None:
        for split, poems in self.splits.items():
            text = "\n".join(poems)
            self.encoded[split] = np.array(
                self.vocab.encode(text), dtype=np.int32
            )
            print(
                f"[DataLoader] {split} encoded length: {len(self.encoded[split]):,} chars"
            )

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def sample_batch(
        self, split: str = "train"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample a random batch of (X, Y) pairs where:
          X: (batch_size, seq_len) — input token indices
          Y: (batch_size, seq_len) — target token indices (X shifted by 1)
        """
        data = self.encoded[split]
        max_start = len(data) - self.seq_len - 1
        if max_start <= 0:
            raise ValueError(
                f"Split '{split}' is too short for seq_len={self.seq_len}."
            )
        starts = self.np_rng.integers(0, max_start, size=self.batch_size)
        X = np.stack([data[s : s + self.seq_len] for s in starts])
        Y = np.stack([data[s + 1 : s + self.seq_len + 1] for s in starts])
        return X, Y

    def iter_epoch(
        self, split: str = "train", steps_per_epoch: int = 200
    ):
        """Yield (X, Y) batches for one epoch."""
        for _ in range(steps_per_epoch):
            yield self.sample_batch(split)

    def compute_perplexity(
        self, model, split: str = "val", steps: int = 50
    ) -> float:
        """
        Estimate perplexity on a split by averaging cross-entropy
        over `steps` random batches.
        """
        total_loss = 0.0
        for _ in range(steps):
            X, Y = self.sample_batch(split)
            # Evaluate one sequence at a time (batch size 1 for simplicity)
            for i in range(len(X)):
                _, loss = model.forward(X[i], Y[i])
                total_loss += loss
        avg_loss = total_loss / (steps * self.batch_size)
        return float(np.exp(avg_loss))
