"""
data_loader.py
--------------
Loads the poetry corpus, builds the word-level vocabulary,
encodes all sequences, and provides batch sampling.

Split is done at PASSAGE level (poem stanzas separated by blank lines)
to prevent data leakage.  Partition: 80% train / 10% val / 10% test.

Changes from the original:
  • Word-level only (mode="word" is the default and only supported mode)
  • max_files cap (default 500) — uses only the first N .txt files
  • Extended special tokens: <PAD>, <UNK>, <NL>, <BOS>, <EOS>
  • vocab_size_cap: keeps only the top-N most frequent words
  • tokenize() preserves newlines as <NL> tokens (important for poetry)
  • DataLoader.jax_batch() removed (JAX-specific; not needed in pure-NumPy path)
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
NL_TOKEN  = "<NL>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, NL_TOKEN, BOS_TOKEN, EOS_TOKEN]


# ──────────────────────────────────────────────
# Tokenisation helpers
# ──────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Split text into word tokens, converting line breaks to <NL> tokens.
    """
    tokens: list[str] = []
    for line in text.split("\n"):
        words = line.strip().split()
        tokens.extend(words)
        tokens.append(NL_TOKEN)
    return tokens


def detokenize(tokens: list[str]) -> str:
    """Convert token list back to a human-readable string."""
    parts: list[str] = []
    for tok in tokens:
        if tok == NL_TOKEN:
            parts.append("\n")
        elif tok in (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN):
            pass
        else:
            parts.append(tok + " ")
    return "".join(parts)


# ──────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────

class Vocabulary:
    """
    Word-level vocabulary with special tokens.

    Attributes
    ----------
    word2idx : dict[str, int]
    idx2word : dict[int, str]
    """

    def __init__(self) -> None:
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._built = False

    def build(self, tokens: list[str], vocab_size_cap: int = 8000) -> None:
        """
        Build vocabulary from a flat token list.

        Parameters
        ----------
        tokens        : list of string tokens from the training split
        vocab_size_cap: maximum vocabulary size (incl. special tokens)
        """
        counts  = Counter(tokens)
        n_slots = vocab_size_cap - len(SPECIAL_TOKENS)
        common  = [w for w, _ in counts.most_common(n_slots)
                   if w not in SPECIAL_TOKENS]
        all_toks = SPECIAL_TOKENS + common

        self.word2idx = {w: i for i, w in enumerate(all_toks)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self._built   = True

        unk_count = sum(c for w, c in counts.items() if w not in self.word2idx)
        print(
            f"[Vocabulary] size={self.size:,}  "
            f"(corpus had {len(counts):,} unique words, "
            f"{unk_count:,} token occurrences → <UNK>)"
        )

    @property
    def size(self) -> int:
        return len(self.word2idx)

    def encode(self, text: str) -> list[int]:
        """Encode a raw string to a list of token indices."""
        unk = self.word2idx[UNK_TOKEN]
        return [self.word2idx.get(t, unk) for t in tokenize(text)]

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        """Encode a pre-tokenised list."""
        unk = self.word2idx[UNK_TOKEN]
        return [self.word2idx.get(t, unk) for t in tokens]

    def decode(self, indices: list[int]) -> str:
        """Decode a list of indices back to a string."""
        return detokenize([self.idx2word.get(int(i), UNK_TOKEN) for i in indices])

    # ── Serialisation (used by TransformerLM checkpoint helpers) ──

    def state_dict(self) -> dict:
        return {"word2idx": dict(self.word2idx)}

    def load_state_dict(self, d: dict) -> None:
        self.word2idx = d["word2idx"]
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self._built   = True


# ──────────────────────────────────────────────
# File I/O helper
# ──────────────────────────────────────────────

def _read_file(path: str | Path) -> list[str]:
    """
    Read one .txt file and return a list of non-trivial passage strings.
    Passages are separated by blank lines (standard Gutenberg format).
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        blocks = re.split(r"\n{2,}", raw.strip())
        return [b.strip() for b in blocks if len(b.strip().split()) > 5]
    except Exception as exc:
        print(f"[DataLoader] Warning: could not read {path}: {exc}")
        return []


# ──────────────────────────────────────────────
# DataLoader
# ──────────────────────────────────────────────

class DataLoader:
    """
    Loads .txt files from data_dir, builds a word-level vocabulary,
    encodes all passages, and provides random batch sampling.

    Parameters
    ----------
    data_dir      : path to genre folders containing .txt files
    seq_len       : context window length in tokens
    batch_size    : sequences per batch
    seed          : random seed
    max_files     : cap on number of .txt files to load (default 500)
    vocab_size_cap: maximum vocabulary size (default 8000)
    """

    def __init__(
        self,
        data_dir: str | Path = GENRES_DATA_DIR,
        seq_len: int          = 64,
        batch_size: int       = 32,
        seed: int             = 42,
        max_files: int        = 500,
        vocab_size_cap: int   = 8000,
    ):
        self.data_dir      = Path(data_dir)
        self.seq_len       = seq_len
        self.batch_size    = batch_size
        self.max_files     = max_files
        self.vocab_size_cap= vocab_size_cap
        self.rng           = random.Random(seed)
        self.np_rng        = np.random.default_rng(seed)

        self.vocab   = Vocabulary()
        self.splits:  dict[str, list[str]]    = {}
        self.encoded: dict[str, np.ndarray]   = {}

        self._load_and_split()
        self._build_vocab()
        self._encode_splits()

    # ── Internal helpers ──────────────────────────────────────────

    def _collect_files(self) -> list[Path]:
        files: list[Path] = []
        for genre in sorted(os.listdir(self.data_dir)):
            gp = self.data_dir / genre
            if not gp.is_dir():
                continue
            for f in sorted(os.listdir(gp)):
                if f.endswith(".txt"):
                    files.append(gp / f)
        files = files[: self.max_files]
        print(f"[DataLoader] Using {len(files):,} .txt files (cap={self.max_files})")
        if not files:
            raise FileNotFoundError(
                f"No .txt files found under '{self.data_dir}'. "
                "Please check your data directory."
            )
        return files

    def _load_passages(self) -> list[str]:
        files    = self._collect_files()
        passages: list[str] = []
        for i, path in enumerate(files, 1):
            passages.extend(_read_file(path))
            if i % 50 == 0 or i == len(files):
                print(f"[DataLoader]   {i}/{len(files)} files → {len(passages):,} passages")
        print(f"[DataLoader] Total passages: {len(passages):,}")
        return passages

    def _load_and_split(self) -> None:
        passages = self._load_passages()
        self.rng.shuffle(passages)
        n       = len(passages)
        n_train = int(0.80 * n)
        n_val   = int(0.10 * n)
        self.splits["train"] = passages[:n_train]
        self.splits["val"]   = passages[n_train: n_train + n_val]
        self.splits["test"]  = passages[n_train + n_val:]
        for split, v in self.splits.items():
            print(f"[DataLoader] {split:5s}: {len(v):,} passages")

    def _build_vocab(self) -> None:
        train_tokens = tokenize("\n".join(self.splits["train"]))
        print(f"[DataLoader] Training tokens: {len(train_tokens):,}")
        self.vocab.build(train_tokens, self.vocab_size_cap)

    def _encode_splits(self) -> None:
        for split, passages in self.splits.items():
            toks = tokenize("\n".join(passages))
            arr  = np.array(self.vocab.encode_tokens(toks), dtype=np.int32)
            self.encoded[split] = arr
            print(f"[DataLoader] {split:5s} encoded: {len(arr):,} token ids")

    # ── Public API ────────────────────────────────────────────────

    def sample_batch(
        self, split: str = "train"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample a random batch of (X, Y) pairs.

        Returns
        -------
        X : (batch_size, seq_len) — input token indices
        Y : (batch_size, seq_len) — target token indices (X shifted by 1)
        """
        data      = self.encoded[split]
        max_start = len(data) - self.seq_len - 1
        if max_start <= 0:
            raise ValueError(
                f"Split '{split}' is too short for seq_len={self.seq_len}."
            )
        starts = self.np_rng.integers(0, max_start, size=self.batch_size)
        X = np.stack([data[s: s + self.seq_len]     for s in starts])
        Y = np.stack([data[s + 1: s + self.seq_len + 1] for s in starts])
        return X, Y

    def iter_epoch(
        self, split: str = "train", steps_per_epoch: int = 300
    ):
        """Yield (X, Y) batches for one epoch."""
        for _ in range(steps_per_epoch):
            yield self.sample_batch(split)

    def compute_perplexity(
        self,
        model,
        split: str = "val",
        steps: int  = 20,
    ) -> float:
        """
        Estimate perplexity on `split` by averaging cross-entropy
        over `steps` random batches (one sequence at a time).
        """
        total_loss = 0.0
        count      = 0
        for _ in range(steps):
            X, Y = self.sample_batch(split)
            for i in range(len(X)):
                _, loss = model.forward(X[i], Y[i], training=False)
                total_loss += loss
                count      += 1
        avg_loss = total_loss / max(count, 1)
        return float(np.exp(avg_loss))


# ──────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("🔍 Testing DataLoader (word-level) ...\n")
    loader = DataLoader(seq_len=64, batch_size=8, max_files=50)

    print("\n📊 Split sizes:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(loader.splits[split])} passages")

    print(f"\n🔤 Vocabulary size: {loader.vocab.size}")

    sample = loader.splits["train"][0][:120]
    enc    = loader.vocab.encode(sample)
    dec    = loader.vocab.decode(enc)
    print("\n🔁 Encode/Decode:")
    print("  Original:", sample[:60])
    print("  Decoded :", dec[:60])

    X, Y = loader.sample_batch("train")
    print(f"\n📦 Batch shapes — X: {X.shape}  Y: {Y.shape}")
    print("\n✅ DataLoader test complete!")
