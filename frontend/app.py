"""
app.py
------
Streamlit inference interface for the RNN Language Model.

Run with:
    streamlit run app.py

Features:
  - Enter seed text
  - Adjust temperature and top-k
  - Step-by-step or bulk generation
  - Top-5 probability bar chart (UC-08)
  - Attention alpha heatmap (UC-10)
"""

import os
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from src.data_loader import DataLoader
from src.rnnlm       import RNNLM
from src.sampling    import top_k_probs

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR   = "data/raw/topics"
CKPT_PATH  = "checkpoints/best.npz"


# ──────────────────────────────────────────────────────────────────────────────
# Cached resource loading
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_resources():
    loader = DataLoader(DATA_DIR, seq_len=100, batch_size=1)
    model  = RNNLM(
        vocab_size = loader.vocab.size,
        embed_dim  = 64,
        hidden_dim = 256,
        num_layers = 2,
        keep_prob  = 1.0,   # no dropout at inference
    )
    if os.path.exists(CKPT_PATH):
        model.load(CKPT_PATH)
    else:
        st.warning("No checkpoint found — using random weights. Train the model first.")
    return loader, model


# ──────────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_top5(logits: np.ndarray, vocab, temperature: float, top_k: int):
    """Bar chart of top-5 token probabilities (UC-08)."""
    indices, probs = top_k_probs(logits, temperature=temperature, k=5)
    labels = [repr(vocab.idx2char.get(i, "?")) for i in indices]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(labels[::-1], probs[::-1], color="#4C72B0")
    ax.set_xlabel("Probability")
    ax.set_title("Top-5 Next Token Probabilities")
    ax.set_xlim(0, 1)
    for bar, p in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{p:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_attention(alpha: np.ndarray, seed_text: str):
    """Heatmap of attention weights over seed tokens (UC-10)."""
    if alpha is None or len(alpha) == 0:
        st.info("Generate at least one token to view the attention heatmap.")
        return

    # Clip to last 60 characters for readability
    chars   = list(seed_text[-60:])
    weights = alpha[-len(chars):]

    fig, ax = plt.subplots(figsize=(max(8, len(chars) * 0.2), 1.5))
    sns.heatmap(
        weights.reshape(1, -1),
        ax=ax,
        xticklabels=chars,
        yticklabels=["α"],
        cmap="Blues",
        vmin=0, vmax=weights.max() + 1e-9,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Attention Weights over Seed Tokens")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="RNN Poetry Generator", page_icon="📜", layout="wide")
    st.title("📜 RNN Language Model — Poetry Generator")
    st.caption("IT08X97 Artificial Intelligence | Clarence Obini Dinkaa Ebebe | 222086329")

    loader, model = load_resources()
    vocab = loader.vocab

    # ── Sidebar controls ──────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.05,
                                help="0 = greedy; higher = more creative")
        top_k = st.slider("Top-k", 0, 50, 0,
                           help="0 = no truncation (sample from full vocab)")
        n_tokens = st.slider("Tokens to generate", 10, 500, 100, 10)
        st.divider()
        st.markdown("**Model info**")
        st.write(f"Vocab size: `{vocab.size}`")
        st.write(f"Checkpoint: `{CKPT_PATH}`")

    # ── Main area ─────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        seed_text = st.text_area(
            "Seed text",
            value="Shall I compare thee to a summer's day?\n",
            height=120,
        )

        generate_btn = st.button("✨ Generate", type="primary")

    with col2:
        st.markdown("### How it works")
        st.markdown(
            """
            1. Your seed text is tokenised character-by-character.
            2. The GRU processes each character, building a hidden state.
            3. Bahdanau attention weights past hidden states.
            4. The output projection produces a probability over the vocabulary.
            5. A token is sampled and appended.
            """
        )

    # ── Generation ────────────────────────────────────────────────
    if generate_btn:
        # Handle unknown characters
        unk = vocab.char2idx.get("<UNK>", 1)
        known_seed = "".join(ch if ch in vocab.char2idx else "?" for ch in seed_text)
        if known_seed != seed_text:
            st.warning("Some characters in your seed were replaced with '?' (out-of-vocabulary).")

        seed_tokens = np.array(
            [vocab.char2idx.get(ch, unk) for ch in seed_text],
            dtype=np.int32,
        )

        with st.spinner("Generating…"):
            completion, alpha = model.generate(
                seed_tokens=seed_tokens,
                vocab=vocab,
                n=n_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        st.subheader("Generated Text")
        st.text(seed_text + completion)

        st.divider()

        # ── Probability chart ─────────────────────────────────────
        st.subheader("Top-5 Token Probabilities (last step)")
        # Re-run a forward pass to get final logits for display
        probs, _ = model.forward(seed_tokens, targets=None, training=False)
        logits   = np.log(probs[-1] + 1e-12)
        plot_top5(logits, vocab, temperature, top_k)

        st.divider()

        # ── Attention heatmap ─────────────────────────────────────
        with st.expander("🔍 Attention Heatmap", expanded=True):
            plot_attention(alpha, seed_text)

    else:
        st.info("Enter a seed text and click **✨ Generate** to start.")


if __name__ == "__main__":
    main()
