"""
app.py
------
Streamlit inference interface for the Transformer Language Model.

Run with:
    streamlit run frontend/app.py

Handles two checkpoint formats transparently:
  • Colab/notebook format — keys like 'emb', 'b0_ff1', 'b0_qkv' (no metadata)
  • train.py format      — same keys + explicit '_vocab_size' etc. scalars
"""

import sys
import os
import re
from pathlib import Path

# Anchor to project root regardless of which directory streamlit is run from
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.data_loader   import DataLoader, tokenize
from src.transformerlm import TransformerLM
from src.sampling      import sample_token, top_k_probs

# Default checkpoint — always relative to project root, never cwd
_DEFAULT_CKPT = str(_PROJECT_ROOT / "checkpoints" / "best_transformer")


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Verse Engine",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS — luxury dark theme
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Montserrat:wght@300;400;500&display=swap');

:root {
    --gold:     #C9A84C;
    --gold-dim: #8B6914;
    --cream:    #F2EAD3;
    --bg:       #0C0B09;
    --bg2:      #141210;
    --bg3:      #1C1A16;
    --border:   rgba(201,168,76,0.18);
    --text:     #D4C5A0;
    --muted:    #7A6F5A;
}

html, body, .stApp { background-color: var(--bg) !important; color: var(--text) !important;
    font-family: 'Montserrat', sans-serif; font-weight: 300; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1200px; }
h1, h2, h3 { font-family: 'Cormorant Garamond', serif !important;
    color: var(--cream) !important; font-weight: 300 !important; letter-spacing: 0.04em; }
section[data-testid="stSidebar"] { background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important; }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
textarea { background: var(--bg3) !important; border: 1px solid var(--border) !important;
    color: var(--cream) !important; font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.1rem !important; border-radius: 2px !important; }
textarea:focus { border-color: var(--gold) !important; }
.stButton > button { background: transparent !important; border: 1px solid var(--gold) !important;
    color: var(--gold) !important; font-family: 'Montserrat', sans-serif !important;
    font-size: 0.72rem !important; font-weight: 500 !important; letter-spacing: 0.18em !important;
    text-transform: uppercase !important; padding: 0.65rem 2.2rem !important;
    border-radius: 0 !important; transition: all 0.25s ease !important; }
.stButton > button:hover { background: var(--gold) !important; color: var(--bg) !important; }
hr { border-color: var(--border) !important; }
.stAlert { background: var(--bg3) !important; border-color: var(--border) !important;
    border-radius: 0 !important; }
div[data-testid="stMetric"] { background: var(--bg3); border: 1px solid var(--border);
    padding: 1rem 1.4rem; }
div[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.72rem !important;
    letter-spacing: 0.12em; text-transform: uppercase; }
div[data-testid="stMetricValue"] { color: var(--gold) !important;
    font-family: 'Cormorant Garamond', serif !important; font-size: 2rem !important; }
details { border: 1px solid var(--border) !important; background: var(--bg2) !important;
    border-radius: 0 !important; }
summary { color: var(--gold) !important; font-size: 0.78rem !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important; }
.stRadio label { color: var(--text) !important; font-size: 0.85rem !important; }
div[data-baseweb="select"] > div { background: var(--bg3) !important;
    border-color: var(--border) !important; border-radius: 0 !important;
    color: var(--cream) !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Masthead
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 0.5rem;">
    <div style="font-size:0.72rem; letter-spacing:0.3em; color:#8B6914;
                text-transform:uppercase; margin-bottom:0.5rem;">
        Transformer Language Model
    </div>
    <h1 style="font-size:3.2rem; margin:0; letter-spacing:0.06em;">Verse Engine</h1>
    <div style="width:60px; height:1px; background:#C9A84C; margin:1rem auto;"></div>
    <div style="font-size:0.75rem; letter-spacing:0.2em; color:#7A6F5A; text-transform:uppercase;">
        Token-by-Token Generation &nbsp;·&nbsp; Probability Visualisation &nbsp;·&nbsp; Attention Maps
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers — handle both Colab and train.py formats
# ──────────────────────────────────────────────────────────────────────────────

def _norm_path(path: str) -> str:
    """Ensure path ends with .npz exactly once."""
    return str(path).removesuffix(".npz") + ".npz"


def _infer_hparams(path: str) -> dict | None:
    """
    Read hyperparams from a checkpoint.

    Supports two formats:
      train.py format  — explicit scalar keys _vocab_size, _embed_dim, etc.
      Colab format     — no metadata; infer from weight shapes using key
                         patterns: 'emb' (V, D), 'b0_ff1' (ffn_dim, D).

    Returns None only if the file does not exist.
    """
    fpath = _norm_path(path)
    if not os.path.exists(fpath):
        return None

    data = np.load(fpath, allow_pickle=False)
    keys = list(data.keys())

    # ── train.py format: explicit metadata ───────────────────────
    if "_vocab_size" in data:
        return {
            "vocab_size" : int(data["_vocab_size"]),
            "embed_dim"  : int(data["_embed_dim"]),
            "num_heads"  : int(data["_num_heads"]),
            "num_layers" : int(data["_num_layers"]),
            "ffn_dim"    : int(data["_ffn_dim"]),
            "max_seq_len": int(data.get("_max_seq_len", 512)),
            "fmt"        : "train",
        }

    # ── Colab/notebook format: shape inference ────────────────────
    # Token embedding key is 'emb' (shape: vocab_size × d_model)
    if "emb" not in data:
        return None

    vocab_size, embed_dim = int(data["emb"].shape[0]), int(data["emb"].shape[1])

    # Count transformer blocks by highest block index in key names
    block_ids = {int(m.group(1))
                 for k in keys
                 for m in [re.match(r"^b(\d+)_", k)] if m}
    num_layers = max(block_ids) + 1 if block_ids else 4

    # FFN inner dim from b0_ff1 weight: shape (ffn_dim, d_model)
    ffn_dim = int(data["b0_ff1"].shape[0]) if "b0_ff1" in data else embed_dim * 4

    # num_heads: d_k is conventionally 32; embed_dim // 32 gives 8H for d=256
    num_heads = max(1, embed_dim // 32)

    return {
        "vocab_size" : vocab_size,
        "embed_dim"  : embed_dim,
        "num_heads"  : num_heads,
        "num_layers" : num_layers,
        "ffn_dim"    : ffn_dim,
        "max_seq_len": 512,
        "fmt"        : "colab",
    }


def _load_colab_weights(model: TransformerLM, path: str) -> None:
    """
    Load a Colab/notebook checkpoint into a TransformerLM instance.

    Colab key mapping (from the GPU notebook):
        emb          → embedding.E
        b{i}_qkv     → block i W_qkv
        b{i}_qkv_b   → block i b_qkv
        b{i}_wo      → block i W_o
        b{i}_wo_b    → block i b_o
        b{i}_ln1_g   → block i ln1_gamma
        b{i}_ln1_b   → block i ln1_beta
        b{i}_ln2_g   → block i ln2_gamma
        b{i}_ln2_b   → block i ln2_beta
        b{i}_ff1     → block i W1
        b{i}_ff1_b   → block i b1
        b{i}_ff2     → block i W2
        b{i}_ff2_b   → block i b2
        final_ln_g   → final_ln_gamma
        final_ln_b   → final_ln_beta
    """
    fpath = _norm_path(path)
    data  = np.load(fpath, allow_pickle=False)

    model.embedding.E = data["emb"].copy()

    for i, block in enumerate(model.blocks):
        p = f"b{i}_"
        block.W_qkv     = data[f"{p}qkv"].copy()
        block.b_qkv     = data[f"{p}qkv_b"].copy()
        block.W_o       = data[f"{p}wo"].copy()
        block.b_o       = data[f"{p}wo_b"].copy()
        block.ln1_gamma = data[f"{p}ln1_g"].copy()
        block.ln1_beta  = data[f"{p}ln1_b"].copy()
        block.ln2_gamma = data[f"{p}ln2_g"].copy()
        block.ln2_beta  = data[f"{p}ln2_b"].copy()
        block.W1        = data[f"{p}ff1"].copy()
        block.b1        = data[f"{p}ff1_b"].copy()
        block.W2        = data[f"{p}ff2"].copy()
        block.b2        = data[f"{p}ff2_b"].copy()
        block.zero_grad()

    # Final layer norm (Colab uses _g/_b suffix)
    if "final_ln_g" in data:
        model.final_ln_gamma = data["final_ln_g"].copy()
        model.final_ln_beta  = data["final_ln_b"].copy()
    elif "final_ln_gamma" in data:
        model.final_ln_gamma = data["final_ln_gamma"].copy()
        model.final_ln_beta  = data["final_ln_beta"].copy()

    print(f"[app] Loaded Colab checkpoint ← {fpath}")


# ──────────────────────────────────────────────────────────────────────────────
# Resource loading (cached — only runs once per ckpt_path)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_resources(ckpt_path: str):
    hp = _infer_hparams(ckpt_path)

    if hp is None:
        st.warning(
            f"No checkpoint found at `{_norm_path(ckpt_path)}`. "
            "Displaying random weights — train the model first."
        )
        loader = DataLoader(seq_len=64, batch_size=1)
        model  = TransformerLM(
            vocab_size=loader.vocab.size, embed_dim=256,
            num_heads=8, num_layers=4, ffn_dim=1024, dropout_rate=0.0,
        )
        return loader, model

    # Build DataLoader with vocab capped to match checkpoint vocab size.
    # Subtract 5 for the special tokens already counted inside the checkpoint.
    vocab_cap = max(hp["vocab_size"] - 5, 100)
    loader = DataLoader(
        seq_len        = min(hp["max_seq_len"], 64),
        batch_size     = 1,
        vocab_size_cap = vocab_cap,
    )

    if loader.vocab.size != hp["vocab_size"]:
        st.warning(
            f"Vocab size mismatch: checkpoint={hp['vocab_size']}, "
            f"loader built={loader.vocab.size}. "
            "Words outside the shared vocabulary will decode as '?'."
        )

    model = TransformerLM(
        vocab_size    = hp["vocab_size"],
        embed_dim     = hp["embed_dim"],
        num_heads     = hp["num_heads"],
        num_layers    = hp["num_layers"],
        ffn_dim       = hp["ffn_dim"],
        dropout_rate  = 0.0,
        max_seq_len   = hp["max_seq_len"],
    )

    if hp["fmt"] == "colab":
        _load_colab_weights(model, ckpt_path)
    else:
        model.load(ckpt_path)

    return loader, model


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def _dark_theme(fig, ax):
    fig.patch.set_facecolor("#141210")
    ax.set_facecolor("#1C1A16")
    ax.tick_params(colors="#7A6F5A", labelsize=8)
    ax.xaxis.label.set_color("#7A6F5A")
    ax.yaxis.label.set_color("#7A6F5A")
    ax.title.set_color("#D4C5A0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2E2B24")


def plot_candidates(logits: np.ndarray, vocab, temperature: float,
                    chosen_idx: int, k: int = 8):
    indices, probs = top_k_probs(logits, temperature=temperature, k=k)
    labels      = [vocab.idx2word.get(i, "?") for i in indices]
    colors      = ["#C9A84C" if i == chosen_idx else "#2E2B24" for i in indices]
    edge_colors = ["#C9A84C" if i == chosen_idx else "#3E3A30" for i in indices]

    fig, ax = plt.subplots(figsize=(7, max(2.5, k * 0.38)))
    bars = ax.barh(labels[::-1], probs[::-1],
                   color=colors[::-1], edgecolor=edge_colors[::-1],
                   linewidth=0.8, height=0.65)
    for bar, p, idx in zip(bars, probs[::-1], indices[::-1]):
        clr = "#C9A84C" if idx == chosen_idx else "#7A6F5A"
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{p:.3f}", va="center", fontsize=8, color=clr)
    ax.set_xlim(0, min(1.0, probs.max() * 1.35))
    ax.set_xlabel("Probability", fontsize=8)
    ax.set_title("Next-token candidates", fontsize=10, pad=10,
                 fontfamily="serif", fontstyle="italic")
    ax.legend(
        handles=[mpatches.Patch(color="#C9A84C", label="Chosen"),
                 mpatches.Patch(color="#2E2B24", ec="#3E3A30", lw=0.8, label="Alternatives")],
        fontsize=7, framealpha=0, labelcolor="#7A6F5A", loc="lower right",
    )
    _dark_theme(fig, ax)
    plt.tight_layout(pad=0.8)
    return fig


def plot_attention_heads(model: TransformerLM, token_ids: np.ndarray,
                         vocab, layer_idx: int = 0, head_idx: int = 0,
                         max_tokens: int = 24):
    """Extract and plot the (T, T) attention matrix for one head/layer."""
    import math
    from src.transformer_block import build_causal_mask, _softmax

    T   = min(len(token_ids), max_tokens)
    ids = token_ids[-T:]
    D   = model.embedding.E.shape[1]
    nH  = model.num_heads
    d_k = D // nH

    if layer_idx >= model.num_layers:
        st.warning(f"Layer {layer_idx} doesn't exist (model has {model.num_layers} layers).")
        return None
    if head_idx >= nH:
        st.warning(f"Head {head_idx} doesn't exist (model has {nH} heads).")
        return None

    # x = embedding + PE (exact for layer 0; approximate for deeper layers)
    x    = model.embedding.E[ids].astype(np.float32) + model.pos_enc.get(T)
    mask = build_causal_mask(T)

    if layer_idx > 0:
        st.caption(
            f"ℹ️ Attention map for layer {layer_idx} is approximate "
            "(uses embedding-level hidden state). Layer 0 is always exact."
        )
        for i in range(layer_idx):
            x = model.blocks[i].forward(x, mask=mask, training=False)

    # Extract QK attention from the target block
    block = model.blocks[layer_idx]
    mean  = x.mean(-1, keepdims=True)
    var   = ((x - mean) ** 2).mean(-1, keepdims=True)
    xn    = block.ln1_gamma * (x - mean) / np.sqrt(var + 1e-5) + block.ln1_beta

    qkv     = xn @ block.W_qkv.T + block.b_qkv
    Q, K, _ = np.split(qkv, 3, axis=-1)
    Q = Q.reshape(T, nH, d_k).transpose(1, 0, 2)
    K = K.reshape(T, nH, d_k).transpose(1, 0, 2)

    scores    = Q @ K.transpose(0, 2, 1) / math.sqrt(d_k) + mask
    attn      = _softmax(scores, axis=-1)
    head_attn = attn[head_idx]

    labels = [vocab.idx2word.get(int(i), "?") for i in ids]
    fig, ax = plt.subplots(figsize=(max(6, T * 0.45), max(5, T * 0.45)))
    im = ax.imshow(head_attn, cmap="YlOrBr", aspect="auto", vmin=0)
    ax.set_xticks(range(T)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(T)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Attention — Layer {layer_idx}, Head {head_idx}",
                 fontsize=9, pad=8, fontfamily="serif", fontstyle="italic")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
        labelsize=6, colors="#7A6F5A")
    _dark_theme(fig, ax)
    plt.tight_layout(pad=0.5)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.25em; color:#8B6914;
                text-transform:uppercase; margin-bottom:1.2rem;">Configuration</div>
    """, unsafe_allow_html=True)

    ckpt_input = st.text_input(
        "Checkpoint path",
        value=_DEFAULT_CKPT,
        help="Path to checkpoint without .npz extension.",
    )
    ckpt_path = ckpt_input.strip() or _DEFAULT_CKPT

    st.markdown("<hr>", unsafe_allow_html=True)
    temperature     = st.slider("Temperature",        0.0, 2.0, 0.85, 0.05)
    top_k           = st.slider("Top-k sampling",     0,   50,  50)
    n_tokens        = st.slider("Tokens to generate", 1,   300, 60)
    show_candidates = st.slider("Candidates to show", 3,   15,  8)
    step_by_step    = st.checkbox("Step-by-step mode", value=False)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.12em; color:#7A6F5A;
                text-transform:uppercase;">Attention Visualisation</div>
    """, unsafe_allow_html=True)
    show_attn  = st.checkbox("Show attention map", value=True)
    attn_layer = st.number_input("Layer index", min_value=0, max_value=15, value=0, step=1)
    attn_head  = st.number_input("Head index",  min_value=0, max_value=31, value=0, step=1)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.12em; color:#7A6F5A;
                text-transform:uppercase;">About</div>
    <div style="font-size:0.78rem; color:#5A5244; margin-top:0.5rem; line-height:1.7;">
        GPT-style Transformer<br>Pre-LN · Fused QKV · GELU FFN<br>
        Weight-tied output · NumPy-only<br>IT08X97 Artificial Intelligence
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────────────────

loader, model = load_resources(ckpt_path)
vocab = loader.vocab
hp    = _infer_hparams(ckpt_path)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Vocabulary",  f"{vocab.size:,}")
c2.metric("Architecture",
          f"{hp['num_layers']}L · {hp['num_heads']}H · d{hp['embed_dim']}" if hp else "—")
c3.metric("Parameters",
          f"{sum(v.size for v in model.params().values()):,}" if hp else "—")
c4.metric("Format", hp.get("fmt", "?").upper() if hp else "—")

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Input
# ──────────────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.22em; color:#8B6914;
                text-transform:uppercase; margin-bottom:0.4rem;">Seed Text</div>
    """, unsafe_allow_html=True)
    seed_text    = st.text_area(
        label="seed", label_visibility="collapsed",
        value="Shall I compare thee to a summer's day?\n",
        height=140,
    )
    generate_btn = st.button("✦  Generate")

with col_right:
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.22em; color:#8B6914;
                text-transform:uppercase; margin-bottom:0.8rem;">How It Works</div>
    <div style="font-size:0.82rem; color:#7A6F5A; line-height:1.9;">
        <span style="color:#C9A84C;">01</span> &nbsp; Seed text is word-tokenised<br>
        <span style="color:#C9A84C;">02</span> &nbsp; Embedding + positional encoding<br>
        <span style="color:#C9A84C;">03</span> &nbsp; N × (Causal self-attention + FFN)<br>
        <span style="color:#C9A84C;">04</span> &nbsp; Weight-tied projection → vocab<br>
        <span style="color:#C9A84C;">05</span> &nbsp; Token sampled, appended, repeated
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

if generate_btn:
    unk_id  = vocab.word2idx.get("<UNK>", 1)
    max_ctx = model.max_seq_len

    seed_word_tokens = tokenize(seed_text.strip())
    seed_ids         = [vocab.word2idx.get(w, unk_id) for w in seed_word_tokens]

    if not seed_ids:
        st.warning("Please enter some seed text.")
        st.stop()

    generated_ids: list[int]   = []
    step_data:     list[tuple] = []
    tokens_so_far              = list(seed_ids)
    last_logits                = None

    progress = st.progress(0, text="Generating…")

    for step in range(n_tokens):
        ctx = tokens_so_far[-max_ctx:]
        inp = np.array(ctx, dtype=np.int32)

        logits, _   = model.forward(inp, targets=None, training=False)
        last_logits = logits[-1]

        next_tok = sample_token(last_logits, temperature, top_k)
        generated_ids.append(next_tok)
        tokens_so_far.append(next_tok)

        if step_by_step:
            step_data.append((next_tok, last_logits.copy()))

        progress.progress((step + 1) / n_tokens,
                          text=f"Generating… {step+1}/{n_tokens}")

    progress.empty()

    # ── Sequence display ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.22em; color:#8B6914;
                text-transform:uppercase; margin-bottom:0.6rem;">Generated Sequence</div>
    """, unsafe_allow_html=True)

    seed_html = " ".join(
        f'<span style="color:#EDE5CC; opacity:0.6; font-family:Cormorant Garamond,serif;'
        f' font-size:1.05rem">{w}</span>'
        for w in seed_word_tokens
    )
    gen_html = " ".join(
        f'<span style="display:inline-block; background:linear-gradient(135deg,#2A2310,#1A1A0E);'
        f' border:1px solid #C9A84C; color:#C9A84C; font-family:Cormorant Garamond,serif;'
        f' font-size:1.05rem; padding:0.1rem 0.4rem; margin:0.1rem; border-radius:1px">'
        f'{vocab.idx2word.get(i, "?")}</span>'
        for i in generated_ids
    )
    st.markdown(
        f'<div style="background:#141210; border:1px solid rgba(201,168,76,0.18);'
        f' padding:1.4rem 1.6rem; border-radius:1px; line-height:2.4;">'
        f'{seed_html} {gen_html}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Last-step candidate chart ─────────────────────────────────────────────
    if last_logits is not None:
        st.markdown("""
        <div style="font-size:0.65rem; letter-spacing:0.22em; color:#8B6914;
                    text-transform:uppercase; margin-bottom:0.6rem;">Last Step — Token Candidates</div>
        """, unsafe_allow_html=True)
        disp_logits = step_data[-1][1] if step_data else last_logits
        fig_cand = plot_candidates(disp_logits, vocab, temperature,
                                   generated_ids[-1], k=show_candidates)
        st.pyplot(fig_cand)
        plt.close(fig_cand)

    # ── Step-by-step breakdown ────────────────────────────────────────────────
    if step_by_step and step_data:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.65rem; letter-spacing:0.22em; color:#8B6914;
                    text-transform:uppercase; margin-bottom:0.8rem;">Step-by-Step Selection</div>
        """, unsafe_allow_html=True)
        shown        = step_data[:12]
        cols_per_row = 3
        for row_start in range(0, len(shown), cols_per_row):
            row  = shown[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col_i, (chosen_idx, logits_s) in enumerate(row):
                step_num     = row_start + col_i + 1
                chosen_label = vocab.idx2word.get(chosen_idx, "?")
                with cols[col_i]:
                    st.markdown(
                        f'<div style="font-size:0.65rem; color:#8B6914; '
                        f'letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.3rem;">'
                        f'Step {step_num} → <span style="color:#C9A84C">{chosen_label}</span></div>',
                        unsafe_allow_html=True,
                    )
                    fig = plot_candidates(logits_s, vocab, temperature,
                                         chosen_idx, k=min(5, show_candidates))
                    st.pyplot(fig)
                    plt.close(fig)
        if len(step_data) > 12:
            st.caption(f"Showing first 12 of {len(step_data)} steps.")

    # ── Attention heatmap ─────────────────────────────────────────────────────
    if show_attn:
        with st.expander("Attention Weights", expanded=True):
            all_ids  = np.array(seed_ids + generated_ids, dtype=np.int32)
            fig_attn = plot_attention_heads(
                model, all_ids, vocab,
                layer_idx=int(attn_layer),
                head_idx=int(attn_head),
                max_tokens=24,
            )
            if fig_attn:
                st.pyplot(fig_attn)
                plt.close(fig_attn)

    # ── Model details ─────────────────────────────────────────────────────────
    with st.expander("Model Details", expanded=False):
        if hp:
            n_params = sum(v.size for v in model.params().values())
            st.markdown(f"""
| Hyperparameter | Value |
|---|---|
| Checkpoint format | {hp.get('fmt','unknown').upper()} |
| Vocab size | {hp['vocab_size']:,} |
| d_model | {hp['embed_dim']} |
| Num heads | {hp['num_heads']} |
| Num layers | {hp['num_layers']} |
| FFN dim | {hp['ffn_dim']} |
| Max seq len | {hp['max_seq_len']} |
| Total parameters | {n_params:,} |
""")
        else:
            st.info("No checkpoint metadata available.")

else:
    st.markdown("""
    <div style="text-align:center; padding: 3.5rem 0; color:#3E3A30;">
        <div style="font-size:2.5rem; margin-bottom:1rem; opacity:0.4;">✦</div>
        <div style="font-family:'Cormorant Garamond', serif; font-size:1.3rem;
                    font-style:italic; color:#5A5244; letter-spacing:0.05em;">
            Enter a seed and let the model continue the verse.
        </div>
    </div>
    """, unsafe_allow_html=True)