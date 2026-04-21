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

_DEFAULT_CKPT = str(_PROJECT_ROOT / "checkpoints" / "best_transformer")


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Verse Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS — Royal Glassmorphic Minimalism
# Palette: Deep indigo night · Ivory · Pale violet · Warm gold accent
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400;1,500&family=Jost:wght@200;300;400;500&display=swap');

:root {
    --ink:        #0D0B1A;
    --ink2:       #13102A;
    --ink3:       #1A1735;
    --violet:     #7B6FD0;
    --violet-dim: #4A4488;
    --violet-pale:#C5BFF5;
    --ivory:      #F0EDE6;
    --ivory-dim:  #C8C3B8;
    --gold:       #D4A843;
    --gold-soft:  #F0C96A;
    --muted:      #6B6590;
    --border:     rgba(123,111,208,0.22);
    --glass-bg:   rgba(26,23,53,0.55);
    --glass-bg2:  rgba(20,17,42,0.72);
    --glass-border: rgba(197,191,245,0.15);
    --shadow:     0 8px 40px rgba(0,0,0,0.45);
}

/* ── Base ── */
html, body, .stApp {
    background: var(--ink) !important;
    color: var(--ivory-dim) !important;
    font-family: 'Jost', sans-serif;
    font-weight: 300;
}
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(75,60,180,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 85% 80%, rgba(100,60,160,0.14) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(212,168,67,0.04) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1180px;
    position: relative;
    z-index: 1;
}

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'EB Garamond', serif !important;
    color: var(--ivory) !important;
    font-weight: 400 !important;
    letter-spacing: 0.03em;
}

/* ── Sidebar glass ── */
section[data-testid="stSidebar"] {
    background: var(--glass-bg2) !important;
    border-right: 1px solid var(--glass-border) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
}
section[data-testid="stSidebar"] * { color: var(--ivory-dim) !important; }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { }

/* ── Glass card mixin — applied via inline style divs ── */

/* ── Inputs ── */
textarea {
    background: rgba(26,23,53,0.6) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--ivory) !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 1.15rem !important;
    border-radius: 8px !important;
    backdrop-filter: blur(12px) !important;
    transition: border-color 0.3s ease !important;
}
textarea:focus {
    border-color: rgba(197,191,245,0.5) !important;
    box-shadow: 0 0 0 3px rgba(123,111,208,0.12) !important;
}

/* ── Button ── */
.stButton > button {
    background: rgba(123,111,208,0.12) !important;
    border: 1px solid rgba(197,191,245,0.35) !important;
    color: var(--violet-pale) !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2.4rem !important;
    border-radius: 40px !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover {
    background: rgba(123,111,208,0.28) !important;
    border-color: rgba(197,191,245,0.6) !important;
    color: var(--ivory) !important;
    box-shadow: 0 4px 24px rgba(123,111,208,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
hr { border-color: var(--glass-border) !important; }

/* ── Alerts ── */
.stAlert {
    background: var(--glass-bg) !important;
    border-color: var(--glass-border) !important;
    border-radius: 8px !important;
    backdrop-filter: blur(12px) !important;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    padding: 1rem 1.2rem !important;
    border-radius: 10px !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    box-shadow: var(--shadow) !important;
}
div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    font-family: 'Jost', sans-serif !important;
}
div[data-testid="stMetricValue"] {
    color: var(--violet-pale) !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 1.75rem !important;
}

/* ── Expander ── */
details {
    border: 1px solid var(--glass-border) !important;
    background: var(--glass-bg) !important;
    border-radius: 8px !important;
    backdrop-filter: blur(12px) !important;
}
summary {
    color: var(--violet-pale) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    font-family: 'Jost', sans-serif !important;
}

/* ── Select / Radio ── */
.stRadio label { color: var(--ivory-dim) !important; font-size: 0.85rem !important; }
div[data-baseweb="select"] > div {
    background: var(--glass-bg) !important;
    border-color: var(--glass-border) !important;
    border-radius: 8px !important;
    color: var(--ivory) !important;
    backdrop-filter: blur(12px) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--violet-dim), var(--violet-pale)) !important;
}

/* ── Slider track colour tweak ── */
div[data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: var(--violet-pale) !important;
}

/* ── Table in expander ── */
table { border-collapse: collapse; width: 100%; }
th, td {
    border: 1px solid var(--glass-border) !important;
    padding: 0.45rem 0.8rem !important;
    font-size: 0.82rem !important;
    color: var(--ivory-dim) !important;
}
th { color: var(--muted) !important; font-weight: 400 !important; letter-spacing: 0.1em !important; }

/* ── Caption ── */
.stCaption { color: var(--muted) !important; font-size: 0.75rem !important; }

/* ── Poem stanza display ── */
.poem-container {
    background: rgba(20,17,42,0.62);
    border: 1px solid rgba(197,191,245,0.13);
    border-radius: 12px;
    padding: 2.4rem 3rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 16px 64px rgba(0,0,0,0.5), inset 0 1px 0 rgba(197,191,245,0.08);
    font-family: 'EB Garamond', serif;
    font-size: 1.15rem;
    line-height: 1.95;
    letter-spacing: 0.01em;
    color: var(--ivory);
}
.poem-line {
    display: block;
    min-height: 1.95em;
}
.poem-line.seed-line { color: rgba(240,237,230,0.52); }
.poem-line.gen-line  { color: var(--ivory); }

.poem-word-gen {
    display: inline-block;
    background: rgba(123,111,208,0.18);
    border: 1px solid rgba(197,191,245,0.25);
    border-radius: 3px;
    padding: 0 0.28em;
    margin: 0 0.05em;
    color: var(--violet-pale);
    font-style: italic;
    backdrop-filter: blur(4px);
    transition: background 0.2s;
}
.poem-stanza-break { display: block; height: 1.1em; }

/* ── Ornamental rule ── */
.ornament {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.6rem 0;
    opacity: 0.4;
}
.ornament::before, .ornament::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--violet-pale), transparent);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Masthead
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding: 1.8rem 0 0.6rem;">
    <div style="font-size:0.62rem; letter-spacing:0.38em; color:#7B6FD0;
                text-transform:uppercase; margin-bottom:0.7rem; font-family:'Jost',sans-serif;">
        Transformer Language Model
    </div>
    <h1 style="font-size:3.4rem; margin:0; letter-spacing:0.05em; font-weight:400;
               background: linear-gradient(135deg, #F0EDE6 0%, #C5BFF5 55%, #D4A843 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               background-clip: text;">
        Verse Engine
    </h1>
    <div style="display:flex; align-items:center; gap:1rem; justify-content:center; margin:1rem auto; opacity:0.35; max-width:260px;">
        <div style="flex:1; height:1px; background:linear-gradient(90deg,transparent,#C5BFF5);"></div>
        <span style="color:#C5BFF5; font-size:0.7rem;">◈</span>
        <div style="flex:1; height:1px; background:linear-gradient(90deg,#C5BFF5,transparent);"></div>
    </div>
    <div style="font-size:0.68rem; letter-spacing:0.2em; color:#6B6590; text-transform:uppercase;
                font-family:'Jost',sans-serif;">
        Token Generation &ensp;·&ensp; Probability Visualisation &ensp;·&ensp; Attention Maps
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def _norm_path(path: str) -> str:
    return str(path).removesuffix(".npz") + ".npz"


def _infer_hparams(path: str) -> dict | None:
    fpath = _norm_path(path)
    if not os.path.exists(fpath):
        return None

    data = np.load(fpath, allow_pickle=False)
    keys = list(data.keys())

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

    if "emb" not in data:
        return None

    vocab_size, embed_dim = int(data["emb"].shape[0]), int(data["emb"].shape[1])
    block_ids = {int(m.group(1))
                 for k in keys
                 for m in [re.match(r"^b(\d+)_", k)] if m}
    num_layers = max(block_ids) + 1 if block_ids else 4
    ffn_dim = int(data["b0_ff1"].shape[0]) if "b0_ff1" in data else embed_dim * 4
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

    if "final_ln_g" in data:
        model.final_ln_gamma = data["final_ln_g"].copy()
        model.final_ln_beta  = data["final_ln_b"].copy()
    elif "final_ln_gamma" in data:
        model.final_ln_gamma = data["final_ln_gamma"].copy()
        model.final_ln_beta  = data["final_ln_beta"].copy()

    print(f"[app] Loaded Colab checkpoint ← {fpath}")


# ──────────────────────────────────────────────────────────────────────────────
# Resource loading
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
# Poem formatter
# ──────────────────────────────────────────────────────────────────────────────

def _words_to_poem_html(seed_words: list[str], gen_words: list[str],
                        words_per_line: int = 8) -> str:
    """
    Render seed + generated words as a poem:
    - Seed words: muted, plain
    - Generated words: violet glass pill highlights
    - Line breaks every ~words_per_line tokens
    - Double-break (stanza) every 4 lines
    """
    all_words = (
        [("seed", w) for w in seed_words] +
        [("gen",  w) for w in gen_words]
    )

    lines_html = []
    current_line: list[str] = []
    word_count = 0
    line_count = 0

    def _flush_line():
        nonlocal line_count
        if current_line:
            line_cls = "gen-line" if any("poem-word-gen" in c for c in current_line) else "seed-line"
            lines_html.append(
                f'<span class="poem-line {line_cls}">{" ".join(current_line)}</span>'
            )
            line_count += 1
            if line_count % 4 == 0:
                lines_html.append('<span class="poem-stanza-break"></span>')
            current_line.clear()

    for kind, word in all_words:
        if kind == "seed":
            current_line.append(f'<span class="poem-word-seed">{word}</span>')
        else:
            current_line.append(f'<span class="poem-word-gen">{word}</span>')
        word_count += 1

        # Break on punctuation endings or at line length
        if word_count % words_per_line == 0 or word.endswith((',', '.', '?', '!', ';', ':')):
            _flush_line()

    if current_line:
        _flush_line()

    inner = "\n".join(lines_html)
    return f'<div class="poem-container">{inner}</div>'


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers — restyled for glass theme
# ──────────────────────────────────────────────────────────────────────────────

_PLOT_BG  = "#0D0B1A"
_PLOT_BG2 = "#13102A"

def _glass_theme(fig, ax):
    fig.patch.set_facecolor(_PLOT_BG)
    ax.set_facecolor(_PLOT_BG2)
    ax.tick_params(colors="#6B6590", labelsize=8)
    ax.xaxis.label.set_color("#6B6590")
    ax.yaxis.label.set_color("#6B6590")
    ax.title.set_color("#C5BFF5")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1A1735")


def plot_candidates(logits: np.ndarray, vocab, temperature: float,
                    chosen_idx: int, k: int = 8):
    indices, probs = top_k_probs(logits, temperature=temperature, k=k)
    labels      = [vocab.idx2word.get(i, "?") for i in indices]
    colors      = ["#7B6FD0" if i == chosen_idx else "#1A1735" for i in indices]
    edge_colors = ["#C5BFF5" if i == chosen_idx else "#2A2650" for i in indices]

    fig, ax = plt.subplots(figsize=(7, max(2.5, k * 0.38)))
    bars = ax.barh(labels[::-1], probs[::-1],
                   color=colors[::-1], edgecolor=edge_colors[::-1],
                   linewidth=0.8, height=0.62)
    for bar, p, idx in zip(bars, probs[::-1], indices[::-1]):
        clr = "#C5BFF5" if idx == chosen_idx else "#6B6590"
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{p:.3f}", va="center", fontsize=8, color=clr)
    ax.set_xlim(0, min(1.0, probs.max() * 1.35))
    ax.set_xlabel("Probability", fontsize=8)
    ax.set_title("Next-token candidates", fontsize=10, pad=10,
                 fontfamily="serif", fontstyle="italic")
    ax.legend(
        handles=[mpatches.Patch(color="#7B6FD0",  label="Chosen"),
                 mpatches.Patch(color="#1A1735", ec="#2A2650", lw=0.8, label="Alternatives")],
        fontsize=7, framealpha=0, labelcolor="#6B6590", loc="lower right",
    )
    _glass_theme(fig, ax)
    plt.tight_layout(pad=0.8)
    return fig


def plot_attention_heads(model: TransformerLM, token_ids: np.ndarray,
                         vocab, layer_idx: int = 0, head_idx: int = 0,
                         max_tokens: int = 24):
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

    x    = model.embedding.E[ids].astype(np.float32) + model.pos_enc.get(T)
    mask = build_causal_mask(T)

    if layer_idx > 0:
        st.caption(
            f"ℹ️ Attention map for layer {layer_idx} is approximate "
            "(uses embedding-level hidden state). Layer 0 is always exact."
        )
        for i in range(layer_idx):
            x = model.blocks[i].forward(x, mask=mask, training=False)

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
    # Custom violet-indigo colormap
    from matplotlib.colors import LinearSegmentedColormap
    royal_cmap = LinearSegmentedColormap.from_list(
        "royal", ["#0D0B1A", "#2A2260", "#7B6FD0", "#C5BFF5", "#F0EDE6"]
    )
    im = ax.imshow(head_attn, cmap=royal_cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(T)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(T)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Attention  ·  Layer {layer_idx}  ·  Head {head_idx}",
                 fontsize=9, pad=8, fontfamily="serif", fontstyle="italic")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6, colors="#6B6590")
    _glass_theme(fig, ax)
    plt.tight_layout(pad=0.5)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-size:0.6rem; letter-spacing:0.3em; color:#7B6FD0;
                text-transform:uppercase; margin-bottom:1.2rem;
                font-family:'Jost',sans-serif;">Configuration</div>
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
    words_per_line  = st.slider("Words per line",     4,   16,  8,
                                help="Controls poem line wrapping in the output.")
    step_by_step    = st.checkbox("Step-by-step mode", value=False)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.6rem; letter-spacing:0.18em; color:#6B6590;
                text-transform:uppercase; font-family:'Jost',sans-serif;">
        Attention Visualisation</div>
    """, unsafe_allow_html=True)
    show_attn  = st.checkbox("Show attention map", value=True)
    attn_layer = st.number_input("Layer index", min_value=0, max_value=15, value=0, step=1)
    attn_head  = st.number_input("Head index",  min_value=0, max_value=31, value=0, step=1)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.6rem; letter-spacing:0.18em; color:#6B6590;
                text-transform:uppercase; font-family:'Jost',sans-serif; margin-bottom:0.5rem;">About</div>
    <div style="font-size:0.76rem; color:#4A4870; line-height:1.85;">
        GPT-style Transformer<br>
        Pre-LN · Fused QKV · GELU FFN<br>
        Weight-tied output · NumPy-only<br>
        IT08X97 Artificial Intelligence
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

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Input
# ──────────────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("""
    <div style="font-size:0.6rem; letter-spacing:0.28em; color:#7B6FD0;
                text-transform:uppercase; margin-bottom:0.5rem;
                font-family:'Jost',sans-serif;">Seed Text</div>
    """, unsafe_allow_html=True)
    seed_text    = st.text_area(
        label="seed", label_visibility="collapsed",
        value="Shall I compare thee to a summer's day?\n",
        height=140,
    )
    generate_btn = st.button("◈  Generate Verse")

with col_right:
    st.markdown("""
    <div style="font-size:0.6rem; letter-spacing:0.28em; color:#7B6FD0;
                text-transform:uppercase; margin-bottom:0.9rem;
                font-family:'Jost',sans-serif;">How It Works</div>
    <div style="font-size:0.82rem; color:#6B6590; line-height:2.1; font-family:'Jost',sans-serif;">
        <span style="color:#C5BFF5; font-size:0.7rem; letter-spacing:0.1em;">I</span>
        &nbsp; Seed text is word-tokenised<br>
        <span style="color:#C5BFF5; font-size:0.7rem; letter-spacing:0.1em;">II</span>
        &nbsp; Embedding + positional encoding<br>
        <span style="color:#C5BFF5; font-size:0.7rem; letter-spacing:0.1em;">III</span>
        &nbsp; N × Causal self-attention + FFN<br>
        <span style="color:#C5BFF5; font-size:0.7rem; letter-spacing:0.1em;">IV</span>
        &nbsp; Weight-tied projection → vocab<br>
        <span style="color:#C5BFF5; font-size:0.7rem; letter-spacing:0.1em;">V</span>
        &nbsp; Token sampled, appended, repeated<br>
        <span style="color:#C5BFF5; font-size:0.7rem; letter-spacing:0.1em;">VI</span>
        &nbsp; Verse formatted line by line
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='ornament'><span style='color:#C5BFF5;font-size:0.6rem;letter-spacing:0.3em;color:#6B6590'>◈</span></div>", unsafe_allow_html=True)


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

    progress = st.progress(0, text="Composing verse…")

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
                          text=f"Composing… {step+1}/{n_tokens}")

    progress.empty()

    # ── Poem display ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:0.6rem; letter-spacing:0.28em; color:#7B6FD0;
                text-transform:uppercase; margin-bottom:0.8rem;
                font-family:'Jost',sans-serif;">Generated Verse</div>
    """, unsafe_allow_html=True)

    gen_words = [vocab.idx2word.get(i, "?") for i in generated_ids]
    poem_html = _words_to_poem_html(seed_word_tokens, gen_words,
                                    words_per_line=words_per_line)
    st.markdown(poem_html, unsafe_allow_html=True)

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    # ── Last-step candidate chart ─────────────────────────────────────────────
    if last_logits is not None:
        st.markdown("""
        <div style="font-size:0.6rem; letter-spacing:0.28em; color:#7B6FD0;
                    text-transform:uppercase; margin-bottom:0.6rem;
                    font-family:'Jost',sans-serif;">Final Step — Token Candidates</div>
        """, unsafe_allow_html=True)
        disp_logits = step_data[-1][1] if step_data else last_logits
        fig_cand = plot_candidates(disp_logits, vocab, temperature,
                                   generated_ids[-1], k=show_candidates)
        st.pyplot(fig_cand)
        plt.close(fig_cand)

    # ── Step-by-step breakdown ────────────────────────────────────────────────
    if step_by_step and step_data:
        st.markdown("<div class='ornament'><span style='color:#6B6590;font-size:0.6rem;'>◈</span></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.6rem; letter-spacing:0.28em; color:#7B6FD0;
                    text-transform:uppercase; margin-bottom:0.8rem;
                    font-family:'Jost',sans-serif;">Step-by-Step Selection</div>
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
                        f'<div style="font-size:0.62rem; color:#6B6590; letter-spacing:0.16em;'
                        f' text-transform:uppercase; margin-bottom:0.3rem;'
                        f' font-family:Jost,sans-serif;">'
                        f'Step {step_num} &rarr; '
                        f'<span style="color:#C5BFF5; font-style:italic;">{chosen_label}</span></div>',
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
    <div style="text-align:center; padding: 4rem 0; color:#2A2650;">
        <div style="font-size:2.2rem; margin-bottom:1.2rem; opacity:0.3;
                    background: linear-gradient(135deg, #7B6FD0, #C5BFF5);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;">◈</div>
        <div style="font-family:'EB Garamond', serif; font-size:1.35rem;
                    font-style:italic; color:#3A3660; letter-spacing:0.04em;">
            Enter a seed and let the model continue the verse.
        </div>
        <div style="font-size:0.65rem; letter-spacing:0.22em; color:#2A2650;
                    text-transform:uppercase; margin-top:0.8rem;
                    font-family:'Jost',sans-serif;">
            Adjust temperature &amp; line length in the sidebar
        </div>
    </div>
    """, unsafe_allow_html=True)