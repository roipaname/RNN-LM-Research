"""
app.py
------
Streamlit inference interface for the RNN Language Model.

Run with:
    streamlit run app.py

Features:
  - Word-level or character-level model toggle
  - Step-by-step token generation with probability visualization
  - Candidate token distribution shown at each step
  - Attention alpha heatmap
  - Luxury dark aesthetic
"""

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

from src.data_loader import DataLoader
from src.rnnlm       import RNNLM
from src.sampling    import top_k_probs
from src.settings    import BEST_MODEL_WORD, BEST_MODEL_LETTER

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Verse Engine",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Global CSS — luxury dark theme
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Montserrat:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --gold:      #C9A84C;
    --gold-dim:  #8B6914;
    --cream:     #F2EAD3;
    --ivory:     #EDE5CC;
    --bg:        #0C0B09;
    --bg2:       #141210;
    --bg3:       #1C1A16;
    --border:    rgba(201,168,76,0.18);
    --text:      #D4C5A0;
    --muted:     #7A6F5A;
}

/* ── Global reset ── */
html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Montserrat', sans-serif;
    font-weight: 300;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1200px; }

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    color: var(--cream) !important;
    font-weight: 300 !important;
    letter-spacing: 0.04em;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Text area ── */
textarea {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--cream) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.1rem !important;
    border-radius: 2px !important;
}
textarea:focus { border-color: var(--gold) !important; box-shadow: 0 0 0 1px var(--gold-dim) !important; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--gold) !important;
    color: var(--gold) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2.2rem !important;
    border-radius: 0 !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: var(--gold) !important;
    color: var(--bg) !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }
div[data-baseweb="slider"] > div > div > div { background: var(--gold) !important; }

/* ── Selectbox / radio ── */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div:hover {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    border-radius: 0 !important;
    color: var(--cream) !important;
}
.stRadio label { color: var(--text) !important; font-size: 0.85rem !important; }

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Info / warning boxes ── */
.stAlert { background: var(--bg3) !important; border-color: var(--border) !important; border-radius: 0 !important; }

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--bg3);
    border: 1px solid var(--border);
    padding: 1rem 1.4rem;
}
div[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.72rem !important; letter-spacing: 0.12em; text-transform: uppercase; }
div[data-testid="stMetricValue"] { color: var(--gold) !important; font-family: 'Cormorant Garamond', serif !important; font-size: 2rem !important; }

/* ── Expander ── */
details { border: 1px solid var(--border) !important; background: var(--bg2) !important; border-radius: 0 !important; }
summary { color: var(--gold) !important; font-size: 0.78rem !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; }

/* ── Token chip styles ── */
.token-chosen {
    display: inline-block;
    background: linear-gradient(135deg, #2A2310, #1A1A0E);
    border: 1px solid var(--gold);
    color: var(--gold);
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05rem;
    padding: 0.15rem 0.5rem;
    margin: 0.1rem;
    border-radius: 1px;
}
.token-seed {
    display: inline-block;
    color: var(--cream);
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05rem;
    opacity: 0.7;
    margin: 0.1rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Masthead
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 0.5rem;">
    <div style="font-size:0.72rem; letter-spacing:0.3em; color:#8B6914; text-transform:uppercase; margin-bottom:0.5rem;">
        Recurrent Neural Language Model
    </div>
    <h1 style="font-size:3.2rem; margin:0; letter-spacing:0.06em;">Verse Engine</h1>
    <div style="width:60px; height:1px; background:#C9A84C; margin:1rem auto;"></div>
    <div style="font-size:0.75rem; letter-spacing:0.2em; color:#7A6F5A; text-transform:uppercase;">
        Token-by-Token Generation &nbsp;·&nbsp; Probability Visualization
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Resource loading (cached per model mode)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Preparing the model…")
def load_resources(mode: str):
    """Load DataLoader + RNNLM for the given tokenisation mode."""
    ckpt = str(BEST_MODEL_WORD if mode == "word" else BEST_MODEL_LETTER)

    loader = DataLoader(seq_len=100, batch_size=1, mode=mode)

    if os.path.exists(ckpt):
        model = RNNLM(vocab_size=loader.vocab.size, embed_dim=64,
                      hidden_dim=256, num_layers=2, keep_prob=1.0)
        model.load(ckpt)
    else:
        st.warning(f"No checkpoint found at `{ckpt}`. Using random weights — train first.")
        model = RNNLM(vocab_size=loader.vocab.size, embed_dim=64,
                      hidden_dim=256, num_layers=2, keep_prob=1.0)

    return loader, model, ckpt


# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib theme helper
# ──────────────────────────────────────────────────────────────────────────────

def _apply_dark_theme(fig, ax):
    fig.patch.set_facecolor("#141210")
    ax.set_facecolor("#1C1A16")
    ax.tick_params(colors="#7A6F5A", labelsize=8)
    ax.xaxis.label.set_color("#7A6F5A")
    ax.yaxis.label.set_color("#7A6F5A")
    ax.title.set_color("#D4C5A0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2E2B24")


# ──────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_candidates(logits: np.ndarray, vocab, temperature: float,
                    chosen_idx: int, k: int = 8):
    """
    Horizontal bar chart showing top-k candidate tokens and their probabilities,
    highlighting the chosen token in gold.
    """
    indices, probs = top_k_probs(logits, temperature=temperature, k=k)

    if vocab.mode == "char":
        labels = [repr(vocab.idx2char.get(i, "?")) for i in indices]
    else:
        labels = [vocab.idx2char.get(i, "?") for i in indices]

    colors = ["#C9A84C" if i == chosen_idx else "#2E2B24" for i in indices]
    edge_colors = ["#C9A84C" if i == chosen_idx else "#3E3A30" for i in indices]

    fig, ax = plt.subplots(figsize=(7, max(2.5, k * 0.38)))
    bars = ax.barh(
        labels[::-1], probs[::-1],
        color=colors[::-1], edgecolor=edge_colors[::-1],
        linewidth=0.8, height=0.65
    )

    # Probability labels
    for bar, p, i in zip(bars, probs[::-1], indices[::-1]):
        clr = "#C9A84C" if i == chosen_idx else "#7A6F5A"
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{p:.3f}", va="center", fontsize=8, color=clr,
                fontfamily="Montserrat")

    ax.set_xlim(0, min(1.0, probs.max() * 1.35))
    ax.set_xlabel("Probability", fontsize=8, color="#7A6F5A")
    ax.set_title("Next-token candidates", fontsize=10, pad=10,
                 fontfamily="Cormorant Garamond", fontstyle="italic")

    # Legend
    chosen_patch = mpatches.Patch(color="#C9A84C", label="Chosen token")
    other_patch  = mpatches.Patch(color="#2E2B24", ec="#3E3A30", lw=0.8, label="Alternatives")
    ax.legend(handles=[chosen_patch, other_patch], fontsize=7,
              framealpha=0, labelcolor="#7A6F5A", loc="lower right")

    _apply_dark_theme(fig, ax)
    plt.tight_layout(pad=0.8)
    return fig


def plot_attention(alpha: np.ndarray, tokens: list[str], mode: str):
    """Heatmap of attention weights over input tokens."""
    if alpha is None or len(alpha) == 0:
        return None

    max_show = 40 if mode == "char" else 20
    tokens  = tokens[-max_show:]
    weights = alpha[-len(tokens):]

    fig, ax = plt.subplots(figsize=(max(7, len(tokens) * 0.28), 1.6))
    cmap = plt.cm.get_cmap("YlOrBr")
    im = ax.imshow(weights.reshape(1, -1), aspect="auto", cmap=cmap, vmin=0)

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(
        [repr(t) if mode == "char" else t for t in tokens],
        rotation=45, ha="right", fontsize=7, color="#7A6F5A"
    )
    ax.set_yticks([])
    ax.set_title("Attention weights", fontsize=9, pad=8,
                 fontfamily="Cormorant Garamond", fontstyle="italic")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02).ax.tick_params(
        labelsize=6, colors="#7A6F5A"
    )

    _apply_dark_theme(fig, ax)
    plt.tight_layout(pad=0.5)
    return fig


def generation_step_display(seed_text: str, generated_tokens: list,
                             vocab, mode: str):
    """Render the current sequence with seed greyed and new tokens gold."""
    if mode == "char":
        seed_parts = list(seed_text)
    else:
        seed_parts = seed_text.strip().split()

    html = '<div style="line-height:2.2; word-break:break-word;">'
    for t in seed_parts:
        html += f'<span class="token-seed">{t if mode == "word" else t}</span>'
    for t in generated_tokens:
        label = vocab.idx2char.get(t, "?")
        html += f'<span class="token-chosen">{label}</span>'
        if mode == "word":
            html += ' '
    html += '</div>'
    return html


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.25em; color:#8B6914;
                text-transform:uppercase; margin-bottom:1.2rem;">
        Configuration
    </div>
    """, unsafe_allow_html=True)

    model_mode = st.radio(
        "Model",
        options=["char", "word"],
        format_func=lambda x: "Character-level" if x == "char" else "Word-level",
        index=0,
        help="Select which trained model to load.",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    temperature = st.slider(
        "Temperature", 0.0, 2.0, 1.0, 0.05,
        help="0 = greedy · higher = more creative"
    )
    top_k = st.slider(
        "Top-k sampling", 0, 50, 0,
        help="0 = sample from full vocab"
    )
    n_tokens = st.slider(
        "Tokens to generate", 1, 300, 60, 1
    )
    show_candidates = st.slider(
        "Candidates to show", 3, 15, 8,
        help="Number of alternative tokens displayed per step"
    )
    step_by_step = st.checkbox(
        "Step-by-step mode",
        value=False,
        help="Show candidate distributions at every generation step (slower)."
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.12em; color:#7A6F5A; text-transform:uppercase;">
        About
    </div>
    <div style="font-size:0.78rem; color:#5A5244; margin-top:0.5rem; line-height:1.7;">
        GRU + Bahdanau Attention<br>
        NumPy-only implementation<br>
        IT08X97 Artificial Intelligence
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────────────────

loader, model, ckpt_path = load_resources(model_mode)
vocab = loader.vocab

# Show model info strip
c1, c2, c3 = st.columns(3)
c1.metric("Vocabulary", f"{vocab.size:,}")
c2.metric("Mode", model_mode.upper())
c3.metric("Checkpoint", os.path.basename(ckpt_path))

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Input area
# ──────────────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.22em; color:#8B6914;
                text-transform:uppercase; margin-bottom:0.4rem;">
        Seed Text
    </div>
    """, unsafe_allow_html=True)

    seed_text = st.text_area(
        label="seed",
        label_visibility="collapsed",
        value="Shall I compare thee to a summer's day?\n",
        height=140,
        placeholder="Enter your seed text here…",
    )

    generate_btn = st.button("✦  Generate")

with col_right:
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.22em; color:#8B6914;
                text-transform:uppercase; margin-bottom:0.8rem;">
        How It Works
    </div>
    <div style="font-size:0.82rem; color:#7A6F5A; line-height:1.9;">
        <span style="color:#C9A84C;">01</span> &nbsp; Seed text is tokenised<br>
        <span style="color:#C9A84C;">02</span> &nbsp; GRU encodes each token<br>
        <span style="color:#C9A84C;">03</span> &nbsp; Attention weights past states<br>
        <span style="color:#C9A84C;">04</span> &nbsp; Projection → probability over vocab<br>
        <span style="color:#C9A84C;">05</span> &nbsp; Token sampled, appended, repeated
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

if generate_btn:
    unk_id = vocab.char2idx.get("<UNK>", 1)

    # Encode seed
    if model_mode == "char":
        seed_tokens = np.array(
            [vocab.char2idx.get(ch, unk_id) for ch in seed_text],
            dtype=np.int32,
        )
        seed_display = list(seed_text)
    else:
        words = seed_text.strip().split()
        seed_tokens = np.array(
            [vocab.char2idx.get(w, unk_id) for w in words],
            dtype=np.int32,
        )
        seed_display = words

    if len(seed_tokens) == 0:
        st.warning("Please enter some seed text.")
        st.stop()

    # ── Generation state ──────────────────────────────────────────
    generated_ids   = []
    step_data       = []   # list of (chosen_idx, logits) for visualization
    alpha_last      = None

    progress = st.progress(0, text="Generating…")

    tokens_so_far = list(seed_tokens)

    for step in range(n_tokens):
        inp = np.array(tokens_so_far[-512:], dtype=np.int32)
        probs, _ = model.forward(inp, targets=None, training=False)
        alpha_last = model._cache["alpha"]

        logits = np.log(probs[-1] + 1e-12)

        # Sample
        from src.sampling import sample_token
        next_tok = sample_token(logits, temperature, top_k)

        generated_ids.append(next_tok)
        tokens_so_far.append(next_tok)

        if step_by_step:
            step_data.append((next_tok, logits.copy()))

        progress.progress((step + 1) / n_tokens,
                          text=f"Generating… {step+1}/{n_tokens}")

    progress.empty()

    # ── Output: sequence display ──────────────────────────────────
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.22em; color:#8B6914;
                text-transform:uppercase; margin-bottom:0.6rem;">
        Generated Sequence
    </div>
    """, unsafe_allow_html=True)

    seq_html = generation_step_display(seed_text, generated_ids, vocab, model_mode)
    st.markdown(
        f'<div style="background:#141210; border:1px solid rgba(201,168,76,0.18); '
        f'padding:1.4rem 1.6rem; border-radius:1px;">{seq_html}</div>',
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Final step: candidate chart ───────────────────────────────
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.22em; color:#8B6914;
                text-transform:uppercase; margin-bottom:0.6rem;">
        Last Step — Token Candidates
    </div>
    """, unsafe_allow_html=True)

    inp_final = np.array(tokens_so_far[-512:-1], dtype=np.int32)
    probs_final, _ = model.forward(inp_final, targets=None, training=False)
    logits_final = np.log(probs_final[-1] + 1e-12)
    chosen_last = generated_ids[-1]

    fig_cand = plot_candidates(logits_final, vocab, temperature,
                               chosen_last, k=show_candidates)
    st.pyplot(fig_cand)
    plt.close(fig_cand)

    # ── Step-by-step mode ─────────────────────────────────────────
    if step_by_step and step_data:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.65rem; letter-spacing:0.22em; color:#8B6914;
                    text-transform:uppercase; margin-bottom:0.8rem;">
            Step-by-Step Token Selection
        </div>
        """, unsafe_allow_html=True)

        max_steps_shown = min(len(step_data), 12)
        cols_per_row = 3
        steps_to_show = step_data[:max_steps_shown]

        for row_start in range(0, len(steps_to_show), cols_per_row):
            row_steps = steps_to_show[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, (chosen_idx, logits) in zip(cols, row_steps):
                step_num = row_start + steps_to_show.index((chosen_idx, logits)) + 1
                chosen_label = vocab.idx2char.get(chosen_idx, "?")
                with col:
                    st.markdown(
                        f'<div style="font-size:0.65rem; color:#8B6914; '
                        f'letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.3rem;">'
                        f'Step {step_num} → <span style="color:#C9A84C">'
                        f'{"repr(chosen_label)" if model_mode == "char" else chosen_label}'
                        f'</span></div>',
                        unsafe_allow_html=True
                    )
                    fig = plot_candidates(logits, vocab, temperature,
                                          chosen_idx, k=min(5, show_candidates))
                    st.pyplot(fig)
                    plt.close(fig)

        if len(step_data) > max_steps_shown:
            st.caption(
                f"Showing first {max_steps_shown} of {len(step_data)} steps. "
                "Reduce 'Tokens to generate' to see all steps."
            )

    # ── Attention heatmap ─────────────────────────────────────────
    with st.expander("Attention Weights", expanded=False):
        if alpha_last is not None and len(alpha_last) > 0:
            all_tokens = seed_display + [
                vocab.idx2char.get(i, "?") for i in generated_ids
            ]
            fig_attn = plot_attention(alpha_last, all_tokens, model_mode)
            if fig_attn:
                st.pyplot(fig_attn)
                plt.close(fig_attn)
        else:
            st.info("No attention weights available.")

else:
    # ── Idle state ────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 3.5rem 0; color:#3E3A30;">
        <div style="font-size:2.5rem; margin-bottom:1rem; opacity:0.4;">✦</div>
        <div style="font-family:'Cormorant Garamond', serif; font-size:1.3rem;
                    font-style:italic; color:#5A5244; letter-spacing:0.05em;">
            Enter a seed and let the model continue the verse.
        </div>
    </div>
    """, unsafe_allow_html=True)