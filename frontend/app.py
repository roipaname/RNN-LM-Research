"""
app.py
------
Streamlit inference interface for the RNN Language Model.

Run with:
    streamlit run app.py

Works with checkpoints saved by either the training notebook or train.py.
Hyperparams (vocab size, embed dim, hidden dim, num layers) are read
directly from the .npz so no hardcoded values are needed.
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

from src.data_loader import DataLoader
from src.rnnlm       import RNNLM
from src.sampling    import top_k_probs
from src.settings    import BEST_MODEL_WORD, BEST_MODEL_LETTER

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
    --gold:      #C9A84C;
    --gold-dim:  #8B6914;
    --cream:     #F2EAD3;
    --bg:        #0C0B09;
    --bg2:       #141210;
    --bg3:       #1C1A16;
    --border:    rgba(201,168,76,0.18);
    --text:      #D4C5A0;
    --muted:     #7A6F5A;
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
# Checkpoint hparam inference (mirrors train.py logic — works for both
# notebook-saved and train.py-saved .npz files)
# ──────────────────────────────────────────────────────────────────────────────

def _infer_hparams_from_npz(ckpt_path: str) -> dict | None:
    """
    Return vocab_size, embed_dim, hidden_dim, num_layers from a checkpoint.
    Returns None if the file doesn't exist.
    Works whether or not the checkpoint contains explicit scalar hparam keys.
    """
    import re
    path = str(ckpt_path)
    if not path.endswith(".npz"):
        path += ".npz"
    if not os.path.exists(path):
        return None

    data = np.load(path)

    # Explicit keys (train.py-saved)
    if "_vocab_size" in data:
        return {
            "vocab_size": int(data["_vocab_size"]),
            "embed_dim":  int(data["_embed_dim"]),
            "hidden_dim": int(data["_hidden_dim"]),
            "num_layers": int(data["_num_layers"]),
        }

    # Shape inference (notebook-saved)
    if "W_out" not in data or "emb_E" not in data:
        return None

    vocab_size, hidden_dim = data["W_out"].shape
    _,          embed_dim  = data["emb_E"].shape

    layer_indices = set()
    for k in data.keys():
        m = re.search(r"(?:cells_|layer_?)(\d+)", k)
        if m:
            layer_indices.add(int(m.group(1)))
    num_layers = len(layer_indices) if layer_indices else 2

    return {
        "vocab_size": vocab_size,
        "embed_dim":  embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Resource loading — cached per (mode) so switching models is fast
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_resources(mode: str):
    """
    Load DataLoader + RNNLM for the given tokenisation mode.

    The checkpoint's stored vocab size is used to cap the DataLoader vocab
    so it matches exactly — no manual max_vocab bookkeeping needed.
    """
    ckpt_path = str(BEST_MODEL_WORD if mode == "word" else BEST_MODEL_LETTER)

    hp = _infer_hparams_from_npz(ckpt_path)

    if hp is not None:
        # Build DataLoader with vocab capped to match checkpoint
        max_vocab = (hp["vocab_size"] - 2) if mode == "word" else None
        loader = DataLoader(seq_len=100, batch_size=1, mode=mode,
                            max_vocab=max_vocab)

        # Hard check — vocab must match before we try to load weights
        if loader.vocab.size != hp["vocab_size"]:
            st.error(
                f"Vocab mismatch after cap: DataLoader={loader.vocab.size}, "
                f"checkpoint={hp['vocab_size']}. "
                "The corpus may differ from when the checkpoint was trained."
            )
            model = RNNLM(vocab_size=loader.vocab.size, embed_dim=hp["embed_dim"],
                          hidden_dim=hp["hidden_dim"], num_layers=hp["num_layers"],
                          keep_prob=1.0)
        else:
            model = RNNLM(vocab_size=hp["vocab_size"], embed_dim=hp["embed_dim"],
                          hidden_dim=hp["hidden_dim"], num_layers=hp["num_layers"],
                          keep_prob=1.0)
            model.load(ckpt_path)
    else:
        st.warning(
            f"No checkpoint found at `{ckpt_path}`. "
            "Using random weights — train the model first."
        )
        loader = DataLoader(seq_len=100, batch_size=1, mode=mode)
        model  = RNNLM(vocab_size=loader.vocab.size, embed_dim=64,
                       hidden_dim=256, num_layers=2, keep_prob=1.0)

    return loader, model, ckpt_path


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
    """Horizontal bar chart of top-k candidates, chosen token highlighted."""
    indices, probs = top_k_probs(logits, temperature=temperature, k=k)
    labels  = [vocab.idx2char.get(i, "?") for i in indices]
    if vocab.mode == "char":
        labels = [repr(l) for l in labels]

    colors      = ["#C9A84C" if i == chosen_idx else "#2E2B24" for i in indices]
    edge_colors = ["#C9A84C" if i == chosen_idx else "#3E3A30" for i in indices]

    fig, ax = plt.subplots(figsize=(7, max(2.5, k * 0.38)))
    bars = ax.barh(labels[::-1], probs[::-1],
                   color=colors[::-1], edgecolor=edge_colors[::-1],
                   linewidth=0.8, height=0.65)

    for bar, p, i in zip(bars, probs[::-1], indices[::-1]):
        clr = "#C9A84C" if i == chosen_idx else "#7A6F5A"
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{p:.3f}", va="center", fontsize=8, color=clr)

    ax.set_xlim(0, min(1.0, probs.max() * 1.35))
    ax.set_xlabel("Probability", fontsize=8)
    ax.set_title("Next-token candidates", fontsize=10, pad=10,
                 fontfamily="serif", fontstyle="italic")

    chosen_patch = mpatches.Patch(color="#C9A84C", label="Chosen")
    other_patch  = mpatches.Patch(color="#2E2B24", ec="#3E3A30", lw=0.8,
                                  label="Alternatives")
    ax.legend(handles=[chosen_patch, other_patch], fontsize=7,
              framealpha=0, labelcolor="#7A6F5A", loc="lower right")

    _dark_theme(fig, ax)
    plt.tight_layout(pad=0.8)
    return fig


def plot_attention(alpha: np.ndarray, tokens: list[str], mode: str):
    """Heatmap of attention weights."""
    if alpha is None or len(alpha) == 0:
        return None

    max_show = 40 if mode == "char" else 20
    tokens   = tokens[-max_show:]
    weights  = alpha[-len(tokens):]

    fig, ax = plt.subplots(figsize=(max(7, len(tokens) * 0.28), 1.6))
    cmap = plt.cm.get_cmap("YlOrBr")
    im   = ax.imshow(weights.reshape(1, -1), aspect="auto", cmap=cmap, vmin=0)

    labels = [repr(t) if mode == "char" else t for t in tokens]
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color="#7A6F5A")
    ax.set_yticks([])
    ax.set_title("Attention weights", fontsize=9, pad=8,
                 fontfamily="serif", fontstyle="italic")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02).ax.tick_params(
        labelsize=6, colors="#7A6F5A"
    )
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

    model_mode = st.radio(
        "Model",
        options=["char", "word"],
        format_func=lambda x: "Character-level" if x == "char" else "Word-level",
        index=0,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    temperature     = st.slider("Temperature",        0.0, 2.0, 1.0, 0.05)
    top_k           = st.slider("Top-k sampling",     0,   50,  0)
    n_tokens        = st.slider("Tokens to generate", 1,   300, 60)
    show_candidates = st.slider("Candidates to show", 3,   15,  8)
    step_by_step    = st.checkbox("Step-by-step mode", value=False,
                                  help="Show candidate chart at every generation step.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.12em; color:#7A6F5A;
                text-transform:uppercase;">About</div>
    <div style="font-size:0.78rem; color:#5A5244; margin-top:0.5rem; line-height:1.7;">
        GRU + Bahdanau Attention<br>NumPy-only implementation<br>
        IT08X97 Artificial Intelligence
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────────────────

loader, model, ckpt_path = load_resources(model_mode)
vocab = loader.vocab

c1, c2, c3 = st.columns(3)
c1.metric("Vocabulary",  f"{vocab.size:,}")
c2.metric("Mode",        model_mode.upper())
c3.metric("Checkpoint",  os.path.basename(ckpt_path))

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
    seed_text    = st.text_area(label="seed", label_visibility="collapsed",
                                value="Shall I compare thee to a summer's day?\n",
                                height=140)
    generate_btn = st.button("✦  Generate")

with col_right:
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.22em; color:#8B6914;
                text-transform:uppercase; margin-bottom:0.8rem;">How It Works</div>
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
    from src.sampling import sample_token

    unk_id = vocab.char2idx.get("<UNK>", 1)

    if model_mode == "char":
        seed_tokens   = np.array([vocab.char2idx.get(ch, unk_id) for ch in seed_text],
                                 dtype=np.int32)
        seed_display  = list(seed_text)
    else:
        words         = seed_text.strip().split()
        seed_tokens   = np.array([vocab.char2idx.get(w, unk_id) for w in words],
                                 dtype=np.int32)
        seed_display  = words

    if len(seed_tokens) == 0:
        st.warning("Please enter some seed text.")
        st.stop()

    generated_ids: list[int]           = []
    step_data:     list[tuple]         = []
    alpha_last:    np.ndarray | None   = None
    tokens_so_far                      = list(seed_tokens)

    progress = st.progress(0, text="Generating…")

    for step in range(n_tokens):
        inp    = np.array(tokens_so_far[-512:], dtype=np.int32)
        probs, _ = model.forward(inp, targets=None, training=False)
        alpha_last = model._cache["alpha"]

        logits   = np.log(probs[-1] + 1e-12)
        next_tok = sample_token(logits, temperature, top_k)

        generated_ids.append(next_tok)
        tokens_so_far.append(next_tok)

        if step_by_step:
            step_data.append((next_tok, logits.copy()))

        progress.progress((step + 1) / n_tokens,
                          text=f"Generating… {step + 1}/{n_tokens}")

    progress.empty()

    # ── Sequence display ──────────────────────────────────────────
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.22em; color:#8B6914;
                text-transform:uppercase; margin-bottom:0.6rem;">Generated Sequence</div>
    """, unsafe_allow_html=True)

    seed_html = "".join(
        f'<span style="color:#EDE5CC; opacity:0.6; font-family:Cormorant Garamond,serif;'
        f' font-size:1.05rem; margin:0.05rem">{t}</span>{"&nbsp;" if model_mode=="word" else ""}'
        for t in seed_display
    )
    gen_html = "".join(
        f'<span style="display:inline-block; background:linear-gradient(135deg,#2A2310,#1A1A0E);'
        f' border:1px solid #C9A84C; color:#C9A84C; font-family:Cormorant Garamond,serif;'
        f' font-size:1.05rem; padding:0.1rem 0.4rem; margin:0.1rem; border-radius:1px">'
        f'{vocab.idx2char.get(i,"?")}</span>{"&nbsp;" if model_mode=="word" else ""}'
        for i in generated_ids
    )
    st.markdown(
        f'<div style="background:#141210; border:1px solid rgba(201,168,76,0.18);'
        f' padding:1.4rem 1.6rem; border-radius:1px; line-height:2.2;">'
        f'{seed_html}{gen_html}</div>',
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Last-step candidate chart ─────────────────────────────────
    st.markdown("""
    <div style="font-size:0.65rem; letter-spacing:0.22em; color:#8B6914;
                text-transform:uppercase; margin-bottom:0.6rem;">Last Step — Token Candidates</div>
    """, unsafe_allow_html=True)

    inp_final       = np.array(tokens_so_far[-512:-1], dtype=np.int32)
    probs_final, _  = model.forward(inp_final, targets=None, training=False)
    logits_final    = np.log(probs_final[-1] + 1e-12)

    fig_cand = plot_candidates(logits_final, vocab, temperature,
                               generated_ids[-1], k=show_candidates)
    st.pyplot(fig_cand)
    plt.close(fig_cand)

    # ── Step-by-step breakdown ────────────────────────────────────
    if step_by_step and step_data:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.65rem; letter-spacing:0.22em; color:#8B6914;
                    text-transform:uppercase; margin-bottom:0.8rem;">Step-by-Step Selection</div>
        """, unsafe_allow_html=True)

        shown      = step_data[:12]
        cols_per_row = 3
        for row_start in range(0, len(shown), cols_per_row):
            row   = shown[row_start:row_start + cols_per_row]
            cols  = st.columns(cols_per_row)
            for col_idx, (chosen_idx, logits) in enumerate(row):
                step_num     = row_start + col_idx + 1
                chosen_label = vocab.idx2char.get(chosen_idx, "?")
                with cols[col_idx]:
                    st.markdown(
                        f'<div style="font-size:0.65rem; color:#8B6914; '
                        f'letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.3rem;">'
                        f'Step {step_num} → '
                        f'<span style="color:#C9A84C">'
                        f'{repr(chosen_label) if model_mode=="char" else chosen_label}'
                        f'</span></div>',
                        unsafe_allow_html=True,
                    )
                    fig = plot_candidates(logits, vocab, temperature,
                                         chosen_idx, k=min(5, show_candidates))
                    st.pyplot(fig)
                    plt.close(fig)

        if len(step_data) > 12:
            st.caption(f"Showing first 12 of {len(step_data)} steps.")

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
    st.markdown("""
    <div style="text-align:center; padding: 3.5rem 0; color:#3E3A30;">
        <div style="font-size:2.5rem; margin-bottom:1rem; opacity:0.4;">✦</div>
        <div style="font-family:'Cormorant Garamond', serif; font-size:1.3rem;
                    font-style:italic; color:#5A5244; letter-spacing:0.05em;">
            Enter a seed and let the model continue the verse.
        </div>
    </div>
    """, unsafe_allow_html=True)