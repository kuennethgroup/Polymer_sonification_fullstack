"""
Polymer Sonification App
========================
Streamlit MVP — single-file implementation.

User provides a polymer structure (PSMILES text or Ketcher drawing),
the app runs inference through a local ML model and returns an audio
file together with a waveform visualisation.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import io
import math
import pickle
import wave
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Polymer2Audio model architecture (must be defined before unpickling)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 20000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class Polymer2Audio(nn.Module):
        def __init__(
            self,
            fp_dim: int = 600,
            vocab_size: int = 2000,
            d_model: int = 512,
            nhead: int = 8,
            num_layers: int = 4,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.d_model = d_model
            self.fp_proj = nn.Linear(fp_dim, d_model)
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout,
                batch_first=True,
            )
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            self.fc_out = nn.Linear(d_model, vocab_size)

        def generate_square_subsequent_mask(self, sz, device):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            return (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
                .to(device)
            )

        def forward(self, polymer_fp, tgt_tokens, tgt_padding_mask=None):
            device = tgt_tokens.device
            memory = self.fp_proj(polymer_fp).unsqueeze(1)
            tgt_emb = self.token_embedding(tgt_tokens) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            tgt_mask = self.generate_square_subsequent_mask(tgt_tokens.size(1), device)
            output = self.transformer_decoder(
                tgt=tgt_emb, memory=memory,
                tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask,
            )
            return self.fc_out(output)

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration  (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Polymer Sonification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Optional dependencies — degrade gracefully if not installed
# ──────────────────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from psmiles import PolymerSmiles as PS
    PSMILES_AVAILABLE = True
except ImportError:
    PSMILES_AVAILABLE = False

try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except ImportError:
    KETCHER_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH     = Path("polymer_soni1_model.pkl")
SAMPLE_RATE    = 22_050       # Hz — used for mock audio generation
MOCK_DURATION  = 5.0          # seconds
EXAMPLE_PSMILES = "[*]CC[*]"  # polyethylene repeat unit


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — only for the lab / results Streamlit sections
# (story sections carry their own inline styles inside st.components.v1.html)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    html, body, .stApp { background: #0a0d14; color: #c9d1d9; }
    .block-container {
        padding-top: 0 !important;
        padding-left: max(16px, 3vw) !important;
        padding-right: max(16px, 3vw) !important;
    }

    .divider { border-top: 1px solid #1f2733; margin: 1.2rem 0; }
    .section-label {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px;
        color: #4a6fa5; text-transform: uppercase; margin-bottom: 6px;
    }
    .psmiles-box {
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 10px 14px; font-family: 'Courier New', monospace;
        font-size: 0.95rem; color: #7ee787; word-break: break-all;
    }
    .result-card {
        background: #161b22; border: 1px solid #1f6feb;
        border-radius: 12px; padding: 24px 28px; margin-top: 10px;
    }
    .badge-mock {
        display: inline-block; background: #1a3a1a; color: #3fb950;
        border: 1px solid #238636; border-radius: 12px;
        padding: 2px 10px; font-size: 0.72rem; font-weight: 600;
    }
    .badge-real {
        display: inline-block; background: #0d2a5c; color: #58a6ff;
        border: 1px solid #1f6feb; border-radius: 12px;
        padding: 2px 10px; font-size: 0.72rem; font-weight: 600;
    }

    /* ── Mobile: reduce block padding ── */
    @media (max-width: 640px) {
        .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
        section[data-testid="stSidebar"] { display: none; }
    }

    /* ── Primary action buttons (Sonify, Start Sonifying) ── */
    div[data-testid="stButton"] > button[kind="secondary"],
    div[data-testid="stButton"] > button {
        background: #2d1b69 !important;
        color: #d4b8ff !important;
        border: 1px solid #5a3ea0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: background 0.15s, border-color 0.15s !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: #3d2880 !important;
        border-color: #7c5cc4 !important;
        color: #e8d8ff !important;
    }
    div[data-testid="stButton"] > button:active {
        background: #1e1047 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ──────────────────────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults: dict = {
        # final confirmed PSMILES used for prediction
        "psmiles": "",
        # audio output
        "audio_bytes": None,
        "predicted": False,
        # real-time validation state (updated as user types / draws)
        "rt_mol": None,           # rdkit Mol object for display
        "rt_psmiles_str": "",     # canonical PSMILES string
        "rt_last_input": "",      # last raw input seen (to detect changes)
        # example loading
        "example_psmiles_for_input": "",
        # whether the inline lab tool is expanded
        "show_lab": False,
        # flag to trigger scroll-to-top of lab section on first render after expand
        "just_expanded": False,
        # reverse design tool state
        "show_reverse_lab": False,
        "just_expanded_reverse": False,
        "reverse_psmiles": "",
        "reverse_mol": None,
        "scroll_to_result": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


def _reset() -> None:
    st.session_state.psmiles            = ""
    st.session_state.audio_bytes        = None
    st.session_state.predicted          = False
    st.session_state.rt_mol             = None
    st.session_state.rt_psmiles_str     = ""
    st.session_state.rt_last_input      = ""
    st.session_state.example_psmiles_for_input = ""
    st.session_state.show_lab           = False
    st.session_state.just_expanded      = False
    st.session_state.show_reverse_lab   = False
    st.session_state.just_expanded_reverse = False
    st.session_state.reverse_psmiles    = ""
    st.session_state.reverse_mol        = None


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
@st.cache_resource
def load_model():
    """
    Load Polymer2Audio from MODEL_PATH and EnCodec backend.
    Returns (p2a_model, encodec_model) or "mock" if unavailable.
    """
    if not MODEL_PATH.exists() or not TORCH_AVAILABLE:
        return "mock"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(MODEL_PATH, "rb") as fh:
            p2a = pickle.load(fh)
        p2a.to(device).eval()

        from encodec import EncodecModel
        enc = EncodecModel.encodec_model_24khz()
        enc.set_target_bandwidth(6.0)
        enc.to(device).eval()
        return (p2a, enc)
    except Exception as e:
        st.warning(f"Model load failed ({e}), using mock audio.")
        return "mock"


# ──────────────────────────────────────────────────────────────────────────────
# RDKit helpers — molecule validation & SVG rendering
# ──────────────────────────────────────────────────────────────────────────────
def validate_psmiles(raw: str) -> Tuple[Optional[object], Optional[object], str]:
    """
    Validate *raw* as a PSMILES string.

    Returns (ps_obj, rdkit_mol, error_message).
    On success: error_message == "".
    On failure: ps_obj and rdkit_mol are None, error_message describes the problem.

    Requires both ``psmiles`` and ``rdkit`` to be installed for full validation;
    falls back to a basic RDKit-only check if ``psmiles`` is absent.
    """
    if not raw or not raw.strip():
        return None, None, "Empty input."

    smiles_for_rdkit = raw.strip()

    # --- psmiles library path ---
    if PSMILES_AVAILABLE:
        try:
            ps_obj = PS(smiles_for_rdkit)
            # PS canonicalises and validates; get the underlying SMILES
            canonical_smiles = str(ps_obj)
        except Exception as exc:
            return None, None, f"PSMILES error: {exc}"
    else:
        ps_obj = None
        canonical_smiles = smiles_for_rdkit

    # --- RDKit structural check ---
    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            # Try replacing * with dummy atom [Xe] so RDKit can parse PSMILES
            relaxed = canonical_smiles.replace("[*]", "[Xe]").replace("*", "[Xe]")
            mol = Chem.MolFromSmiles(relaxed)
        if mol is None:
            return None, None, "RDKit could not parse the structure. Check your SMILES/PSMILES."
        try:
            AllChem.Compute2DCoords(mol)
        except Exception:
            pass
        return ps_obj, mol, ""

    # --- No RDKit: accept if psmiles parsed it ---
    if ps_obj is not None:
        return ps_obj, None, ""

    return None, None, "Install rdkit or psmiles for validation."


def mol_to_svg(mol, width: int = 300, height: int = 300) -> Optional[str]:
    """
    Render an RDKit Mol to an SVG string.
    Returns None if rendering fails or RDKit is unavailable.
    """
    if mol is None or not RDKIT_AVAILABLE:
        return None
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().addStereoAnnotation = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction / preprocessing
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(psmiles: str) -> np.ndarray:
    """600-bit Morgan fingerprint (radius=2) via RDKit."""
    n_bits = 600
    arr = np.zeros(n_bits, dtype=np.float32)
    if RDKIT_AVAILABLE:
        from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
        smiles = psmiles.strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            relaxed = smiles.replace("[*]", "[Xe]").replace("*", "[Xe]")
            mol = Chem.MolFromSmiles(relaxed)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            ConvertToNumpyArray(fp, arr)
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# Mock audio generation (stdlib only — no extra dependencies)
# ──────────────────────────────────────────────────────────────────────────────
def generate_mock_audio(psmiles: str) -> bytes:
    """
    FM-synthesis + chord + tremolo + reverb-tail mock audio seeded from PSMILES.
    Each polymer gets a unique tonal character.  Returns raw WAV bytes.
    """
    rng = np.random.default_rng(abs(hash(psmiles)) % (2 ** 32))
    SR  = SAMPLE_RATE
    DUR = MOCK_DURATION
    N   = int(SR * DUR)
    t   = np.linspace(0, DUR, N, endpoint=False)

    # ── 1. Pick a root note and a 3-note chord ────────────────────────────────
    # Roots spread across two octaves (110 – 440 Hz)
    root  = 110.0 * 2 ** (rng.random() * 2.0)
    # Intervals: minor/major third + fifth, randomly chosen
    ratios = rng.choice(
        [[1.0, 1.25, 1.5], [1.0, 1.2, 1.5], [1.0, 1.333, 1.667],
         [1.0, 1.25, 1.667], [1.0, 1.5, 2.0]],
    )
    freqs = [root * r for r in ratios]

    # ── 2. FM synthesis per note ──────────────────────────────────────────────
    def fm_tone(fc: float, fm_ratio: float, fm_idx: float) -> np.ndarray:
        fm = fm_ratio * fc
        modulator = fm_idx * np.sin(2 * math.pi * fm * t)
        return np.sin(2 * math.pi * fc * t + modulator)

    fm_ratio = 1.5 + rng.random() * 2.5   # modulator/carrier ratio
    fm_idx   = 1.0 + rng.random() * 4.0   # modulation depth

    chord = sum(
        w * fm_tone(f, fm_ratio, fm_idx)
        for f, w in zip(freqs, [0.55, 0.30, 0.20])
    )

    # ── 3. Add sub-bass pulse (root / 2) ─────────────────────────────────────
    sub = 0.18 * np.sin(2 * math.pi * (root / 2) * t)

    # ── 4. Tremolo (amplitude LFO) ────────────────────────────────────────────
    lfo_rate = 2.0 + rng.random() * 5.0          # 2–7 Hz
    lfo_depth = 0.25 + rng.random() * 0.35       # 25–60 %
    tremolo = 1.0 - lfo_depth * (0.5 - 0.5 * np.sin(2 * math.pi * lfo_rate * t))

    signal = (chord + sub) * tremolo

    # ── 5. Pitch vibrato on the chord ────────────────────────────────────────
    vib_rate  = 4.5 + rng.random() * 2.0
    vib_depth = 0.003 + rng.random() * 0.007     # ±semitone-ish
    phase_mod = vib_depth * np.cumsum(np.sin(2 * math.pi * vib_rate * t)) / SR
    signal   += 0.3 * np.sin(2 * math.pi * root * (t + phase_mod))

    # ── 6. Sparse high-frequency partials (brightness) ───────────────────────
    for k in [4, 6, 8]:
        amp = rng.random() * 0.06
        signal += amp * np.sin(2 * math.pi * root * k * t)

    # ── 7. Envelope: smooth ADSR ──────────────────────────────────────────────
    a = int(0.08 * SR)   # attack  80 ms
    d = int(0.15 * SR)   # decay  150 ms
    s_level = 0.72
    r = int(0.55 * SR)   # release 550 ms

    env = np.ones(N) * s_level
    env[:a]        = np.linspace(0, 1, a)
    env[a:a+d]     = np.linspace(1, s_level, d)
    env[N-r:]      = np.linspace(s_level, 0, r)
    signal        *= env

    # ── 8. Simple comb-filter reverb tail ────────────────────────────────────
    delay_samples = int(0.035 * SR)   # ~35 ms room
    decay_fb      = 0.45
    reverb = signal.copy()
    for _ in range(6):
        reverb[delay_samples:] += decay_fb * reverb[:-delay_samples]
        decay_fb *= 0.55
    signal = 0.7 * signal + 0.3 * reverb

    # ── 9. Soft clip + normalise ──────────────────────────────────────────────
    signal = np.tanh(signal * 1.4)                          # soft clip
    signal = signal / np.max(np.abs(signal) + 1e-9)        # normalise
    signal = (signal * 32_767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(signal.tobytes())
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def run_inference(model, features: np.ndarray, psmiles: str) -> bytes:
    """Run Polymer2Audio + EnCodec decode pipeline, returns WAV bytes."""
    if model == "mock":
        return generate_mock_audio(psmiles)

    p2a, enc = model
    device = next(p2a.parameters()).device

    fp_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    # Autoregressive generation
    generated = [1]
    with torch.no_grad():
        for _ in range(500):
            tgt = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
            logits = p2a(fp_tensor, tgt)
            next_token = torch.argmax(logits[0, -1, :]).item()
            if next_token == 0:
                break
            generated.append(next_token)

    # Decode tokens → WAV via EnCodec
    pure_tokens = generated[1:]
    converted = [max(0, min(1023, t - 1)) for t in pure_tokens]

    CODEBOOKS = 8
    if len(converted) % CODEBOOKS != 0:
        converted = converted[: (len(converted) // CODEBOOKS) * CODEBOOKS]

    if len(converted) == 0:
        return generate_mock_audio(psmiles)

    num_steps = len(converted) // CODEBOOKS
    code_tensor = (
        torch.tensor(converted, dtype=torch.int64)
        .to(device)
        .view(num_steps, CODEBOOKS)
        .transpose(0, 1)
        .unsqueeze(0)
        .contiguous()
    )

    with torch.no_grad():
        audio = enc.decode([(code_tensor, None)]).detach().cpu()

    buf = io.BytesIO()
    torchaudio.save(buf, audio.squeeze(0), sample_rate=enc.sample_rate, format="wav")
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Waveform plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_waveform(audio_bytes: bytes) -> plt.Figure:
    """
    Decode audio bytes (WAV; MP3 via pydub if available) and return a
    dark-themed matplotlib Figure showing amplitude over time.
    """
    samples: Optional[np.ndarray] = None
    sr = SAMPLE_RATE

    try:
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, "rb") as wf:
            sr    = wf.getframerate()
            n_ch  = wf.getnchannels()
            raw   = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if n_ch > 1:
                samples = samples.reshape(-1, n_ch).mean(axis=1)
            samples /= 32_768.0
    except Exception:
        pass

    if samples is None:
        try:
            from pydub import AudioSegment  # type: ignore
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            sr  = seg.frame_rate
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            samples /= float(2 ** (8 * seg.sample_width - 1))
        except Exception:
            samples = np.zeros(sr * 3, dtype=np.float32)

    t = np.linspace(0, len(samples) / sr, len(samples))
    BG = "#0e1117"
    fig, ax = plt.subplots(figsize=(10, 2.6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(t, samples, color="#00c0f0", linewidth=0.55, alpha=0.95)
    ax.fill_between(t, samples, alpha=0.18, color="#00c0f0")
    ax.axhline(0, color="#2a3a4a", linewidth=0.5)
    ax.set_xlabel("Time (s)", color="#6b7888", fontsize=8)
    ax.set_ylabel("Amplitude", color="#6b7888", fontsize=8)
    ax.tick_params(colors="#6b7888", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1f2733")
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout(pad=0.6)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Input section — Ketcher draw + manual PSMILES, with real-time viz
# ──────────────────────────────────────────────────────────────────────────────
def render_input_section() -> str:
    """
    Render the polymer input UI (radio selector + Ketcher or text area on the
    left, real-time structural visualisation on the right).

    Returns the raw PSMILES/SMILES string entered by the user, or "" if empty.
    """
    input_method = st.radio(
        "Choose PSMILES Input Method:",
        ("Draw with Ketcher", "Enter PSMILES Manually"),
        horizontal=True,
        key="input_method_selector",
    )

    psmiles_input_value = ""

    col_editor, col_viz = st.columns([0.6, 0.4])

    with col_editor:
        if input_method == "Draw with Ketcher":
            if not KETCHER_AVAILABLE:
                st.error(
                    "streamlit-ketcher is not installed. "
                    "Run: `pip install streamlit-ketcher`"
                )
            else:
                st.write(
                    "Draw your polymer structure with two `*` atoms to define "
                    "the repeating unit."
                )
                ketcher_default = st.session_state.get("example_psmiles_for_input", "")
                # Preserve whatever the user last drew if no new example was loaded
                if (
                    not ketcher_default
                    and "ketcher_editor" in st.session_state
                    and st.session_state.ketcher_editor
                ):
                    ketcher_default = st.session_state.ketcher_editor

                psmiles_input_value = st_ketcher(
                    ketcher_default,
                    key="ketcher_editor",
                )

        else:  # manual text area
            st.write(
                "Paste your PSMILES string below. "
                "It must contain exactly two `*` symbols marking the repeat unit."
            )
            manual_default = st.session_state.get("example_psmiles_for_input", "")
            if (
                not manual_default
                and "manual_psmiles_text_area" in st.session_state
                and st.session_state.manual_psmiles_text_area
            ):
                manual_default = st.session_state.manual_psmiles_text_area

            psmiles_input_value = st.text_area(
                "Enter PSMILES:",
                value=manual_default,
                height=100,
                key="manual_psmiles_text_area",
            )

    # ── Real-time validation & session state update ──────────────────────────
    if psmiles_input_value and psmiles_input_value != st.session_state.rt_last_input:
        st.session_state.rt_last_input = psmiles_input_value
        ps_obj, mol, err = validate_psmiles(psmiles_input_value)
        if err:
            st.session_state.rt_mol         = None
            st.session_state.rt_psmiles_str = ""
            st.error(f"Validation error: {err}")
        else:
            st.session_state.rt_mol         = mol
            st.session_state.rt_psmiles_str = str(ps_obj) if ps_obj else psmiles_input_value
    elif not psmiles_input_value:
        st.session_state.rt_mol         = None
        st.session_state.rt_psmiles_str = ""
        st.session_state.rt_last_input  = ""

    # ── Right column: structural visualisation ───────────────────────────────
    with col_viz:
        st.subheader("Structural Preview")

        if st.session_state.rt_mol is not None:
            svg = mol_to_svg(st.session_state.rt_mol, width=300, height=280)
            if svg:
                st.image(svg, width=300)
            else:
                st.warning("Could not render structure image.")
            st.markdown(
                f"**Canonical PSMILES:** `{st.session_state.rt_psmiles_str}`"
            )
        elif st.session_state.rt_last_input:
            st.warning("Check PSMILES validity — see error above.")
        else:
            st.info("Draw or enter a PSMILES to see the structure here.")

        # Show library availability hints
        if not RDKIT_AVAILABLE:
            st.caption("⚠ Install `rdkit` for structure rendering.")
        if not PSMILES_AVAILABLE:
            st.caption("⚠ Install `psmiles` for canonical PSMILES output.")

    st.markdown(
        f"**Current input:** `{psmiles_input_value}`"
        if psmiles_input_value
        else "_Waiting for PSMILES input…_"
    )

    return psmiles_input_value


# ──────────────────────────────────────────────────────────────────────────────
# Shared HTML style block (injected into every st.components.v1.html panel)
# ──────────────────────────────────────────────────────────────────────────────
_PANEL_STYLE = """
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, sans-serif;
    color: #c9d1d9;
  }
  .eyebrow {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; margin-bottom: 14px;
  }
  h1 { font-size: clamp(1.8rem, 5vw, 3rem); font-weight: 800; line-height: 1.1;
       color: #e6edf3; margin-bottom: 22px; letter-spacing: -0.5px; }
  h1 em { color: #00c0f0; font-style: normal; }
  h2 { font-size: clamp(1.3rem, 4vw, 2rem); font-weight: 800; line-height: 1.15;
       color: #e6edf3; margin-bottom: 18px; letter-spacing: -0.3px; }
  p.body { font-size: clamp(0.85rem, 2.5vw, 1rem); line-height: 1.85;
           color: #8b95a5; margin-bottom: 14px; }
  .img-box {
    width: 100%; height: 100%;
    border: 2px dashed #1f6feb55; border-radius: 14px;
    background: linear-gradient(135deg, #0d2a5c18 0%, #0a162820 100%);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 10px;
    color: #30363d;
  }
  .img-box .icon { font-size: 2.8rem; opacity: 0.5; }
  .img-box .label { font-size: 0.7rem; letter-spacing: 2px;
                    text-transform: uppercase; opacity: 0.6; }
  /* ── Responsive helpers ── */
  .row {
    display: flex; gap: 60px; align-items: center; width: 100%;
  }
  .row > * { flex: 1; min-width: 0; }
  @media (max-width: 600px) {
    .row { flex-direction: column; gap: 28px; }
    .row-reverse { flex-direction: column-reverse !important; }
  }
</style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — Hero: text left, image right
# ──────────────────────────────────────────────────────────────────────────────
def render_hero_panel() -> None:
    _hero_img_path = Path("images/group_kv.jpeg")
    if _hero_img_path.exists():
        _hero_b64 = _compress_image(str(_hero_img_path), max_px=900, quality=70)
        _hero_html = f'<img src="data:image/jpeg;base64,{_hero_b64}" style="width:100%;height:100%;object-fit:cover;border-radius:16px;" />'
    else:
        _hero_html = '<div class="img-box"><div class="icon">🖼</div><div class="label">Add images/group_kv.jpeg</div></div>'

    st.components.v1.html(
        _PANEL_STYLE + f"""
        <body style="background:linear-gradient(150deg,#0a0d14 0%,#0c1a2e 100%);
                     padding: clamp(24px,5vw,60px) clamp(16px,4vw,48px);">
          <div class="row">
            <div>
              <p class="eyebrow" style="color:#00c0f0;">Polymer Sonification</p>
              <h1>What if<br>polymers could<br><em>sing</em>?</h1>
              <p class="body">
                Every polymer carries a unique sequence of atoms in its repeat unit —
                a molecular fingerprint that defines its properties.
                We turn that fingerprint into sound.
              </p>
              <p class="body">
                Scroll down to meet the team, learn the story,
                and then draw your own polymer to hear it.
              </p>
            </div>
            <div style="overflow:hidden;border-radius:16px;">
              {_hero_html}
            </div>
          </div>
        </body>
        """,
        height=680,
        scrolling=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — Team: 10 image holder spaces in a 5×2 grid
# ──────────────────────────────────────────────────────────────────────────────

# TODO: Replace placeholder entries with real names, roles, and image paths.
TEAM_MEMBERS: list[dict] = [
    {"name": "Chris",   "role": "Role / Affiliation", "img": "images/chris.jpeg"},
    {"name": "Mashid",  "role": "Role / Affiliation", "img": "images/mashid.jpeg"},
    {"name": "Ibra",    "role": "Role / Affiliation", "img": "images/ibra.jpeg"},
    {"name": "Henri",   "role": "Role / Affiliation", "img": "images/henri.jpeg"},
    {"name": "Kaushik", "role": "Role / Affiliation", "img": "images/kaushik.jpeg"},
    {"name": "Krishna", "role": "Role / Affiliation", "img": "images/krishna.jpeg"},
    {"name": "Lukas",   "role": "Role / Affiliation", "img": "images/lukas.jpeg"},
    {"name": "Niklas",  "role": "Role / Affiliation", "img": "images/niklas.jpeg"},
    {"name": "Rayan",   "role": "Role / Affiliation", "img": "images/rayan.jpeg"},
    {"name": "Subhash", "role": "Role / Affiliation", "img": "images/subhash.jpeg"},
]


def _member_card(name: str, role: str, img_b64: str, ext: str, idx: int) -> str:
    delay = idx * 80
    photo = f'<img src="data:image/{ext};base64,{img_b64}" style="width:100%;height:100%;object-fit:cover;display:block;" />'
    return f"""
    <div style="border-radius:14px;background:#161b2e;border:1px solid #2d1b69;
                overflow:hidden;opacity:0;
                animation:fadeUp 0.5s cubic-bezier(0.23,1,0.32,1) {delay}ms forwards;
                transition:box-shadow 0.2s;">
      <div style="width:100%;height:240px;overflow:hidden;border-radius:12px 12px 0 0;">{photo}</div>
      <div style="padding:8px 6px 10px;text-align:center;">
        <div style="font-size:0.82rem;font-weight:700;color:#e6edf3;">{name}</div>
        <div style="font-size:0.72rem;color:#8b95a5;margin-top:2px;">{role}</div>
      </div>
    </div>
    """


def _compress_image(img_path: str, max_px: int = 400, quality: int = 60) -> str:
    """Return a base64 JPEG string resized to max_px and compressed to quality."""
    import base64
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def render_team_panel() -> None:
    import random

    members = TEAM_MEMBERS.copy()
    random.shuffle(members)

    cards_html = ""
    for idx, m in enumerate(members):
        img_path = m.get("img", "")
        if img_path and Path(img_path).exists():
            b64 = _compress_image(img_path)
        else:
            b64 = ""
        cards_html += _member_card(m["name"], m["role"], b64, "jpeg", idx)

    # Worst-case height = mobile layout (2 cols → more rows)
    card_h     = 240 + 52
    gap        = 20
    header_h   = 160
    padding    = 100
    n_mobile   = math.ceil(len(members) / 2)   # 2-col mobile rows
    panel_h    = header_h + n_mobile * card_h + (n_mobile - 1) * gap + padding

    st.components.v1.html(
        _PANEL_STYLE + f"""
        <style>
          @keyframes fadeUp {{
            from {{ opacity:0; transform:translateY(16px); }}
            to   {{ opacity:1; transform:translateY(0);    }}
          }}
          .team-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 20px;
          }}
          @media (max-width: 600px) {{
            .team-grid {{ grid-template-columns: repeat(2, 1fr); gap: 12px; }}
          }}
        </style>

        <body style="background:#0d1117; padding:clamp(24px,5vw,60px) clamp(16px,4vw,48px);">
          <div style="text-align:center; margin-bottom:clamp(24px,4vw,48px);">
            <p class="eyebrow" style="color:#00c0f0;">The People</p>
            <h2>Meet the Team</h2>
            <p class="body" style="max-width:520px;margin:10px auto 0;">
              The researchers and engineers behind Polymer Sonification.
            </p>
          </div>
          <div class="team-grid">
            {cards_html}
          </div>
          <script>
            /* Directly resize the iframe to its content — works on both desktop and mobile */
            function fitFrame() {{
              try {{
                const h = document.body.scrollHeight + 20;
                if (window.frameElement) window.frameElement.style.height = h + 'px';
              }} catch(e) {{}}
            }}
            fitFrame();
            window.addEventListener('load', fitFrame);
            new ResizeObserver(fitFrame).observe(document.body);
          </script>
        </body>
        """,
        height=panel_h,
        scrolling=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — About: image left, text right, CTA button at bottom
# ──────────────────────────────────────────────────────────────────────────────
def render_about_panel() -> None:
    st.components.v1.html(
        _PANEL_STYLE + """
        <body style="background:linear-gradient(160deg,#0a1628 0%,#0d1117 100%);
                     padding: clamp(24px,5vw,60px) clamp(16px,4vw,48px);">
          <div class="row row-reverse">
            <div>
              <div class="img-box" style="height:280px;">
                <div class="icon">🔬</div>
                <div class="label">Add assets/about.png</div>
              </div>
            </div>
            <div>
              <p class="eyebrow" style="color:#7ee787;">The Project</p>
              <h2>Turning chemistry<br>into music</h2>
              <p class="body">
                We started this project with a curiosity: if every polymer has a unique
                molecular structure, could that structure produce a unique sound?
              </p>
              <p class="body">
                Using a pre-trained machine-learning model on polymer data, we map each
                repeat unit's atoms to musical notes across two octaves — heavier atoms
                play lower notes, lighter atoms play higher ones.
              </p>
              <p class="body">
                The result is a sonic fingerprint: an audio identity that is as unique
                to a polymer as its SMILES string.
              </p>
            </div>
          </div>
        </body>
        """,
        height=560,
        scrolling=False,
    )


def render_reverse_panel() -> None:
    st.components.v1.html(
        _PANEL_STYLE + """
        <body style="background:linear-gradient(160deg,#0d1117 0%,#0a1628 100%);
                     padding: clamp(24px,5vw,60px) clamp(16px,4vw,48px);">
          <div class="row">
            <div>
              <p class="eyebrow" style="color:#d2a8ff;">Coming Soon</p>
              <h2>Turning music<br>into chemistry</h2>
              <p class="body">
                The inverse challenge: can a melody describe a polymer?
                Given an audio fingerprint, we aim to reconstruct the molecular
                structure that would have produced it.
              </p>
              <p class="body">
                By reversing the sonification pipeline we open a new design space —
                composing a sound and letting the model propose a polymer that matches it.
              </p>
              <p class="body">
                This feature is under development. Upload an audio file below to try it.
              </p>
            </div>
            <div>
              <div class="img-box" style="height:280px;border-color:#d2a8ff44;">
                <div class="icon">🎵</div>
                <div class="label">Add assets/reverse.png</div>
              </div>
            </div>
          </div>
        </body>
        """,
        height=560,
        scrolling=False,
    )


def render_reverse_lab_section() -> None:
    """Render the reverse-design tool: upload audio → model → PSMILES → structure."""
    st.markdown(
        '<p style="font-size:0.7rem;font-weight:700;letter-spacing:3px;color:#d2a8ff;'
        'text-transform:uppercase;margin-bottom:6px;">Reverse Design</p>'
        '<h2 style="font-size:1.9rem;font-weight:800;color:#e6edf3;margin-bottom:6px;">'
        'Upload your audio</h2>'
        '<p style="font-size:0.95rem;color:#8b95a5;margin-bottom:20px;">'
        'Upload a WAV or MP3 file. The model will predict the polymer structure '
        'that matches the audio fingerprint.</p>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose a WAV or MP3 file",
        type=["wav", "mp3"],
        key="reverse_audio_upload",
    )

    if uploaded is not None:
        st.audio(uploaded, format=uploaded.type)
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        col_pred, col_reset = st.columns([3, 1])
        with col_pred:
            predict_clicked = st.button(
                "🔬 Predict Structure",
                type="primary",
                use_container_width=True,
                key="reverse_predict_btn",
            )
        with col_reset:
            if st.button("↺ Reset", use_container_width=True, key="reverse_reset_btn"):
                st.session_state.reverse_psmiles = ""
                st.session_state.reverse_mol     = None
                st.rerun()

        if predict_clicked:
            with st.spinner("Running reverse model… (placeholder)"):
                # ── TODO: replace with real reverse model call ──────────────
                # audio_bytes = uploaded.read()
                # result_psmiles = reverse_model.predict(audio_bytes)
                # For now return a placeholder PSMILES so the UI can be tested.
                result_psmiles = "[*]CC[*]"
                # ────────────────────────────────────────────────────────────

            st.session_state.reverse_psmiles = result_psmiles
            _, mol, err = validate_psmiles(result_psmiles)
            st.session_state.reverse_mol = mol if not err else None

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.session_state.reverse_psmiles:
        st.markdown(
            '<p style="font-size:0.7rem;font-weight:700;letter-spacing:3px;color:#d2a8ff;'
            'text-transform:uppercase;margin-bottom:10px;">Predicted Structure</p>',
            unsafe_allow_html=True,
        )
        col_info, col_img = st.columns([1, 1])
        with col_info:
            st.markdown(
                f'<p style="font-size:0.85rem;color:#8b95a5;margin-bottom:8px;">PSMILES</p>'
                f'<div class="psmiles-box">{st.session_state.reverse_psmiles}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            st.caption("Model output is a placeholder until the reverse model is trained.")
        with col_img:
            if st.session_state.reverse_mol is not None:
                svg = mol_to_svg(st.session_state.reverse_mol, width=300, height=260)
                if svg:
                    st.image(svg, width=300)
            else:
                st.info("Could not render structure — install `rdkit` for visualisation.")


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 — Lab: Streamlit input widgets
# ──────────────────────────────────────────────────────────────────────────────
def render_lab_section(model, is_mock: bool) -> bool:
    """Renders the input panel. Returns True if Sonify was clicked."""
    badge = (
        '<span class="badge-mock">Mock model</span>'
        if is_mock
        else '<span class="badge-real">Model loaded</span>'
    )
    st.markdown(
        f'<p style="font-size:0.7rem;font-weight:700;letter-spacing:3px;color:#00c0f0;'
        f'text-transform:uppercase;margin-bottom:6px;">The Lab</p>'
        f'<h2 style="font-size:1.9rem;font-weight:800;color:#e6edf3;margin-bottom:6px;">'
        f'Draw your polymer &nbsp;{badge}</h2>'
        f'<p style="font-size:0.95rem;color:#8b95a5;margin-bottom:20px;">'
        f'Use the Ketcher editor or paste a PSMILES string. '
        f'Mark the repeat unit with two <code style="color:#7ee787">*</code> atoms.</p>',
        unsafe_allow_html=True,
    )

    if not KETCHER_AVAILABLE:
        st.warning("**streamlit-ketcher** not installed — `pip install streamlit-ketcher`")
    if not RDKIT_AVAILABLE:
        st.warning("**rdkit** not installed — `pip install rdkit`")

    raw_input = render_input_section()

    if st.session_state.rt_psmiles_str:
        st.session_state.psmiles = st.session_state.rt_psmiles_str
    elif raw_input:
        st.session_state.psmiles = raw_input.strip()
    st.session_state.example_psmiles_for_input = ""

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_predict, col_example, col_reset = st.columns([2, 2, 1])

    with col_predict:
        predict_clicked = st.button(
            "🎵  Sonify",
            disabled=not bool(st.session_state.psmiles),
            type="primary",
            use_container_width=True,
        )
    with col_example:
        if st.button("Load example  ([*]CC[*])", use_container_width=True):
            st.session_state.example_psmiles_for_input = EXAMPLE_PSMILES
            st.session_state.psmiles     = EXAMPLE_PSMILES
            st.session_state.audio_bytes = None
            st.session_state.predicted   = False
            st.rerun()
    with col_reset:
        if st.button("↺ Reset", use_container_width=True):
            _reset()
            st.rerun()

    return predict_clicked


# ──────────────────────────────────────────────────────────────────────────────
# Section 5 — Results
# ──────────────────────────────────────────────────────────────────────────────
def render_audio_visualizer(audio_bytes: bytes) -> None:
    """
    Oscilloscope-style waveform: X = time (within the analysis window),
    Y = amplitude (-1 → +1), animated in real-time via Web Audio API.
    """
    import base64
    fmt = "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/mpeg"
    b64 = base64.b64encode(audio_bytes).decode()
    html = f"""
    <style>
      #viz-wrap {{
        background: #080c12;
        border-radius: 14px;
        padding: 0;
        border: 1px solid #1a2a3a;
        overflow: hidden;
      }}
      #viz-canvas {{
        width: 100%;
        height: 160px;
        display: block;
        background: #080c12;
      }}
      #viz-controls {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 16px 12px;
        background: #0c1520;
        border-top: 1px solid #1a2a3a;
      }}
      #play-btn {{
        background: #00c0f0;
        border: none;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        cursor: pointer;
        font-size: 14px;
        color: #080c12;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s;
        font-weight: bold;
      }}
      #play-btn:hover {{ background: #33d6ff; }}
      #seek-bar {{
        flex: 1;
        -webkit-appearance: none;
        height: 3px;
        border-radius: 2px;
        background: #1f2d3d;
        outline: none;
        cursor: pointer;
      }}
      #seek-bar::-webkit-slider-thumb {{
        -webkit-appearance: none;
        width: 11px;
        height: 11px;
        border-radius: 50%;
        background: #00c0f0;
        cursor: pointer;
      }}
      #time-label {{
        font-size: 10px;
        color: #4a6070;
        font-family: monospace;
        flex-shrink: 0;
        min-width: 68px;
        text-align: right;
        letter-spacing: 0.5px;
      }}
    </style>

    <div id="viz-wrap">
      <canvas id="viz-canvas"></canvas>
      <div id="viz-controls">
        <button id="play-btn">&#9654;</button>
        <input type="range" id="seek-bar" value="0" min="0" step="0.01">
        <span id="time-label">0:00 / 0:00</span>
      </div>
    </div>

    <script>
    (function() {{
      const b64   = "{b64}";
      const mime  = "{fmt}";
      const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
      const blob  = new Blob([bytes], {{type: mime}});
      const url   = URL.createObjectURL(blob);

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      let ctx, analyser, source, buf;
      let playing   = false;
      let startedAt = 0;
      let pausedAt  = 0;
      let animId    = null;

      const canvas     = document.getElementById("viz-canvas");
      const cc         = canvas.getContext("2d");
      const playBtn    = document.getElementById("play-btn");
      const seekBar    = document.getElementById("seek-bar");
      const timeLabel  = document.getElementById("time-label");

      function resizeCanvas() {{
        canvas.width  = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
      }}
      resizeCanvas();
      window.addEventListener("resize", resizeCanvas);

      // Decode audio once
      fetch(url)
        .then(r => r.arrayBuffer())
        .then(ab => {{
          ctx = new AudioCtx();
          return ctx.decodeAudioData(ab);
        }})
        .then(decoded => {{
          buf = decoded;
          seekBar.max = buf.duration.toFixed(2);
          timeLabel.textContent = "0:00 / " + fmtT(buf.duration);
          drawIdle();
        }});

      function fmtT(s) {{
        const m  = Math.floor(s / 60);
        const ss = Math.floor(s % 60).toString().padStart(2, "0");
        return m + ":" + ss;
      }}

      function startPlay(from) {{
        if (!buf) return;
        if (ctx.state === "suspended") ctx.resume();

        analyser = ctx.createAnalyser();
        analyser.fftSize = 2048;           // large buffer → smooth oscilloscope
        analyser.smoothingTimeConstant = 0.6;

        source = ctx.createBufferSource();
        source.buffer = buf;
        source.connect(analyser);
        analyser.connect(ctx.destination);
        source.start(0, from);
        startedAt = ctx.currentTime - from;
        playing = true;
        playBtn.innerHTML = "&#9646;&#9646;";

        source.onended = () => {{
          if (playing) {{
            playing  = false;
            pausedAt = 0;
            playBtn.innerHTML = "&#9654;";
            seekBar.value = 0;
            timeLabel.textContent = "0:00 / " + fmtT(buf.duration);
            cancelAnimationFrame(animId);
            drawIdle();
          }}
        }};
        drawLoop();
      }}

      function stopPlay() {{
        if (!source) return;
        source.onended = null;
        source.stop();
        pausedAt = ctx.currentTime - startedAt;
        playing  = false;
        playBtn.innerHTML = "&#9654;";
        cancelAnimationFrame(animId);
      }}

      playBtn.addEventListener("click", () => {{
        if (!buf) return;
        playing ? stopPlay() : startPlay(parseFloat(pausedAt) || 0);
      }});

      seekBar.addEventListener("input", () => {{
        const t = parseFloat(seekBar.value);
        if (playing) {{ stopPlay(); startPlay(t); }}
        else {{ pausedAt = t; }}
        timeLabel.textContent = fmtT(t) + " / " + fmtT(buf ? buf.duration : 0);
      }});

      // ── Oscilloscope drawing ──────────────────────────────────────────────
      function drawLoop() {{
        animId = requestAnimationFrame(drawLoop);

        if (playing && buf) {{
          const elapsed = ctx.currentTime - startedAt;
          seekBar.value = Math.min(elapsed, buf.duration).toFixed(2);
          timeLabel.textContent = fmtT(elapsed) + " / " + fmtT(buf.duration);
        }}

        const W = canvas.width, H = canvas.height;
        const bufLen = analyser.fftSize;
        const data   = new Float32Array(bufLen);   // -1.0 to +1.0
        analyser.getFloatTimeDomainData(data);

        cc.clearRect(0, 0, W, H);

        // Grid lines: centre + amplitude guides at ±0.5
        cc.strokeStyle = "#0f1e2e";
        cc.lineWidth   = 1;
        [[0.5], [0], [-0.5]].forEach(([norm]) => {{
          const y = H / 2 - norm * (H / 2) * 0.88;
          cc.beginPath(); cc.moveTo(0, y); cc.lineTo(W, y); cc.stroke();
        }});

        // Axis labels
        cc.fillStyle  = "#253545";
        cc.font       = "9px monospace";
        cc.fillText("+1", 4, H * 0.5 - H * 0.44 - 2);
        cc.fillText(" 0", 4, H / 2 + 4);
        cc.fillText("-1", 4, H * 0.5 + H * 0.44 + 11);

        // Waveform path
        const grad = cc.createLinearGradient(0, 0, W, 0);
        grad.addColorStop(0,    "#0070c0");
        grad.addColorStop(0.35, "#00c0f0");
        grad.addColorStop(0.65, "#00c0f0");
        grad.addColorStop(1,    "#0070c0");

        cc.beginPath();
        cc.strokeStyle = grad;
        cc.lineWidth   = 2;
        cc.lineJoin    = "round";

        const step = W / bufLen;
        for (let i = 0; i < bufLen; i++) {{
          const x = i * step;
          const y = H / 2 - data[i] * (H / 2) * 0.88;
          i === 0 ? cc.moveTo(x, y) : cc.lineTo(x, y);
        }}
        cc.stroke();

        // Glow pass — same path, wider & transparent
        cc.beginPath();
        cc.strokeStyle = "rgba(0, 192, 240, 0.15)";
        cc.lineWidth   = 6;
        for (let i = 0; i < bufLen; i++) {{
          const x = i * step;
          const y = H / 2 - data[i] * (H / 2) * 0.88;
          i === 0 ? cc.moveTo(x, y) : cc.lineTo(x, y);
        }}
        cc.stroke();
      }}

      function drawIdle() {{
        const W = canvas.width, H = canvas.height;
        cc.clearRect(0, 0, W, H);

        // Grid
        cc.strokeStyle = "#0f1e2e";
        cc.lineWidth   = 1;
        [[0.5], [0], [-0.5]].forEach(([norm]) => {{
          const y = H / 2 - norm * (H / 2) * 0.88;
          cc.beginPath(); cc.moveTo(0, y); cc.lineTo(W, y); cc.stroke();
        }});

        // Flat centre line
        cc.beginPath();
        cc.strokeStyle = "#1a3a50";
        cc.lineWidth   = 2;
        cc.moveTo(0, H / 2);
        cc.lineTo(W, H / 2);
        cc.stroke();

        // Labels
        cc.fillStyle = "#253545";
        cc.font      = "9px monospace";
        cc.fillText("+1", 4, H * 0.5 - H * 0.44 - 2);
        cc.fillText(" 0", 4, H / 2 + 4);
        cc.fillText("-1", 4, H * 0.5 + H * 0.44 + 11);
      }}

    }})();
    </script>
    """
    st.components.v1.html(html, height=220)


def render_result_section(audio_bytes: bytes, psmiles: str, is_mock: bool) -> None:
    st.markdown(
        '<p style="font-size:0.7rem;font-weight:700;letter-spacing:3px;color:#7ee787;'
        'text-transform:uppercase;margin-bottom:6px;">The Sound</p>'
        '<h2 style="font-size:1.9rem;font-weight:800;color:#e6edf3;margin-bottom:6px;">'
        'Your polymer has a voice.</h2>'
        '<p style="font-size:0.95rem;color:#8b95a5;margin-bottom:20px;">'
        'Each atom in the repeat unit was mapped to a musical note and concatenated.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.markdown(
        f'<p class="section-label">Polymer</p>'
        f'<div class="psmiles-box">{psmiles}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-label">Playback & Visualiser</p>', unsafe_allow_html=True)
    render_audio_visualizer(audio_bytes)

    fmt = "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/mpeg"
    ext = "wav" if fmt == "audio/wav" else "mp3"
    st.download_button(
        label="⬇ Download audio",
        data=audio_bytes,
        file_name=f"polymer_sound.{ext}",
        mime=fmt,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Debug / details", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**PSMILES**")
            st.code(psmiles, language="text")
        with c2:
            st.markdown("**Model**")
            st.code(str(MODEL_PATH) + ("\n[mock]" if is_mock else "\n[loaded]"), language="text")
        st.code(f"{len(audio_bytes):,} bytes  |  {fmt}", language="text")


# ──────────────────────────────────────────────────────────────────────────────
# Main — five-section scroll story
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    model   = load_model()
    is_mock = model == "mock"

    # ── 1. Hero ───────────────────────────────────────────────────────────────
    render_hero_panel()

    # ── 2. Team ───────────────────────────────────────────────────────────────
    render_team_panel()

    # ── 3. About panel ────────────────────────────────────────────────────────
    render_about_panel()

    # ── Inline expand: show CTA button or the full lab tool ───────────────────
    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        if not st.session_state.show_lab:
            if st.button("🎵 Start Sonifying →", use_container_width=True, key="start_sonifying_btn"):
                st.session_state.show_lab = True
                st.session_state.just_expanded = True
                st.rerun()
        else:
            if st.button("✕ Close Tool", use_container_width=True, key="close_lab_btn"):
                st.session_state.show_lab = False
                st.rerun()

    if st.session_state.show_lab:
        predict_clicked = render_lab_section(model, is_mock)

        # ── Run inference ────────────────────────────────────────────────────
        if predict_clicked and st.session_state.psmiles:
            with st.spinner("Sonifying polymer…"):
                features    = preprocess(st.session_state.psmiles)
                audio_bytes = run_inference(model, features, st.session_state.psmiles)
            st.session_state.audio_bytes    = audio_bytes
            st.session_state.predicted      = True
            st.session_state.scroll_to_result = True

        # ── Results ──────────────────────────────────────────────────────────
        if st.session_state.predicted and st.session_state.audio_bytes is not None:
            st.markdown('<div id="section-result-top" style="height:40px;background:#0a0d14"></div>', unsafe_allow_html=True)
            if st.session_state.scroll_to_result:
                st.session_state.scroll_to_result = False
                st.components.v1.html(
                    "<script>"
                    "window.parent.document.getElementById('section-result-top')"
                    ".scrollIntoView({behavior:'smooth', block:'start'});"
                    "</script>",
                    height=0,
                )
            render_result_section(
                audio_bytes=st.session_state.audio_bytes,
                psmiles=st.session_state.psmiles,
                is_mock=is_mock,
            )

    st.markdown('<div style="height:60px;background:#0d1117"></div>', unsafe_allow_html=True)

    # ── 4. Reverse panel ─────────────────────────────────────────────────────
    render_reverse_panel()

    # ── Inline expand: reverse design tool ───────────────────────────────────
    col_l2, col_mid2, col_r2 = st.columns([1, 2, 1])
    with col_mid2:
        st.markdown(
            """
            <div style="
                width:100%; padding:10px 16px;
                border-radius:8px;
                border: 1px solid #3a2560;
                background: #1a0f35;
                color: #6a4a9a;
                font-size:0.88rem; font-weight:600;
                text-align:center;
                cursor:not-allowed;
                user-select:none;
                opacity:0.5;
                letter-spacing:0.3px;
            ">
                🔒 Try Reverse Design →
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.show_reverse_lab:
        st.markdown('<div id="section-reverse-top" style="height:32px;background:#0d1117"></div>', unsafe_allow_html=True)
        if st.session_state.just_expanded_reverse:
            st.session_state.just_expanded_reverse = False
            st.components.v1.html(
                "<script>"
                "window.parent.document.getElementById('section-reverse-top')"
                ".scrollIntoView({behavior:'smooth', block:'start'});"
                "</script>",
                height=0,
            )
        render_reverse_lab_section()

    st.markdown('<div style="height:80px;background:#0d1117"></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
