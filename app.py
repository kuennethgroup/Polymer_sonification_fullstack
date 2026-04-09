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
MODEL_PATH     = Path("model/model.pkl")
SAMPLE_RATE    = 22_050       # Hz — used for mock audio generation
MOCK_DURATION  = 3.0          # seconds
EXAMPLE_PSMILES = "[*]CC[*]"  # polyethylene repeat unit


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — story scroll layout
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── global ── */
    html, body, .stApp { background: #0a0d14; color: #c9d1d9; }

    /* hide default Streamlit top padding so sections butt up cleanly */
    .block-container { padding-top: 0 !important; max-width: 100% !important; }

    /* ── story section panels ── */
    .story-section {
        min-height: 100vh;
        display: flex; flex-direction: column; justify-content: center;
        padding: 5vh 8vw;
        scroll-margin-top: 0;
        position: relative;
    }
    .section-story  { background: linear-gradient(160deg, #0a0d14 0%, #0d1b2a 60%, #0a1628 100%); }
    .section-lab    { background: #0d1117; border-top: 1px solid #1f2733; }
    .section-result { background: linear-gradient(180deg, #0d1117 0%, #050d18 100%);
                      border-top: 1px solid #1f6feb33; }

    /* ── story section: hero image placeholder ── */
    .hero-image-wrap {
        width: 100%; max-width: 820px;
        aspect-ratio: 16/7;
        border-radius: 16px; overflow: hidden;
        border: 1px solid #1f6feb44;
        background: linear-gradient(135deg, #0d2a5c 0%, #0a1628 50%, #1a0d2e 100%);
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 2.5rem auto;
        position: relative;
    }
    .hero-image-wrap img {
        width: 100%; height: 100%; object-fit: cover; border-radius: 16px;
    }
    .hero-image-placeholder {
        color: #30363d; font-size: 1rem; letter-spacing: 2px; text-transform: uppercase;
    }

    /* ── story text ── */
    .story-eyebrow {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 3px; color: #00c0f0;
        text-transform: uppercase; margin-bottom: 1rem;
    }
    .story-headline {
        font-size: clamp(2rem, 4vw, 3.2rem); font-weight: 800; line-height: 1.15;
        color: #e6edf3; margin-bottom: 1.2rem; letter-spacing: -0.5px;
    }
    .story-body {
        font-size: 1.05rem; line-height: 1.85; color: #8b95a5;
        max-width: 640px;
    }
    .story-body p { margin-bottom: 1rem; }

    /* ── scroll-down cue ── */
    .scroll-cue {
        display: inline-flex; align-items: center; gap: 8px;
        color: #00c0f0; font-size: 0.85rem; font-weight: 600;
        text-decoration: none; margin-top: 2rem;
        padding: 10px 22px; border: 1px solid #00c0f044;
        border-radius: 30px; transition: background 0.2s;
    }
    .scroll-cue:hover { background: #00c0f011; }

    /* ── lab section header ── */
    .lab-eyebrow { font-size: 0.7rem; font-weight: 700; letter-spacing: 3px;
                   color: #00c0f0; text-transform: uppercase; margin-bottom: 0.5rem; }
    .lab-headline { font-size: clamp(1.6rem, 3vw, 2.4rem); font-weight: 800;
                    color: #e6edf3; margin-bottom: 0.4rem; }
    .lab-sub { font-size: 0.95rem; color: #8b95a5; margin-bottom: 1.8rem; }

    /* ── result section header ── */
    .result-eyebrow { font-size: 0.7rem; font-weight: 700; letter-spacing: 3px;
                      color: #7ee787; text-transform: uppercase; margin-bottom: 0.5rem; }
    .result-headline { font-size: clamp(1.6rem, 3vw, 2.4rem); font-weight: 800;
                       color: #e6edf3; margin-bottom: 0.4rem; }
    .result-sub { font-size: 0.95rem; color: #8b95a5; margin-bottom: 1.8rem; }

    /* ── misc reused components ── */
    .divider  { border-top: 1px solid #1f2733; margin: 1.2rem 0; }
    .section-label {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px;
        color: #4a6fa5; text-transform: uppercase; margin-bottom: 6px;
    }
    .psmiles-box {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 10px 14px;
        font-family: 'Courier New', monospace; font-size: 0.95rem;
        color: #7ee787; word-break: break-all;
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


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """
    Load the serialised ML model from MODEL_PATH.
    Returns the model object, or the string "mock" when the file is absent.
    """
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as fh:
            return pickle.load(fh)
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
    """
    Convert a PSMILES string into a fixed-length feature vector.

    TODO: Replace with real polymer featurisation, e.g.:
          - Morgan fingerprints via RDKit
          - Mordred descriptors
          - A custom polymer descriptor library

    Currently returns a seeded random vector so the rest of the pipeline
    can be exercised end-to-end without a cheminformatics dependency.
    """
    # TODO: replace with real featurisation
    rng = np.random.default_rng(abs(hash(psmiles)) % (2 ** 32))
    return rng.random(128).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Mock audio generation (stdlib only — no extra dependencies)
# ──────────────────────────────────────────────────────────────────────────────
def generate_mock_audio(psmiles: str) -> bytes:
    """
    Produce a deterministic multi-tone WAV waveform seeded from the PSMILES
    string.  Each unique polymer yields a distinct pitch combination.
    Returns raw WAV bytes.
    """
    rng = np.random.default_rng(abs(hash(psmiles)) % (2 ** 32))
    n_samples = int(SAMPLE_RATE * MOCK_DURATION)
    t = np.linspace(0, MOCK_DURATION, n_samples, endpoint=False)

    base_freq = 220.0 + rng.random() * 440.0
    signal = (
        0.60 * np.sin(2 * math.pi * base_freq * t)
        + 0.25 * np.sin(2 * math.pi * base_freq * 2 * t)
        + 0.10 * np.sin(2 * math.pi * base_freq * 3 * t)
        + 0.05 * np.sin(2 * math.pi * base_freq * 5 * t)
    )

    attack  = int(0.05 * SAMPLE_RATE)
    release = int(0.30 * SAMPLE_RATE)
    env = np.ones(n_samples)
    env[:attack]   = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    signal *= env

    signal = (signal / np.max(np.abs(signal)) * 32_767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(signal.tobytes())
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def run_inference(model, features: np.ndarray, psmiles: str) -> bytes:
    """
    Run the ML model and return audio bytes.

    TODO: Replace the stub call with the real model's API once trained.
    """
    if model == "mock":
        return generate_mock_audio(psmiles)

    # TODO: adapt to real model call signature
    audio_bytes: bytes = model.predict(features)  # type: ignore[union-attr]
    return audio_bytes


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
# JS helper — smooth-scroll the Streamlit viewport to a named element
# ──────────────────────────────────────────────────────────────────────────────
def _scroll_to(element_id: str) -> None:
    """
    Inject a tiny invisible iframe whose JS scrolls the parent Streamlit
    viewport to the element with the given id.  Height=0 keeps it invisible.
    """
    st.components.v1.html(
        f"""
        <script>
          (function() {{
            var el = window.parent.document.getElementById('{element_id}');
            if (el) {{
              el.scrollIntoView({{behavior: 'smooth', block: 'start'}});
            }}
          }})();
        </script>
        """,
        height=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — The Story
# ──────────────────────────────────────────────────────────────────────────────
def render_story_section(image_path: Optional[str] = None) -> None:
    """
    Full-viewport hero panel: project image + narrative.
    Replace image_path with the path to your actual project image once ready.
    """
    st.markdown('<div id="section-story" class="story-section section-story">', unsafe_allow_html=True)

    # ── Hero image ────────────────────────────────────────────────────────────
    if image_path and Path(image_path).exists():
        import base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        ext = Path(image_path).suffix.lstrip(".")
        st.markdown(
            f'<div class="hero-image-wrap"><img src="data:image/{ext};base64,{img_b64}" /></div>',
            unsafe_allow_html=True,
        )
    else:
        # Placeholder — swap in a real image by setting image_path above
        st.markdown(
            """
            <div class="hero-image-wrap">
              <span class="hero-image-placeholder">
                [ Project image — add assets/hero.png to replace this ]
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Narrative text ────────────────────────────────────────────────────────
    st.markdown(
        """
        <p class="story-eyebrow">The Origin Story</p>
        <h1 class="story-headline">What if polymers<br>could speak?</h1>
        <div class="story-body">
          <p>
            It started with a simple question in the lab: every polymer has a unique
            molecular fingerprint — a sequence of atoms arranged in a repeating chain
            that determines everything from tensile strength to conductivity.
            But fingerprints are silent.  What if we gave them a voice?
          </p>
          <p>
            We mapped each atom type to a musical note across two octaves —
            carbon to a warm middle C, oxygen to an open G, nitrogen to the
            concert A — and let the repeat unit of a polymer compose its own melody.
            The result is a sonic identity: two polymers that look almost identical
            on paper can sound strikingly different when you listen.
          </p>
          <p>
            This tool is the first step.  Draw a polymer, press play, and hear
            its structure for the very first time.
          </p>
        </div>
        <a class="scroll-cue" href="#section-lab">Begin in the lab &darr;</a>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — The Lab  (input + sonify button)
# ──────────────────────────────────────────────────────────────────────────────
def render_lab_section(model, is_mock: bool) -> bool:
    """
    Renders the input panel.  Returns True if the Sonify button was clicked.
    """
    badge = (
        '<span class="badge-mock">Mock model</span>'
        if is_mock
        else '<span class="badge-real">Model loaded</span>'
    )

    st.markdown(
        f"""
        <div id="section-lab" class="story-section section-lab">
          <p class="lab-eyebrow">Step 2 — The Lab</p>
          <h2 class="lab-headline">Draw your polymer {badge}</h2>
          <p class="lab-sub">
            Use the Ketcher editor to sketch a repeat unit with two
            <code style="color:#7ee787">*</code> attachment points,
            or paste a PSMILES string directly.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # The actual input widgets must live outside the raw HTML div so Streamlit
    # can render its own components inside.
    if not KETCHER_AVAILABLE:
        st.warning("**streamlit-ketcher** not installed — `pip install streamlit-ketcher`")
    if not RDKIT_AVAILABLE:
        st.warning("**rdkit** not installed — structure preview disabled — `pip install rdkit`")

    raw_input = render_input_section()

    # Commit PSMILES
    if st.session_state.rt_psmiles_str:
        st.session_state.psmiles = st.session_state.rt_psmiles_str
    elif raw_input:
        st.session_state.psmiles = raw_input.strip()
    st.session_state.example_psmiles_for_input = ""

    # Controls row
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
# Section 3 — The Sound  (results)
# ──────────────────────────────────────────────────────────────────────────────
def render_result_section(audio_bytes: bytes, psmiles: str, is_mock: bool) -> None:
    """Renders the audio result panel."""

    st.markdown(
        """
        <div id="section-result" class="story-section section-result">
          <p class="result-eyebrow">Step 3 — The Sound</p>
          <h2 class="result-headline">Your polymer has a voice.</h2>
          <p class="result-sub">
            Each atom in the repeat unit played its assigned note in sequence.
            What you hear below is the acoustic fingerprint of your polymer.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    # PSMILES recap
    st.markdown(
        f'<p class="section-label">Polymer</p>'
        f'<div class="psmiles-box">{psmiles}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # Waveform
    st.markdown('<p class="section-label">Waveform</p>', unsafe_allow_html=True)
    with st.spinner("Rendering waveform…"):
        fig = plot_waveform(audio_bytes)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Player + download
    st.markdown('<p class="section-label" style="margin-top:12px">Playback</p>', unsafe_allow_html=True)
    fmt = "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/mpeg"
    audio_col, dl_col = st.columns([3, 1])
    with audio_col:
        st.audio(audio_bytes, format=fmt)
    with dl_col:
        ext = "wav" if fmt == "audio/wav" else "mp3"
        st.download_button(
            label="⬇ Download",
            data=audio_bytes,
            file_name=f"polymer_sound.{ext}",
            mime=fmt,
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Debug expander
    with st.expander("Debug / details", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**PSMILES**")
            st.code(psmiles, language="text")
        with c2:
            st.markdown("**Model**")
            st.code(
                str(MODEL_PATH) + ("\n[mock]" if is_mock else "\n[loaded]"),
                language="text",
            )
        st.code(f"{len(audio_bytes):,} bytes  |  {fmt}", language="text")


# ──────────────────────────────────────────────────────────────────────────────
# Main — orchestrates the three story sections
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    model   = load_model()
    is_mock = model == "mock"

    # ── Section 1: Story ─────────────────────────────────────────────────────
    # Pass a path like "assets/hero.png" once you have a project image.
    render_story_section(image_path=None)

    # ── Section 2: Lab (input) ───────────────────────────────────────────────
    predict_clicked = render_lab_section(model, is_mock)

    # ── Run inference ─────────────────────────────────────────────────────────
    if predict_clicked and st.session_state.psmiles:
        with st.spinner("Sonifying polymer…"):
            features    = preprocess(st.session_state.psmiles)
            audio_bytes = run_inference(model, features, st.session_state.psmiles)
        st.session_state.audio_bytes = audio_bytes
        st.session_state.predicted   = True

    # ── Section 3: Sound (results) ───────────────────────────────────────────
    if st.session_state.predicted and st.session_state.audio_bytes is not None:
        render_result_section(
            audio_bytes=st.session_state.audio_bytes,
            psmiles=st.session_state.psmiles,
            is_mock=is_mock,
        )
        # Auto-scroll to results only right after the button was clicked
        if predict_clicked:
            _scroll_to("section-result")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
