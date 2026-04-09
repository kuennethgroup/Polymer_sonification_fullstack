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
# Custom CSS — only for the lab / results Streamlit sections
# (story sections carry their own inline styles inside st.components.v1.html)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    html, body, .stApp { background: #0a0d14; color: #c9d1d9; }
    .block-container { padding-top: 0 !important; }

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
# Shared HTML style block (injected into every st.components.v1.html panel)
# ──────────────────────────────────────────────────────────────────────────────
_PANEL_STYLE = """
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
  h1 { font-size: 3rem; font-weight: 800; line-height: 1.1;
       color: #e6edf3; margin-bottom: 22px; letter-spacing: -0.5px; }
  h1 em { color: #00c0f0; font-style: normal; }
  h2 { font-size: 2rem; font-weight: 800; line-height: 1.15;
       color: #e6edf3; margin-bottom: 18px; letter-spacing: -0.3px; }
  p.body { font-size: 1rem; line-height: 1.85; color: #8b95a5; margin-bottom: 14px; }
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
</style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — Hero: text left, image right
# ──────────────────────────────────────────────────────────────────────────────
def render_hero_panel() -> None:
    st.components.v1.html(
        _PANEL_STYLE + """
        <body style="background:linear-gradient(150deg,#0a0d14 0%,#0c1a2e 100%);
                     min-height:680px; display:flex; align-items:center; padding:60px 48px;">
          <div style="display:flex; gap:60px; align-items:center; width:100%;">

            <!-- Left: text -->
            <div style="flex:1; min-width:0;">
              <p class="eyebrow" style="color:#00c0f0;">Polymer Sonification</p>
              <h1>What if<br>polymers could<br><em>speak</em>?</h1>
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

            <!-- Right: image placeholder -->
            <div style="flex:1; min-width:0; height:400px;">
              <div class="img-box">
                <div class="icon">🖼</div>
                <div class="label">Add assets/hero.png</div>
              </div>
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
    {"name": "Member 1", "role": "Role / Affiliation"},
    {"name": "Member 2", "role": "Role / Affiliation"},
    {"name": "Member 3", "role": "Role / Affiliation"},
    {"name": "Member 4", "role": "Role / Affiliation"},
    {"name": "Member 5", "role": "Role / Affiliation"},
    {"name": "Member 6", "role": "Role / Affiliation"},
    {"name": "Member 7", "role": "Role / Affiliation"},
    {"name": "Member 8", "role": "Role / Affiliation"},
    {"name": "Member 9", "role": "Role / Affiliation"},
    {"name": "Member 10", "role": "Role / Affiliation"},
]


def _member_card(name: str, role: str, img_path: Optional[str] = None) -> str:
    """Return the HTML snippet for one team member card."""
    import base64
    if img_path and Path(img_path).exists():
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = Path(img_path).suffix.lstrip(".")
        photo = f'<img src="data:image/{ext};base64,{b64}" style="width:100%;height:100%;object-fit:cover;border-radius:12px;" />'
    else:
        photo = '<div class="img-box" style="height:100%;border-radius:12px;"><div class="icon" style="font-size:2rem;">👤</div></div>'

    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;gap:10px;">
      <div style="width:100%;aspect-ratio:1;">{photo}</div>
      <div style="text-align:center;">
        <div style="font-size:0.82rem;font-weight:700;color:#e6edf3;">{name}</div>
        <div style="font-size:0.72rem;color:#8b95a5;margin-top:2px;">{role}</div>
      </div>
    </div>
    """


def render_team_panel() -> None:
    cards_html = "".join(
        _member_card(m["name"], m["role"], m.get("img"))
        for m in TEAM_MEMBERS
    )
    st.components.v1.html(
        _PANEL_STYLE + f"""
        <body style="background:#0d1117; padding:60px 48px; min-height:700px;">

          <!-- Header -->
          <div style="text-align:center; margin-bottom:48px;">
            <p class="eyebrow" style="color:#00c0f0;">The People</p>
            <h2>Meet the Team</h2>
            <p class="body" style="max-width:520px;margin:10px auto 0;">
              The researchers and engineers behind Polymer Sonification.
            </p>
          </div>

          <!-- 5-column grid, 2 rows = 10 slots -->
          <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:24px;">
            {cards_html}
          </div>

        </body>
        """,
        height=940,
        scrolling=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — About: image left, text right, CTA button at bottom
# ──────────────────────────────────────────────────────────────────────────────
def render_about_panel() -> None:
    st.components.v1.html(
        _PANEL_STYLE + """
        <body style="background:linear-gradient(160deg,#0a1628 0%,#0d1117 100%);
                     min-height:600px; display:flex; align-items:center; padding:60px 48px;">
          <div style="display:flex; gap:60px; align-items:center; width:100%;">

            <!-- Left: image placeholder -->
            <div style="flex:1; min-width:0; height:380px;">
              <div class="img-box" style="height:100%;">
                <div class="icon">🔬</div>
                <div class="label">Add assets/about.png</div>
              </div>
            </div>

            <!-- Right: text -->
            <div style="flex:1; min-width:0;">
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
        height=620,
        scrolling=False,
    )


def render_reverse_panel() -> None:
    st.components.v1.html(
        _PANEL_STYLE + """
        <body style="background:linear-gradient(160deg,#0d1117 0%,#0a1628 100%);
                     min-height:600px; display:flex; align-items:center; padding:60px 48px;">
          <div style="display:flex; gap:60px; align-items:center; width:100%;">

            <!-- Left: text -->
            <div style="flex:1; min-width:0;">
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
                This feature is under development. Stay tuned.
              </p>
            </div>

            <!-- Right: image placeholder -->
            <div style="flex:1; min-width:0; height:380px;">
              <div class="img-box" style="height:100%;border-color:#d2a8ff44;">
                <div class="icon">🎵</div>
                <div class="label">Add assets/reverse.png</div>
              </div>
            </div>

          </div>
        </body>
        """,
        height=620,
        scrolling=False,
    )


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

    st.markdown('<p class="section-label">Waveform</p>', unsafe_allow_html=True)
    with st.spinner("Rendering waveform…"):
        fig = plot_waveform(audio_bytes)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

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

    # ── 3. About ──────────────────────────────────────────────────────────────
    render_about_panel()
    render_reverse_panel()

    # ── CTA button (bottom of about → jumps to lab) ───────────────────────────
    st.markdown(
        '<div style="background:linear-gradient(160deg,#0a1628 0%,#0d1117 100%);"'
        ' id="cta-row">',
        unsafe_allow_html=True,
    )
    _, col_btn, _ = st.columns([2, 3, 2])
    with col_btn:
        go_clicked = st.button(
            "Start Sonifying  →",
            type="primary",
            use_container_width=True,
            key="cta_go",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="height:60px;background:#0d1117"></div>', unsafe_allow_html=True)

    # ── 4. Lab ────────────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div id="section-lab"></div>', unsafe_allow_html=True)
        predict_clicked = render_lab_section(model, is_mock)

    # scroll to lab when CTA is clicked
    if go_clicked:
        st.components.v1.html(
            "<script>window.parent.document.getElementById('section-lab')"
            ".scrollIntoView({behavior:'smooth',block:'start'});</script>",
            height=0,
        )

    # ── Run inference ──────────────────────────────────────────────────────────
    if predict_clicked and st.session_state.psmiles:
        with st.spinner("Sonifying polymer…"):
            features    = preprocess(st.session_state.psmiles)
            audio_bytes = run_inference(model, features, st.session_state.psmiles)
        st.session_state.audio_bytes = audio_bytes
        st.session_state.predicted   = True

    # ── 5. Results ────────────────────────────────────────────────────────────
    if st.session_state.predicted and st.session_state.audio_bytes is not None:
        st.markdown('<div style="height:40px;background:#0a0d14"></div>', unsafe_allow_html=True)
        st.markdown('<div id="section-result"></div>', unsafe_allow_html=True)
        render_result_section(
            audio_bytes=st.session_state.audio_bytes,
            psmiles=st.session_state.psmiles,
            is_mock=is_mock,
        )
        if predict_clicked:
            st.components.v1.html(
                "<script>window.parent.document.getElementById('section-result')"
                ".scrollIntoView({behavior:'smooth',block:'start'});</script>",
                height=0,
            )


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
