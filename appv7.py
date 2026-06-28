"""
app.py — AudioForensics AI  (v6.0 — with MFCC Visualization)
Run:  streamlit run app.py

CHANGES from v5.0:
  - extract_mfcc() result is now cached and stored alongside mel
  - New "MFCC Feature Analysis" card rendered via plot_mfcc()
  - Optional raw matrix viewer (st.dataframe) behind a toggle
  - mfcc_img_path passed through to generate_report()
"""

import os
import io
import tempfile

import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model

# ── Local modules ──────────────────────────────────────────────────────────────
from databasev1 import (
    init_db, save_analysis, get_user_analyses, get_user_stats,
    get_analysis_by_id, delete_analysis, save_report, get_report, has_report,
    get_all_users, get_all_analyses, admin_delete_analysis, get_global_stats,
)
from authv1 import (
    signup_user, login_user, set_session, get_session,
    is_logged_in, is_admin, logout, validate_signup,
)
# ▼ plot_mfcc is new in utilsv3
from utilsv4 import extract_mfcc, extract_mel, plot_mel, plot_mfcc, transcribe_audio
from reportsv4 import generate_report

# ── DB bootstrap ───────────────────────────────────────────────────────────────
init_db()

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AudioForensics AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===============================
# GLOBAL CSS  (unchanged from v5)
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0d0f14; color: #e0e6f0; }
[data-testid="stSidebar"] { background-color: #111520; border-right: 1px solid #1e2535; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.mono { font-family: 'Space Mono', monospace; }

.hero-block {
    background: linear-gradient(135deg,#0d1b2e 0%,#0d0f14 60%,#130d1e 100%);
    border: 1px solid #1e2a40; border-radius: 16px;
    padding: 40px 48px; margin-bottom: 28px; position: relative; overflow: hidden;
}
.hero-block::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px;
    background:radial-gradient(circle,rgba(56,189,248,.08) 0%,transparent 70%);
    border-radius:50%;
}
.hero-title  { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700; color:#f0f6ff; margin:0 0 8px; }
.hero-sub    { font-size:.95rem; color:#7a90b0; font-weight:300; }
.hero-badge  {
    display:inline-block; background:rgba(56,189,248,.1); border:1px solid rgba(56,189,248,.25);
    color:#38bdf8; font-family:'Space Mono',monospace; font-size:.68rem;
    padding:3px 10px; border-radius:100px; margin-bottom:14px; letter-spacing:1px;
}
.hero-badge-admin {
    display:inline-block; background:rgba(251,191,36,.1); border:1px solid rgba(251,191,36,.3);
    color:#fbbf24; font-family:'Space Mono',monospace; font-size:.68rem;
    padding:3px 10px; border-radius:100px; margin-bottom:14px; letter-spacing:1px;
}

.card { background:#111520; border:1px solid #1e2535; border-radius:12px; padding:24px 28px; margin-bottom:20px; }
.card-title { font-family:'Space Mono',monospace; font-size:.72rem; letter-spacing:2px; text-transform:uppercase; color:#38bdf8; margin-bottom:14px; }

.info-row { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #1a2030; font-size:.88rem; }
.info-row:last-child { border-bottom:none; }
.info-label { color:#5a7090; font-weight:500; }
.info-value { color:#c8d8f0; font-family:'Space Mono',monospace; font-size:.80rem; }

.result-real { background:linear-gradient(135deg,#0a2418,#0d1e14); border:1.5px solid #22c55e; border-radius:12px; padding:28px 32px; text-align:center; }
.result-fake { background:linear-gradient(135deg,#2a0a0a,#1e0d0d); border:1.5px solid #ef4444; border-radius:12px; padding:28px 32px; text-align:center; }
.result-label { font-family:'Space Mono',monospace; font-size:.70rem; letter-spacing:3px; text-transform:uppercase; margin-bottom:10px; }
.result-real .result-label { color:#4ade80; }
.result-fake .result-label { color:#f87171; }
.result-verdict { font-family:'Space Mono',monospace; font-size:2.2rem; font-weight:700; letter-spacing:4px; }
.result-real .result-verdict { color:#22c55e; }
.result-fake .result-verdict { color:#ef4444; }

.conf-bar-wrap { margin:6px 0; }
.conf-bar-label { display:flex; justify-content:space-between; font-size:.80rem; margin-bottom:4px; color:#7a90b0; font-family:'Space Mono',monospace; }
.conf-bar-bg { background:#1a2030; border-radius:100px; height:8px; overflow:hidden; }
.conf-bar-fill-real { height:8px; border-radius:100px; background:linear-gradient(90deg,#16a34a,#4ade80); }
.conf-bar-fill-fake { height:8px; border-radius:100px; background:linear-gradient(90deg,#dc2626,#f87171); }

.transcript-box { background:#0d1220; border:1px solid #1e3050; border-radius:8px; padding:18px 22px; font-size:.92rem; color:#c8d8f0; line-height:1.75; white-space:pre-wrap; word-break:break-word; }
.transcript-empty { color:#3a5070; font-style:italic; font-size:.84rem; }

.sidebar-logo    { font-family:'Space Mono',monospace; font-size:1.05rem; font-weight:700; color:#f0f6ff; margin-bottom:4px; }
.sidebar-version { font-family:'Space Mono',monospace; font-size:.68rem; color:#38bdf8; letter-spacing:1.5px; margin-bottom:20px; }
.sidebar-section { font-family:'Space Mono',monospace; font-size:.64rem; letter-spacing:2px; text-transform:uppercase; color:#38bdf8; margin:20px 0 8px; padding-bottom:6px; border-bottom:1px solid #1e2535; }
.sidebar-section-admin { font-family:'Space Mono',monospace; font-size:.64rem; letter-spacing:2px; text-transform:uppercase; color:#fbbf24; margin:20px 0 8px; padding-bottom:6px; border-bottom:1px solid #2a2010; }
.sidebar-item    { font-size:.84rem; color:#7a90b0; padding:4px 0; line-height:1.5; }
.sidebar-item strong { color:#c8d8f0; }

.hist-row {
    background:#111520; border:1px solid #1a2535; border-radius:10px;
    padding:16px 20px; margin-bottom:10px;
    display:flex; align-items:center; gap:16px;
}
.hist-badge-real { background:rgba(34,197,94,.12); color:#4ade80; border:1px solid rgba(34,197,94,.3); font-family:'Space Mono',monospace; font-size:.68rem; padding:3px 10px; border-radius:100px; }
.hist-badge-fake { background:rgba(239,68,68,.12); color:#f87171; border:1px solid rgba(239,68,68,.3); font-family:'Space Mono',monospace; font-size:.68rem; padding:3px 10px; border-radius:100px; }
.hist-badge-admin { background:rgba(251,191,36,.12); color:#fbbf24; border:1px solid rgba(251,191,36,.3); font-family:'Space Mono',monospace; font-size:.68rem; padding:3px 10px; border-radius:100px; }
.hist-badge-user { background:rgba(56,189,248,.10); color:#38bdf8; border:1px solid rgba(56,189,248,.25); font-family:'Space Mono',monospace; font-size:.68rem; padding:3px 10px; border-radius:100px; }
.hist-fname  { font-size:.9rem; color:#c8d8f0; font-weight:500; }
.hist-meta   { font-size:.78rem; color:#4a6080; font-family:'Space Mono',monospace; }

.stat-card { background:#111520; border:1px solid #1e2535; border-radius:12px; padding:20px 24px; text-align:center; }
.stat-num  { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700; color:#38bdf8; }
.stat-num-admin { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700; color:#fbbf24; }
.stat-lbl  { font-size:.78rem; color:#5a7090; margin-top:4px; text-transform:uppercase; letter-spacing:1px; }

.stButton>button {
    background:linear-gradient(135deg,#1d4ed8,#0ea5e9); color:white; border:none;
    border-radius:8px; font-family:'Space Mono',monospace; font-size:.78rem;
    letter-spacing:1px; padding:10px 24px; width:100%; transition:opacity .2s;
}
.stButton>button:hover { opacity:.85; }

[data-testid="stFileUploader"] { background:#0d1220; border:1.5px dashed #1e3050; border-radius:10px; padding:8px; }
[data-testid="stFileUploader"]:hover { border-color:#38bdf8; }
audio { width:100%; border-radius:8px; }
[data-testid="stMetric"] { background:#0d1220; border:1px solid #1e2535; border-radius:10px; padding:14px 18px; }

[data-testid="stTextInput"] input, [data-testid="stTextInput"] input:focus {
    background:#0d1220 !important; border-color:#1e3050 !important; color:#e0e6f0 !important;
    border-radius:8px !important;
}

.footer { margin-top:48px; padding-top:20px; border-top:1px solid #1e2535; text-align:center; font-family:'Space Mono',monospace; font-size:.66rem; color:#3a5070; letter-spacing:1px; }

.admin-table { width:100%; border-collapse:collapse; font-size:.82rem; }
.admin-table th { font-family:'Space Mono',monospace; font-size:.66rem; letter-spacing:1.5px; text-transform:uppercase; color:#38bdf8; border-bottom:1px solid #1e2535; padding:10px 12px; text-align:left; }
.admin-table td { padding:10px 12px; border-bottom:1px solid #131a27; color:#c8d8f0; vertical-align:middle; }
.admin-table tr:hover td { background:#141c2c; }
</style>
""", unsafe_allow_html=True)

# ===============================
# MODEL LOAD
# ===============================
MODEL_PATH = "models/deepfake_modelv3.keras"

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()


# ===============================
# SIDEBAR
# ===============================
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🎙️ AudioForensics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-version">V 6.0 — CNN/MFCC + MFCC Visual</div>', unsafe_allow_html=True)

        if is_logged_in():
            user = get_session()
            role_badge = (
                '<span style="color:#fbbf24;font-size:.72rem;">👑 admin</span>'
                if is_admin() else
                '<span style="color:#38bdf8;font-size:.72rem;">user</span>'
            )
            st.markdown(f"""
            <div class="sidebar-section">Logged in as</div>
            <div class="sidebar-item"><strong>@{user['username']}</strong> &nbsp; {role_badge}</div>
            <div class="sidebar-item" style="font-size:.76rem;">{user['email']}</div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
            if st.button("🔬  New Analysis", key="nav_analyze"):
                st.session_state["active_page"] = "analyze"
                st.rerun()
            if st.button("📋  My Reports", key="nav_reports"):
                st.session_state["active_page"] = "reports"
                st.rerun()

            if is_admin():
                st.markdown('<div class="sidebar-section-admin">Admin</div>', unsafe_allow_html=True)
                if st.button("🛡️  Admin Dashboard", key="nav_admin"):
                    st.session_state["active_page"] = "admin"
                    st.rerun()

            st.markdown("---")
            if st.button("⎋  Log Out", key="nav_logout"):
                logout()
                st.rerun()
        else:
            st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
            if st.button("🔑  Login", key="nav_login"):
                st.session_state["active_page"] = "login"
                st.rerun()
            if st.button("📝  Sign Up", key="nav_signup"):
                st.session_state["active_page"] = "signup"
                st.rerun()

        st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-item"><strong>Architecture:</strong> CNN (3 conv blocks)</div>
        <div class="sidebar-item"><strong>Features:</strong> MFCC — 40 coefficients</div>
        <div class="sidebar-item"><strong>Dataset:</strong> ASVspoof 2019 LA</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="footer">
            AUDIOFORENSICS AI &nbsp;·&nbsp; CNN + MFCC<br>ASVspoof 2019 LA &nbsp;·&nbsp; TF + Streamlit
        </div>
        """, unsafe_allow_html=True)


# ===============================
# AUTH PAGES  (unchanged)
# ===============================
def page_login():
    st.markdown("""
    <div class="hero-block">
        <div class="hero-badge">⬡ AUTHENTICATION</div>
        <div class="hero-title">Welcome Back</div>
        <div class="hero-sub">Sign in to access your forensic analysis history.</div>
    </div>
    """, unsafe_allow_html=True)

    col, _ = st.columns([1, 1])
    with col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔑 Sign In</div>', unsafe_allow_html=True)

        identifier = st.text_input("Username or Email", placeholder="john_doe or john@example.com", key="li_id")
        password   = st.text_input("Password", type="password", placeholder="••••••••", key="li_pw")

        if st.button("Sign In →", key="btn_login"):
            if not identifier or not password:
                st.error("Please fill in all fields.")
            else:
                user = login_user(identifier, password)
                if user:
                    set_session(user)
                    st.session_state["active_page"] = "analyze"
                    st.success(f"Welcome back, **{user['username']}**! 🎉")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

        st.markdown("---")
        st.markdown('<div style="font-size:.84rem;color:#5a7090;text-align:center;">No account yet?</div>', unsafe_allow_html=True)
        if st.button("Create an account →", key="goto_signup"):
            st.session_state["active_page"] = "signup"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def page_signup():
    st.markdown("""
    <div class="hero-block">
        <div class="hero-badge">⬡ CREATE ACCOUNT</div>
        <div class="hero-title">Join AudioForensics</div>
        <div class="hero-sub">Create an account to save and revisit your analysis history.</div>
    </div>
    """, unsafe_allow_html=True)

    col, _ = st.columns([1, 1])
    with col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📝 New Account</div>', unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="john_doe", key="su_user")
        email    = st.text_input("Email", placeholder="john@example.com", key="su_email")
        pw       = st.text_input("Password", type="password", placeholder="Min 8 characters", key="su_pw")
        pw2      = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="su_pw2")

        if st.button("Create Account →", key="btn_signup"):
            errors = validate_signup(username, email, pw, pw2)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                user_id = signup_user(username, email, pw)
                from database import get_user_by_id
                user = dict(get_user_by_id(user_id))
                set_session(user)
                st.session_state["active_page"] = "analyze"
                st.success(f"Account created! Welcome, **{username}**! 🎉")
                st.rerun()

        st.markdown("---")
        st.markdown('<div style="font-size:.84rem;color:#5a7090;text-align:center;">Already have an account?</div>', unsafe_allow_html=True)
        if st.button("Sign in →", key="goto_login"):
            st.session_state["active_page"] = "login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ===============================
# ANALYSIS PAGE  ← MODIFIED
# ===============================
def page_analyze():
    user = get_session()

    st.markdown("""
    <div class="hero-block">
        <div class="hero-badge">⬡ FORENSIC AI SYSTEM</div>
        <div class="hero-title">EchoShield: Audio Deepfake Detection</div>
        <div class="hero-sub">Upload a voice recording to analyze authenticity using deep neural feature extraction and CNN classification.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">⬆ Upload Audio File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="",
        type=["wav", "flac", "mp3"],
        help="Supported: WAV, FLAC, MP3 — Recommended: 16kHz mono",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if not uploaded_file:
        st.session_state.pop("analysis_cache", None)
        return

    file_ext   = os.path.splitext(uploaded_file.name)[-1].lower()
    file_bytes = uploaded_file.read()

    file_key = f"{uploaded_file.name}_{len(file_bytes)}"
    cache    = st.session_state.get("analysis_cache", {})

    if cache.get("file_key") == file_key:
        analysis_id = cache["analysis_id"]
        duration    = cache["duration"]
        result      = cache["result"]
        real_conf   = cache["real_conf"]
        fake_conf   = cache["fake_conf"]
        transcript  = cache["transcript"]
        mel         = cache["mel"]
        mfcc        = cache["mfcc"]
        audio_bytes = cache["audio_bytes"]
    else:
        file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(file_bytes)
                file_path = tmp.name

            with st.spinner("🔬 Analyzing audio signal..."):
                y, sr    = librosa.load(file_path, sr=16000)
                duration = librosa.get_duration(y=y, sr=sr)

                # Convert to WAV bytes for consistent playback
                audio_buffer = io.BytesIO()
                wavfile.write(audio_buffer, sr, y.astype(np.float32))
                audio_bytes = audio_buffer.getvalue()

                mel      = extract_mel(file_path)
                mfcc     = extract_mfcc(file_path)           # shape (40, 157)

                mfcc_input = np.expand_dims(np.expand_dims(mfcc, -1), 0)
                pred       = model.predict(mfcc_input, verbose=0)
                fake_conf  = float(pred[0][0]) * 100
                real_conf  = float(pred[0][1]) * 100
                result     = "REAL" if real_conf > fake_conf else "FAKE"

            with st.spinner("🗣 Transcribing speech..."):
                try:
                    transcript = transcribe_audio(file_path)
                except Exception as te:
                    transcript = ""
                    st.warning(f"Transcription failed: {te}")

            analysis_id = save_analysis(
                user_id=user["id"],
                file_name=uploaded_file.name,
                duration_sec=duration,
                prediction=result,
                real_conf=real_conf,
                fake_conf=fake_conf,
                transcript=transcript,
            )
            st.session_state["current_analysis_id"] = analysis_id

            st.session_state["analysis_cache"] = {
                "file_key":    file_key,
                "analysis_id": analysis_id,
                "duration":    duration,
                "result":      result,
                "real_conf":   real_conf,
                "fake_conf":   fake_conf,
                "transcript":  transcript,
                "mel":         mel,
                "mfcc":        mfcc,
                "audio_bytes": audio_bytes,
            }

        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
            return
        finally:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

    # ── File Info + Player ─────────────────────────────────────────────────────
    col_info, col_player = st.columns([1.4, 1])

    with col_info:
        st.markdown('<div class="card"><div class="card-title">📁 File Information</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-row"><span class="info-label">File Name</span><span class="info-value">{uploaded_file.name}</span></div>
        <div class="info-row"><span class="info-label">Duration</span><span class="info-value">{duration:.2f} sec</span></div>
        <div class="info-row"><span class="info-label">Sample Rate</span><span class="info-value">16,000 Hz</span></div>
        <div class="info-row"><span class="info-label">MFCC Coefficients</span><span class="info-value">40</span></div>
        <div class="info-row"><span class="info-label">Feature Frames</span><span class="info-value">157</span></div>
        <div class="info-row"><span class="info-label">Analysis ID</span><span class="info-value">#{analysis_id}</span></div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_player:
        st.markdown('<div class="card"><div class="card-title">▶ Playback</div>', unsafe_allow_html=True)
        st.audio(audio_bytes, format="audio/wav")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Mel Spectrogram ────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">🌊 Mel Spectrogram Analysis</div>', unsafe_allow_html=True)
    fig_mel = plot_mel(mel)
    fig_mel.patch.set_facecolor('#111520')
    st.pyplot(fig_mel, use_container_width=True)
    plt.close(fig_mel)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── MFCC Feature Analysis  ← NEW SECTION ──────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">🧩 MFCC Feature Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:.82rem;color:#5a7090;margin-bottom:14px;line-height:1.6;">
        The heatmap below shows the <strong style="color:#c8d8f0;">40 Mel-Frequency Cepstral Coefficients</strong>
        (rows) extracted across <strong style="color:#c8d8f0;">157 temporal frames</strong> (columns).
        Warm colours (red) indicate high positive activation; cool colours (blue) indicate negative values.
        This matrix is the exact input tensor fed to the CNN classifier.
    </div>
    """, unsafe_allow_html=True)

    fig_mfcc = plot_mfcc(mfcc)
    st.pyplot(fig_mfcc, use_container_width=True)
    plt.close(fig_mfcc)

    # Optional: raw dataframe toggle — zero re-computation cost
    with st.expander("🔢  View raw MFCC matrix (40 × 157)", expanded=False):
        import pandas as pd
        df_mfcc = pd.DataFrame(
            mfcc,
            index=[f"C{i:02d}" for i in range(mfcc.shape[0])],
            columns=[f"F{j:03d}" for j in range(mfcc.shape[1])],
        )
        st.dataframe(
            df_mfcc.style.background_gradient(cmap="coolwarm", axis=None),
            height=400,
            use_container_width=True,
        )
        st.markdown(
            f'<div style="font-size:.75rem;color:#4a6080;margin-top:6px;">'
            f'Shape: {mfcc.shape[0]} coefficients × {mfcc.shape[1]} frames &nbsp;|&nbsp; '
            f'Min: {mfcc.min():.3f} &nbsp;|&nbsp; Max: {mfcc.max():.3f} &nbsp;|&nbsp; '
            f'Mean: {mfcc.mean():.3f}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Result ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">🧠 Detection Result</div>', unsafe_allow_html=True)
    col_verdict, col_conf = st.columns([1, 1.2])

    with col_verdict:
        if result == "REAL":
            st.markdown(f"""
            <div class="result-real">
                <div class="result-label">✓ Verdict</div>
                <div class="result-verdict">REAL</div>
                <div style="color:#4ade80;font-size:.78rem;margin-top:8px;font-family:'Space Mono',monospace;">Authentic Voice Signal</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fake">
                <div class="result-label">⚠ Verdict</div>
                <div class="result-verdict">FAKE</div>
                <div style="color:#f87171;font-size:.78rem;margin-top:8px;font-family:'Space Mono',monospace;">Synthetic / Spoofed Audio Detected</div>
            </div>""", unsafe_allow_html=True)

    with col_conf:
        st.markdown(f"""
        <div style="padding:8px 0;">
            <div class="conf-bar-wrap">
                <div class="conf-bar-label"><span>REAL CONFIDENCE</span><span>{real_conf:.1f}%</span></div>
                <div class="conf-bar-bg"><div class="conf-bar-fill-real" style="width:{real_conf}%"></div></div>
            </div>
            <div style="margin-top:14px"></div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-label"><span>FAKE CONFIDENCE</span><span>{fake_conf:.1f}%</span></div>
                <div class="conf-bar-bg"><div class="conf-bar-fill-fake" style="width:{fake_conf}%"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Real", f"{real_conf:.2f}%")
        st.metric("Fake", f"{fake_conf:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Transcript ─────────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">🗣 Speech Transcription</div>', unsafe_allow_html=True)
    if transcript:
        st.markdown(f'<div class="transcript-box">{transcript}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="transcript-empty">No speech detected or audio is silent.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Report  ← passes mfcc_img_path too ────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">📄 Forensic Report</div>', unsafe_allow_html=True)

    mel_img_path  = tempfile.mktemp(suffix=".png")
    mfcc_img_path = tempfile.mktemp(suffix=".png")

    # Render and save Mel
    fig_mel_save = plot_mel(mel)
    fig_mel_save.savefig(mel_img_path, bbox_inches="tight", facecolor="white")
    plt.close(fig_mel_save)

    # Render and save MFCC  ← NEW
    fig_mfcc_save = plot_mfcc(mfcc)
    fig_mfcc_save.savefig(mfcc_img_path, bbox_inches="tight", facecolor="white")
    plt.close(fig_mfcc_save)

    if st.button("📄 Generate & Download Forensic Report"):
        report_path = generate_report(
            file_name=uploaded_file.name,
            prediction=result,
            real_conf=real_conf,
            fake_conf=fake_conf,
            duration=duration,
            mel_img_path=mel_img_path,
            mfcc_img_path=mfcc_img_path,
            mfcc=mfcc,
            transcript=transcript,
        )
        with open(report_path, "rb") as f:
            pdf_bytes = f.read()

        save_report(analysis_id, pdf_bytes)

        st.download_button(
            label="⬇ Download PDF Report",
            data=pdf_bytes,
            file_name=f"forensic_report_{analysis_id}.pdf",
            mime="application/pdf",
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Cleanup temp images
    for p in (mel_img_path, mfcc_img_path):
        if os.path.exists(p):
            os.remove(p)


# ===============================
# MY REPORTS PAGE  (unchanged)
# ===============================
def page_reports():
    user    = get_session()
    user_id = user["id"]

    st.markdown("""
    <div class="hero-block">
        <div class="hero-badge">⬡ ANALYSIS HISTORY</div>
        <div class="hero-title">My Reports</div>
        <div class="hero-sub">All previous deepfake analyses linked to your account.</div>
    </div>
    """, unsafe_allow_html=True)

    stats = get_user_stats(user_id)
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, str(stats.get("total", 0)),        "Total Analyses"),
        (c2, str(stats.get("real_count", 0)),   "REAL Results"),
        (c3, str(stats.get("fake_count", 0)),   "FAKE Results"),
        (c4, f"{float(stats.get('avg_real_conf') or 0):.1f}%", "Avg Real Conf"),
    ]
    for col, num, lbl in metrics:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-num">{num}</div>
                <div class="stat-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    analyses = get_user_analyses(user_id)

    if not analyses:
        st.markdown('<div class="card"><div class="transcript-empty">No analyses found. Run your first analysis to see it here.</div></div>', unsafe_allow_html=True)
        return

    st.markdown(f'<div class="card-title" style="margin-bottom:14px;">📋 {len(analyses)} Record(s)</div>', unsafe_allow_html=True)

    for rec in analyses:
        aid        = rec["id"]
        pred       = rec["prediction"]
        badge_cls  = "hist-badge-real" if pred == "REAL" else "hist-badge-fake"
        has_pdf    = rec.get("has_report", False)

        with st.expander(f"#{aid}  ·  {rec['file_name']}  ·  {rec['created_at'][:16]}", expanded=False):
            col_det, col_act = st.columns([2, 1])

            with col_det:
                st.markdown(f"""
                <div class="info-row"><span class="info-label">File</span><span class="info-value">{rec['file_name']}</span></div>
                <div class="info-row"><span class="info-label">Verdict</span>
                    <span><span class="{badge_cls}">{pred}</span></span>
                </div>
                <div class="info-row"><span class="info-label">Real Confidence</span><span class="info-value">{rec['real_conf']:.1f}%</span></div>
                <div class="info-row"><span class="info-label">Fake Confidence</span><span class="info-value">{rec['fake_conf']:.1f}%</span></div>
                <div class="info-row"><span class="info-label">Duration</span><span class="info-value">{rec['duration_sec']:.2f}s</span></div>
                <div class="info-row"><span class="info-label">Date</span><span class="info-value">{rec['created_at']}</span></div>
                """, unsafe_allow_html=True)

                if rec.get("transcript"):
                    st.markdown("<br>**Transcript:**", unsafe_allow_html=False)
                    st.markdown(f'<div class="transcript-box" style="margin-top:6px;">{rec["transcript"]}</div>', unsafe_allow_html=True)

            with col_act:
                st.markdown("<br>", unsafe_allow_html=True)
                if has_pdf:
                    pdf_bytes = get_report(aid, user_id)
                    if pdf_bytes:
                        st.download_button(
                            label="⬇ Download Report",
                            data=pdf_bytes,
                            file_name=f"forensic_report_{aid}.pdf",
                            mime="application/pdf",
                            key=f"dl_{aid}",
                        )
                else:
                    st.markdown('<div class="transcript-empty">No PDF saved yet.</div>', unsafe_allow_html=True)

                if st.button("🗑 Delete", key=f"del_{aid}"):
                    deleted = delete_analysis(aid, user_id)
                    if deleted:
                        st.success("Record deleted.")
                        st.rerun()


# ===============================
# ADMIN DASHBOARD PAGE  (unchanged)
# ===============================
def page_admin():
    if not is_admin():
        st.error("⛔ Access denied. This page is restricted to administrators.")
        return

    st.markdown("""
    <div class="hero-block">
        <div class="hero-badge-admin">👑 ADMIN PANEL</div>
        <div class="hero-title">Admin Dashboard</div>
        <div class="hero-sub">Site-wide overview — manage all users and analyses.</div>
    </div>
    """, unsafe_allow_html=True)

    gstats = get_global_stats()
    c1, c2, c3, c4 = st.columns(4)
    admin_metrics = [
        (c1, str(gstats["total_users"]),     "Total Users"),
        (c2, str(gstats["total_analyses"]),  "Total Analyses"),
        (c3, str(gstats["real_count"]),      "REAL Results"),
        (c4, str(gstats["fake_count"]),      "FAKE Results"),
    ]
    for col, num, lbl in admin_metrics:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-num-admin">{num}</div>
                <div class="stat-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_users, tab_analyses = st.tabs(["👤  All Users", "🔬  All Analyses"])

    with tab_users:
        st.markdown("<br>", unsafe_allow_html=True)
        users = get_all_users()

        if not users:
            st.markdown('<div class="transcript-empty">No users found.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card-title">👥 {len(users)} Registered User(s)</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            h1, h2, h3, h4, h5 = st.columns([0.5, 1.5, 2.5, 1, 2])
            for col, label in zip([h1, h2, h3, h4, h5], ["ID", "Username", "Email", "Role", "Joined"]):
                col.markdown(f'<span style="font-family:\'Space Mono\',monospace;font-size:.68rem;letter-spacing:1.5px;text-transform:uppercase;color:#38bdf8;">{label}</span>', unsafe_allow_html=True)

            st.markdown('<hr style="border-color:#1e2535;margin:6px 0 10px;">', unsafe_allow_html=True)

            for u in users:
                c1, c2, c3, c4, c5 = st.columns([0.5, 1.5, 2.5, 1, 2])
                role_html = (
                    '<span class="hist-badge-admin">admin</span>'
                    if u["role"] == "admin" else
                    '<span class="hist-badge-user">user</span>'
                )
                c1.markdown(f'<span style="color:#5a7090;font-family:\'Space Mono\',monospace;font-size:.78rem;">#{u["id"]}</span>', unsafe_allow_html=True)
                c2.markdown(f'<span style="color:#c8d8f0;">@{u["username"]}</span>', unsafe_allow_html=True)
                c3.markdown(f'<span style="color:#7a90b0;font-size:.82rem;">{u["email"]}</span>', unsafe_allow_html=True)
                c4.markdown(role_html, unsafe_allow_html=True)
                c5.markdown(f'<span style="color:#5a7090;font-family:\'Space Mono\',monospace;font-size:.76rem;">{u["created_at"][:16]}</span>', unsafe_allow_html=True)

    with tab_analyses:
        st.markdown("<br>", unsafe_allow_html=True)
        analyses = get_all_analyses()

        if not analyses:
            st.markdown('<div class="transcript-empty">No analyses found.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card-title">🔬 {len(analyses)} Analysis Record(s)</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            for rec in analyses:
                aid       = rec["id"]
                pred      = rec["prediction"]
                badge_cls = "hist-badge-real" if pred == "REAL" else "hist-badge-fake"

                with st.expander(
                    f"#{aid}  ·  @{rec['username']}  ·  {rec['file_name']}  ·  {rec['created_at'][:16]}",
                    expanded=False,
                ):
                    col_det, col_act = st.columns([2, 1])

                    with col_det:
                        st.markdown(f"""
                        <div class="info-row"><span class="info-label">Analysis ID</span><span class="info-value">#{aid}</span></div>
                        <div class="info-row"><span class="info-label">User</span><span class="info-value">@{rec['username']}</span></div>
                        <div class="info-row"><span class="info-label">File</span><span class="info-value">{rec['file_name']}</span></div>
                        <div class="info-row"><span class="info-label">Verdict</span>
                            <span><span class="{badge_cls}">{pred}</span></span>
                        </div>
                        <div class="info-row"><span class="info-label">Real Confidence</span><span class="info-value">{rec['real_conf']:.1f}%</span></div>
                        <div class="info-row"><span class="info-label">Fake Confidence</span><span class="info-value">{rec['fake_conf']:.1f}%</span></div>
                        <div class="info-row"><span class="info-label">Duration</span><span class="info-value">{rec['duration_sec']:.2f}s</span></div>
                        <div class="info-row"><span class="info-label">Date</span><span class="info-value">{rec['created_at']}</span></div>
                        """, unsafe_allow_html=True)

                        if rec.get("transcript"):
                            st.markdown("**Transcript:**", unsafe_allow_html=False)
                            st.markdown(f'<div class="transcript-box" style="margin-top:6px;">{rec["transcript"]}</div>', unsafe_allow_html=True)

                    with col_act:
                        st.markdown("<br>", unsafe_allow_html=True)
                        confirm_key = f"admin_confirm_del_{aid}"
                        if st.session_state.get(confirm_key):
                            st.warning("Are you sure?")
                            col_yes, col_no = st.columns(2)
                            with col_yes:
                                if st.button("Yes, delete", key=f"admin_yes_{aid}"):
                                    admin_delete_analysis(aid)
                                    st.session_state.pop(confirm_key, None)
                                    st.success("Deleted.")
                                    st.rerun()
                            with col_no:
                                if st.button("Cancel", key=f"admin_no_{aid}"):
                                    st.session_state.pop(confirm_key, None)
                                    st.rerun()
                        else:
                            if st.button("🗑 Delete", key=f"admin_del_{aid}"):
                                st.session_state[confirm_key] = True
                                st.rerun()


# ===============================
# LANDING  (unchanged)
# ===============================
def page_landing():
    st.markdown("""
    <div class="hero-block">
        <div class="hero-badge">⬡ FORENSIC AI SYSTEM</div>
        <div class="hero-title">EchoShield: Audio Deepfake Detection</div>
        <div class="hero-sub">
            AI-powered deepfake detection trained on ASVspoof 2019 LA.<br>
            Sign in or create a free account to start analyzing audio.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🔬 How it works</div>
            <div class="sidebar-item">1. Upload a WAV / FLAC / MP3 file.</div>
            <div class="sidebar-item">2. The CNN extracts 40-coefficient MFCCs.</div>
            <div class="sidebar-item">3. Get a REAL / FAKE verdict + confidence scores.</div>
            <div class="sidebar-item">4. Download a detailed forensic PDF report.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
            <div class="card-title">✅ Features</div>
            <div class="sidebar-item">• CNN trained on ASVspoof 2019 LA dataset</div>
            <div class="sidebar-item">• Whisper-based speech transcription</div>
            <div class="sidebar-item">• Mel spectrogram + MFCC heatmap visualization</div>
            <div class="sidebar-item">• Persistent analysis history per user</div>
            <div class="sidebar-item">• Downloadable PDF forensic reports</div>
        </div>""", unsafe_allow_html=True)

    col_a, col_b, _ = st.columns([1, 1, 2])
    with col_a:
        if st.button("🔑  Sign In"):
            st.session_state["active_page"] = "login"
            st.rerun()
    with col_b:
        if st.button("📝  Create Account"):
            st.session_state["active_page"] = "signup"
            st.rerun()


# ===============================
# ROUTER
# ===============================
def main():
    render_sidebar()

    page = st.session_state.get("active_page", "landing")

    if is_logged_in():
        if page in ("landing", "login", "signup"):
            page = "analyze"
            st.session_state["active_page"] = "analyze"

        if page == "reports":
            page_reports()
        elif page == "admin":
            page_admin()
        else:
            page_analyze()
    else:
        if page in ("reports", "analyze", "admin"):
            page = "landing"
            st.session_state["active_page"] = "landing"

        if page == "login":
            page_login()
        elif page == "signup":
            page_signup()
        else:
            page_landing()


if __name__ == "__main__":
    main()