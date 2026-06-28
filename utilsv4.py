import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# =========================
# CONSTANTS
# =========================
SAMPLE_RATE = 16000
N_MFCC = 40
N_MELS = 128
MAX_LEN = 157

# =========================
# MFCC (USED FOR MODEL)
# =========================
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # normalization (VERY IMPORTANT)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    # padding
    if mfcc.shape[1] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc  # shape: (40, 157)


# =========================
# MEL (USED FOR VISUALIZATION)
# =========================
def extract_mel(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Normalize audio
    y = librosa.util.normalize(y)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=N_MELS
    )

    # Proper dB scaling
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db  # NO PADDING


# =========================
# PLOT MEL
# =========================
def plot_mel(mel):
    fig, ax = plt.subplots(figsize=(10, 4))

    img = librosa.display.specshow(
        mel,
        sr=SAMPLE_RATE,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        cmap='magma',
        ax=ax
    )

    ax.set_title("Mel Spectrogram", color="blue")
    ax.set_xlabel("Time", color="blue")
    ax.set_ylabel("Frequency (Hz)", color="blue")

    ax.tick_params(axis='x', colors='blue')
    ax.tick_params(axis='y', colors='blue')

    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='blue')
    plt.setp(cbar.ax.get_yticklabels(), color='blue')

    fig.patch.set_facecolor('#111520')
    ax.set_facecolor('#111520')

    plt.tight_layout()
    return fig


# =========================
# PLOT MFCC HEATMAP
# =========================
def plot_mfcc(mfcc):
    """
    Renders the MFCC matrix (40 × 157) as a styled heatmap
    matching the dark UI aesthetic.

    Args:
        mfcc (np.ndarray): Normalized MFCC array of shape (40, 157).

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    # Use 'coolwarm' for MFCC — contrasts well on dark backgrounds
    # and makes positive/negative coefficient values intuitively readable.
    img = ax.imshow(
        mfcc,
        aspect='auto',
        origin='lower',
        cmap='coolwarm',
        interpolation='nearest',
    )

    ax.set_title("MFCC Feature Matrix  (40 coefficients × 157 frames)", color="#38bdf8", fontsize=10)
    ax.set_xlabel("Frame Index  →  Time", color="#7a90b0", fontsize=9)
    ax.set_ylabel("MFCC Coefficient Index", color="#7a90b0", fontsize=9)

    ax.tick_params(axis='x', colors='#7a90b0')
    ax.tick_params(axis='y', colors='#7a90b0')

    # Y-axis ticks: every 5 coefficients
    ax.set_yticks(range(0, 40, 5))
    ax.set_yticklabels([f"C{i}" for i in range(0, 40, 5)], color="#7a90b0", fontsize=7)

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Normalized Amplitude", color="#7a90b0", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#7a90b0")
    plt.setp(cbar.ax.get_yticklabels(), color="#7a90b0")

    fig.patch.set_facecolor('#111520')
    ax.set_facecolor('#0d1220')

    # Subtle grid lines for readability
    ax.set_xticks(range(0, 157, 20))
    ax.grid(axis='y', color='#1e2535', linewidth=0.4, linestyle='--')

    plt.tight_layout()
    return fig


# =========================
# TRANSCRIPTION (Whisper)
# =========================
import whisper

_whisper_model = None

def transcribe_audio(file_path):
    """
    Transcribe audio file to text using OpenAI Whisper.
    Uses the 'tiny' model by default for speed.
    Swap to 'base' or 'small' for better accuracy.
    Install: pip install openai-whisper
    """
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("tiny")
    result = _whisper_model.transcribe(file_path)
    return result.get("text", "").strip()