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

    # ✅ normalization (VERY IMPORTANT)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    # padding
    if mfcc.shape[1] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc

# =========================
# MEL (USED FOR VISUALIZATION)
# =========================
def extract_mel(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel)

    if mel_db.shape[1] < MAX_LEN:
        pad = MAX_LEN - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)))
    else:
        mel_db = mel_db[:, :MAX_LEN]

    return mel_db

# =========================
# PLOT MEL
# =========================
def plot_mel(mel):
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel,
        sr=SAMPLE_RATE,
        x_axis='time',
        y_axis='mel',
        ax=ax
    )
    fig.colorbar(img, ax=ax)
    ax.set_title("Mel Spectrogram")

    return fig