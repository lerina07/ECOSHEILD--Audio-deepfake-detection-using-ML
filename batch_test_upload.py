import streamlit as st
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import load_model
import tempfile
import io

# ===============================
# LOAD MODEL
# ===============================
model = load_model(r"D:\Test 1\models\deepfake_modelv3.keras")

st.title("🎙️ Audio Deepfake Forensic System (Batch Upload)")

# ===============================
# MULTIPLE FILE UPLOAD
# ===============================
uploaded_files = st.file_uploader(
    "Upload Multiple Audio Files",
    type=["wav", "flac", "mp3"],
    accept_multiple_files=True
)

# ===============================
# PROCESS FILES
# ===============================
if uploaded_files:

    results = []

    st.info(f"Processing {len(uploaded_files)} files...")

    for uploaded_file in uploaded_files:
        try:
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name

            # ===============================
            # LOAD AUDIO (SAME AS TRAINING)
            # ===============================
            y, sr = librosa.load(file_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)

            # ===============================
            # MFCC (EXACT SAME AS TRAINING)
            # ===============================
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

            # Normalization (VERY IMPORTANT)
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

            # Padding / Trimming
            if mfcc.shape[1] < 157:
                pad = 157 - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
            else:
                mfcc = mfcc[:, :157]

            mfcc = mfcc.reshape(1, 40, 157, 1)

            # ===============================
            # PREDICTION
            # ===============================
            pred = model.predict(mfcc, verbose=0)

            real_conf = float(pred[0][1] * 100)
            fake_conf = float(pred[0][0] * 100)

            result = "REAL" if real_conf > fake_conf else "FAKE"

            # Speaker (optional logic)
            name = uploaded_file.name.lower()
            speaker = "Ashley" if "ashley" in name else ("Lerina" if "lerina" in name else "Unknown")

            results.append({
                "File Name": uploaded_file.name,
                "Speaker": speaker,
                "Duration (sec)": round(duration, 2),
                "Prediction": result,
                "Real Confidence (%)": round(real_conf, 2),
                "Fake Confidence (%)": round(fake_conf, 2),
                "Verdict": "Bonafide Voice" if result == "REAL" else "Spoofed Audio"
            })

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    # ===============================
    # DISPLAY TABLE
    # ===============================
    df = pd.DataFrame(results)

    st.subheader("📊 Results")
    st.dataframe(df)

    # ===============================
    # DOWNLOAD EXCEL
    # ===============================
    def convert_to_excel(dataframe):
        output = io.BytesIO()
        dataframe.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        return output

    excel_file = convert_to_excel(df)

    st.download_button(
        label="📥 Download Results as Excel",
        data=excel_file,
        file_name="batch_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )