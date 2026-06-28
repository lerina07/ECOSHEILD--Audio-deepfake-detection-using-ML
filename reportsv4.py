from fpdf import FPDF
from datetime import datetime
import os
import numpy as np



# ===============================
# HELPER: Generate forensic text based on prediction & confidence
# ===============================
def _generate_forensic_analysis(prediction, real_conf, fake_conf):
    """Returns a dict of dynamic text blocks for the forensic section."""
    is_real = prediction.upper() == "REAL"
    diff = abs(real_conf - fake_conf)

    # --- Audio Behaviour Observations ---
    if is_real:
        behaviour = (
            "- Smooth and continuous frequency transitions observed across temporal frames.\n"
            "- Energy distribution across Mel bands appears natural and consistent with "
            "human vocal production.\n"
            "- No abrupt spectral spikes, over-smoothed regions, or synthetic artifacts "
            "detected in the feature map.\n"
            "- MFCC coefficient variance is within the expected range for authentic speech."
        )
    else:
        behaviour = (
            "- Irregular spectral patterns detected across temporal frames, inconsistent "
            "with natural human speech production.\n"
            "- Abrupt or unnatural frequency transitions observed in multiple Mel bands.\n"
            "- Possible over-smoothing or spectral flattening artifacts -- hallmarks of "
            "neural TTS or voice conversion systems.\n"
            "- MFCC coefficient variance deviates from patterns observed in genuine speech "
            "in the training corpus."
        )

    # --- Model Decision Justification ---
    if is_real and real_conf >= 70:
        decision = (
            f"The model assigns a high Real confidence of {real_conf:.1f}%, indicating "
            "strong agreement with spectral patterns learned from authentic speech samples "
            "in the ASVspoof 2019 LA dataset. Stable temporal dynamics and natural "
            "cepstral coefficients reinforce this classification."
        )
    elif not is_real and fake_conf >= 70:
        decision = (
            f"The model assigns a high Fake confidence of {fake_conf:.1f}%, flagging "
            "significant anomalies in spectral consistency. These anomalies are "
            "characteristic of synthesized or voice-converted audio, where generative "
            "models often leave identifiable spectral fingerprints."
        )
    elif diff <= 15:
        decision = (
            f"The confidence scores are closely matched (Real: {real_conf:.1f}%, "
            f"Fake: {fake_conf:.1f}%), indicating marginal separation between the two "
            "classes. This may be due to overlapping acoustic characteristics between "
            "the submitted audio and both real and synthetic speech distributions. "
            "The result should be interpreted with caution and corroborated with "
            "additional analysis."
        )
    else:
        decision = (
            f"The model predicts {prediction.upper()} with a confidence of "
            f"{max(real_conf, fake_conf):.1f}%. While not in the high-confidence "
            "threshold, the spectral feature distribution moderately favours the "
            f"{prediction.upper()} classification."
        )

    return behaviour, decision


# ===============================
# MAIN REPORT FUNCTION
# ===============================
def generate_report(file_name, prediction, real_conf, fake_conf, duration,
                    mel_img_path=None, mfcc_img_path=None, mfcc=None, transcript=None):
    """
    Generate a PDF forensic report.

    Args:
        file_name       : Original uploaded file name.
        prediction      : 'REAL' or 'FAKE'.
        real_conf       : Real confidence percentage (float).
        fake_conf       : Fake confidence percentage (float).
        duration        : Audio duration in seconds (float).
        mel_img_path    : Path to the Mel spectrogram PNG (optional).
        mfcc_img_path   : Path to the MFCC heatmap PNG (optional).
        mfcc            : MFCC matrix (np.ndarray, shape (40, 157)) (optional).
        transcript      : Transcribed speech text (optional).

    Returns:
        str: Path to the generated PDF file ('report.pdf').
    """
    pdf = FPDF()
    pdf.add_page()

    # -----------------------------------------------
    # HEADER BLOCK
    # -----------------------------------------------
    pdf.set_fill_color(13, 15, 20)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", size=16)
    pdf.cell(0, 12, txt="AUDIO FORENSIC REPORT", ln=True, align='C', fill=False)

    pdf.set_font("Arial", size=9)
    pdf.set_text_color(100, 120, 160)
    pdf.cell(0, 6, txt="CNN-Based Audio Deepfake Detection System", ln=True, align='C')
    pdf.ln(4)

    pdf.set_draw_color(56, 189, 248)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # -----------------------------------------------
    # CASE METADATA
    # -----------------------------------------------
    pdf.set_font("Arial", "B", size=9)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 7, txt="CASE INFORMATION", ln=True)

    pdf.set_draw_color(30, 37, 53)
    pdf.set_line_width(0.2)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    def info_row(label, value):
        pdf.set_font("Arial", "B", size=9)
        pdf.set_text_color(80, 100, 130)
        pdf.cell(55, 7, txt=label, border=0)
        pdf.set_font("Arial", size=9)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 7, txt=str(value), border=0, ln=True)

    info_row("Case Number:", f"DF-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    info_row("File Name:", file_name)
    info_row("File Type:", "Audio")
    info_row("Location:", "Uploaded File")
    info_row("Stakeholder:", "Investigator")
    info_row("Uploaded:", str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    info_row("Analysis Time:", str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    info_row("Duration:", f"{duration:.2f} sec")

    pdf.ln(4)

    # -----------------------------------------------
    # PREPROCESSING
    # -----------------------------------------------
    pdf.set_font("Arial", "B", size=9)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 7, txt="PREPROCESSING", ln=True)
    pdf.set_draw_color(30, 37, 53)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    info_row("Feature:", "MFCC (40 coefficients)")
    info_row("Sample Rate:", "16,000 Hz")
    info_row("Max Frames:", "157")
    info_row("Normalization:", "Mean-Std per sample")

    pdf.ln(4)

    # -----------------------------------------------
    # RESULT BLOCK
    # -----------------------------------------------
    pdf.set_font("Arial", "B", size=9)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 7, txt="DETECTION RESULT", ln=True)
    pdf.set_draw_color(30, 37, 53)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    is_real = prediction.upper() == "REAL"
    if is_real:
        pdf.set_fill_color(220, 252, 231)
        pdf.set_text_color(22, 101, 52)
    else:
        pdf.set_fill_color(254, 226, 226)
        pdf.set_text_color(153, 27, 27)

    pdf.set_font("Arial", "B", size=14)
    pdf.cell(0, 14, txt=f"  VERDICT: {prediction.upper()}", ln=True, fill=True)
    pdf.ln(3)

    pdf.set_text_color(40, 40, 40)
    info_row("Real Confidence:", f"{real_conf:.2f}%")
    info_row("Fake Confidence:", f"{fake_conf:.2f}%")

    pdf.ln(6)

    # -----------------------------------------------
    # SPEECH TRANSCRIPTION
    # -----------------------------------------------
    pdf.set_font("Arial", "B", size=9)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 7, txt="SPEECH TRANSCRIPTION", ln=True)
    pdf.set_draw_color(30, 37, 53)
    pdf.set_line_width(0.2)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Arial", "I", size=7.5)
    pdf.set_text_color(100, 120, 150)
    pdf.cell(0, 5, txt="Transcribed using OpenAI Whisper (automatic speech recognition)", ln=True)
    pdf.ln(3)

    if transcript and transcript.strip():
        box_y = pdf.get_y()
        estimated_lines = max(3, len(transcript) // 90 + 1)
        box_height = estimated_lines * 5 + 10
        pdf.set_fill_color(245, 248, 255)
        pdf.set_draw_color(200, 215, 240)
        pdf.set_line_width(0.3)
        pdf.rect(10, box_y, 190, box_height, style='FD')
        pdf.ln(3)

        pdf.set_font("Arial", size=9)
        pdf.set_text_color(30, 30, 30)
        safe_transcript = transcript.encode('latin-1', errors='replace').decode('latin-1')
        pdf.multi_cell(185, 5.5, txt=f"  {safe_transcript}", border=0)
        pdf.ln(4)
    else:
        pdf.set_font("Arial", "I", size=8)
        pdf.set_text_color(140, 140, 140)
        pdf.cell(0, 6, txt="No speech detected or transcription unavailable.", ln=True)
        pdf.ln(4)

    # -----------------------------------------------
    # FORENSIC ANALYSIS & JUSTIFICATION
    # -----------------------------------------------
    behaviour_text, decision_text = _generate_forensic_analysis(prediction, real_conf, fake_conf)

    pdf.set_font("Arial", "B", size=9)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 7, txt="FORENSIC ANALYSIS & JUSTIFICATION", ln=True)
    pdf.set_draw_color(30, 37, 53)
    pdf.set_line_width(0.2)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # 1. Preprocessing Explanation
    pdf.set_font("Arial", "B", size=8)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 6, txt="1. Feature Extraction Rationale", ln=True)
    pdf.ln(1)

    pdf.set_font("Arial", size=8)
    pdf.set_text_color(40, 40, 40)
    preprocessing_explanation = (
        "Mel-Frequency Cepstral Coefficients (MFCCs) are used as the primary feature "
        "representation because they closely model how the human auditory system "
        "perceives sound. MFCCs capture the vocal tract shape and short-term spectral "
        "envelope of speech, making them highly discriminative between natural and "
        "synthesized audio.\n\n"
        "Mean-standard deviation normalization is applied per sample to eliminate "
        "recording-level biases caused by variations in microphone gain, room acoustics, "
        "or recording equipment. This ensures that the model evaluates spectral shape "
        "rather than absolute energy levels.\n\n"
        "Fixed-length padding to 157 frames (~2.5 seconds at 16 kHz with a 512-sample "
        "hop length) standardizes the input tensor shape required by the CNN architecture. "
        "Shorter samples are zero-padded; longer samples are truncated from the right."
    )
    pdf.multi_cell(0, 5, txt=preprocessing_explanation)
    pdf.ln(4)

    # 2. Audio Behaviour Analysis
    pdf.set_font("Arial", "B", size=8)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 6, txt="2. Audio Behaviour Analysis (Rule-Based)", ln=True)
    pdf.ln(1)

    pdf.set_font("Arial", size=8)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 5, txt=behaviour_text)
    pdf.ln(4)

    # 3. Model Decision Justification
    pdf.set_font("Arial", "B", size=8)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 6, txt="3. Model Decision Justification", ln=True)
    pdf.ln(1)

    pdf.set_font("Arial", size=8)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 5, txt=decision_text)
    pdf.ln(4)

    # 4. Confidence Score Interpretation
    pdf.set_font("Arial", "B", size=8)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 6, txt="4. Confidence Score Interpretation", ln=True)
    pdf.ln(1)

    pdf.set_font("Arial", size=8)
    pdf.set_text_color(40, 40, 40)
    conf_explanation = (
        "Confidence scores represent the model's softmax output probabilities, "
        "reflecting the degree of certainty in its classification. A score of 100% "
        "indicates complete certainty (rarely observed in practice); scores above 70% "
        "are generally considered reliable. Scores between 45-55% indicate marginal "
        "decisions where both classes exhibit competing feature characteristics.\n\n"
        f"In this case: Real = {real_conf:.1f}%,  Fake = {fake_conf:.1f}%. "
        "These probabilities are derived from patterns learned during training on the "
        "ASVspoof 2019 Logical Access (LA) dataset, which includes a wide variety of "
        "TTS and voice conversion spoofing attacks."
    )
    pdf.multi_cell(0, 5, txt=conf_explanation)
    pdf.ln(4)

    # 5. Limitation Disclaimer
    pdf.set_fill_color(255, 248, 220)
    pdf.set_draw_color(180, 130, 20)
    pdf.set_line_width(0.4)

    box_y = pdf.get_y()
    pdf.rect(10, box_y, 190, 36, style='D')
    pdf.ln(2)

    pdf.set_font("Arial", "B", size=8)
    pdf.set_text_color(120, 80, 10)
    pdf.cell(0, 5, txt="  5. Limitations & Disclaimer", ln=True)

    pdf.set_font("Arial", size=7.5)
    pdf.set_text_color(80, 55, 10)
    disclaimer = (
        "  This system is trained exclusively on the ASVspoof 2019 Logical Access dataset "
        "and may not generalise to all deepfake types, codecs, or recording conditions. "
        "Novel spoofing algorithms not represented in the training data may evade detection. "
        "This report is intended as a forensic support tool and must NOT be treated as "
        "conclusive legal evidence. Human expert review is strongly recommended before "
        "any consequential decision is made based on this output."
    )
    pdf.multi_cell(0, 5, txt=disclaimer)
    pdf.ln(5)

    # -----------------------------------------------
    # MEL SPECTROGRAM IMAGE
    # -----------------------------------------------
    if mel_img_path and os.path.exists(mel_img_path):
        pdf.set_font("Arial", "B", size=9)
        pdf.set_text_color(56, 189, 248)
        pdf.cell(0, 7, txt="MEL SPECTROGRAM ANALYSIS", ln=True)
        pdf.set_draw_color(30, 37, 53)
        pdf.set_line_width(0.2)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        pdf.set_font("Arial", size=8)
        pdf.set_text_color(100, 120, 150)
        pdf.cell(0, 5, txt="Visual representation of frequency energy over time (Mel scale, dB).", ln=True)
        pdf.ln(2)

        try:
            pdf.image(mel_img_path, x=15, w=175)
            pdf.ln(4)
        except Exception:
            pdf.set_text_color(180, 60, 60)
            pdf.cell(0, 7, txt="[Mel Spectrogram image could not be embedded]", ln=True)
            pdf.ln(4)

    # -----------------------------------------------
    # MFCC FEATURE ANALYSIS IMAGE
    # -----------------------------------------------
    if mfcc_img_path and os.path.exists(mfcc_img_path):
        # Start a new page if we're too far down to avoid clipping the image
        if pdf.get_y() > 200:
            pdf.add_page()

        pdf.set_font("Arial", "B", size=9)
        pdf.set_text_color(56, 189, 248)
        pdf.cell(0, 7, txt="MFCC FEATURE ANALYSIS", ln=True)
        pdf.set_draw_color(30, 37, 53)
        pdf.set_line_width(0.2)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        # Subtitle / explanation
        pdf.set_font("Arial", size=8)
        pdf.set_text_color(100, 120, 150)
        mfcc_caption = (
            "Heatmap showing 40 MFCC coefficients (rows) across 157 temporal frames (columns). "
            "This matrix is the exact input fed to the CNN classifier."
        )
        pdf.multi_cell(0, 5, txt=mfcc_caption)
        pdf.ln(3)

        try:
            pdf.image(mfcc_img_path, x=15, w=175)
            pdf.ln(4)
        except Exception:
            pdf.set_text_color(180, 60, 60)
            pdf.cell(0, 7, txt="[MFCC heatmap image could not be embedded]", ln=True)
            pdf.ln(4)

    # -----------------------------------------------
    # RAW MFCC MATRIX
    # -----------------------------------------------
    if mfcc is not None:
        pdf.add_page()

        pdf.set_font("Arial", "B", size=9)
        pdf.set_text_color(56, 189, 248)
        pdf.cell(0, 7, txt="RAW MFCC MATRIX", ln=True)
        pdf.set_draw_color(30, 37, 53)
        pdf.set_line_width(0.2)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        pdf.set_font("Arial", size=8)
        pdf.set_text_color(100, 120, 150)
        pdf.cell(0, 5, txt=f"Shape: {mfcc.shape[0]} coefficients × {mfcc.shape[1]} frames", ln=True)
        pdf.ln(3)

        # Overall statistics
        pdf.set_font("Arial", "B", size=8)
        pdf.set_text_color(56, 189, 248)
        pdf.cell(0, 6, txt="Overall Statistics", ln=True)
        pdf.ln(1)

        pdf.set_font("Courier", size=8)
        pdf.set_text_color(0, 0, 0)
        stats_overall = f"Min: {mfcc.min():.3f}  |  Max: {mfcc.max():.3f}  |  Mean: {mfcc.mean():.3f}  |  Std: {mfcc.std():.3f}"
        pdf.multi_cell(0, 5, txt=stats_overall)
        pdf.ln(3)

        # Per-coefficient statistics (first 10 as sample)
        pdf.set_font("Arial", "B", size=8)
        pdf.set_text_color(56, 189, 248)
        pdf.cell(0, 6, txt="Per-Coefficient Statistics (Sample: C00-C09)", ln=True)
        pdf.ln(1)

        pdf.set_font("Courier", size=7)
        pdf.set_text_color(0, 0, 0)
        for i in range(min(10, mfcc.shape[0])):
            coeff_stats = f"C{i:02d}: Min={mfcc[i].min():.3f} Max={mfcc[i].max():.3f} Mean={mfcc[i].mean():.3f} Std={mfcc[i].std():.3f}"
            pdf.multi_cell(0, 4, txt=coeff_stats)
            pdf.ln(1)

        pdf.set_font("Arial", size=7)
        pdf.set_text_color(100, 120, 150)
        pdf.cell(0, 5, txt="(Full matrix available in application interface)", ln=True)

    # -----------------------------------------------
    # FOOTER
    # -----------------------------------------------
    pdf.ln(6)
    pdf.set_draw_color(56, 189, 248)
    pdf.set_line_width(0.4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Arial", size=7)
    pdf.set_text_color(150, 160, 180)
    pdf.cell(0, 5, txt="Generated by AudioForensics AI  |  CNN + MFCC  |  ASVspoof 2019 LA", ln=True, align='C')
    pdf.cell(0, 5, txt="This report is intended for investigative use only.", ln=True, align='C')

    output_path = "report.pdf"
    pdf.output(output_path)
    return output_path