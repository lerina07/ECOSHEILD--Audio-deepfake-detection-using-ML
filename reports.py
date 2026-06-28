from fpdf import FPDF
from datetime import datetime

def generate_report(file_name, prediction, real_conf, fake_conf, duration):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="AUDIO FORENSIC REPORT", ln=True, align='C')
    pdf.ln(5)

    pdf.cell(200, 10, txt=f"Case Number: DF-{datetime.now().strftime('%Y%m%d%H%M%S')}", ln=True)
    pdf.cell(200, 10, txt=f"File Name: {file_name}", ln=True)
    pdf.cell(200, 10, txt=f"File Type: Audio", ln=True)
    pdf.cell(200, 10, txt=f"Location: Uploaded File", ln=True)
    pdf.cell(200, 10, txt=f"Stakeholder: Investigator", ln=True)

    pdf.cell(200, 10, txt=f"Uploaded: {datetime.now()}", ln=True)
    pdf.cell(200, 10, txt=f"Analysis Time: {datetime.now()}", ln=True)
    pdf.cell(200, 10, txt=f"Duration: {duration:.2f} sec", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="--- PREPROCESSING ---", ln=True)
    pdf.cell(200, 10, txt="Feature: MFCC (40)", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="--- RESULT ---", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {prediction}", ln=True)
    pdf.cell(200, 10, txt=f"Real: {real_conf:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Fake: {fake_conf:.2f}%", ln=True)

    pdf.output("report.pdf")
    return "report.pdf"