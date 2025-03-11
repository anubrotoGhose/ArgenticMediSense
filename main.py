import streamlit as st
import pdfplumber
import pandas as pd
import docx
import xml.etree.ElementTree as ET
from PIL import Image
import pytesseract  # OCR for images
import ollama

def extract_text(uploaded_file):
    """Extracts text from uploaded files based on type."""
    if uploaded_file is None:
        return None  # No file uploaded
    
    file_type = uploaded_file.type

    if file_type == "text/plain":  # TXT
        return uploaded_file.read().decode("utf-8")

    elif file_type == "application/pdf":  # PDF
        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    elif file_type in ["image/png", "image/jpeg", "image/webp"]:  # Images
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)  # Extract text using OCR
        return text.strip() if text else "No text detected in image."

    elif file_type in ["text/csv", "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:  # CSV/XLS
        df = pd.read_csv(uploaded_file) if file_type == "text/csv" else pd.read_excel(uploaded_file)
        return df.to_csv(index=False)  # Convert DataFrame to CSV text

    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # DOCX
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file_type == "text/xml":  # XML
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
        return ET.tostring(root, encoding="utf-8").decode()

    elif file_type == "text/html":  # HTML
        return uploaded_file.read().decode("utf-8")

    return "Unsupported file type."


def translate_text(text, target_lang="English"):
    prompt = f"Translate this text to {target_lang}:\n\n{text}"
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Streamlit UI
st.title("File Uploader & Text Extractor")

uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "png", "jpg", "jpeg", "csv", "xlsx", "xls", "docx", "xml", "html"])


manual_text = st.text_area("Or enter text manually", "")

if uploaded_file or manual_text:
    if uploaded_file:
        extracted_text = extract_text(uploaded_file)
    
    else:
        extracted_text = manual_text

    st.text_area("Extracted Text", extracted_text, height=300)
    print(extracted_text)
    translate_text = translate_text(extracted_text)

    print("Translated")
    print(translate_text)
    st.text_area("Translated Text", translate_text, height=300)