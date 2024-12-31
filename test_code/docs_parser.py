from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import os

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_word(file_path: str) -> str:
    doc = DocxDocument(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def save_to_txt(text, file_name, output_dir):
    output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

directory = r"C:\Users\Uvais\Documents\coding\streamlit\docs"
output_directory = r"C:\Users\Uvais\Documents\coding\streamlit\extracted_text"  # Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

for file_name in os.listdir(directory):
    file_path = os.path.join(directory, file_name)
    if file_name.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_name.endswith(".docx"):
        text = extract_text_from_word(file_path)
    else:
        continue

    save_to_txt(text, file_name, output_directory)

print(f"Text extracted and saved to: {output_directory}")