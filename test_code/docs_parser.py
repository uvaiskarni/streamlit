from PyPDF2 import PdfReader
from docx import Document as DocxDocument

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

import os
documents = []
directory = r"C:\Users\Uvais\Documents\coding\streamlit\docs"   

for file_name in os.listdir(directory):
    file_path = os.path.join(directory, file_name)
    if file_name.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_name.endswith(".docx"):
        text = extract_text_from_word(file_path)
    else:
        continue

    documents.append(dict(
            page_content=text, 
            metadata={"source": file_name}
        ))


print(documents[0])