from pypdf import PdfReader
from docx import Document
import re
import pypinyin

def doc_file(uploaded_file):
    if not uploaded_file: return ""
    ext = uploaded_file.name.split('.')[-1].lower()
    try:
        if ext == "pdf":
            reader = PdfReader(uploaded_file)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        elif ext == "docx":
            doc = Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext in ["txt", "md", "html"]:
            return str(uploaded_file.read(), "utf-8")
    except:
        return ""
    return ""

def clean_pdf_text(text):
    if not text: return ""
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('•', '•')
    # common PDF split fixes
    text = text.replace('impor tant', 'important').replace('scienti c', 'scientific')
    return text.strip()

def split_smart_chunks(text, chunk_size=1500, max_total_chars=50000):
    if not text:
        return []
    if len(text) > max_total_chars:
        text = text[:max_total_chars]
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += s + " "
        else:
            if current:
                chunks.append(current.strip())
            current = s + " "
    if current:
        chunks.append(current.strip())
    return chunks

def convert_to_pinyin(text):
    if not text: return ""
    if any('\u4e00' <= ch <= '\u9fff' for ch in text):
        try:
            return ' '.join([i[0] for i in pypinyin.pinyin(text, style=pypinyin.TONE)])
        except:
            return ""
    return ""
