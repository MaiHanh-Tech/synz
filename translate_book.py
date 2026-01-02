import pypinyin
import re
import os
import sys
import time
import streamlit as st
from translator import Translator
from concurrent.futures import ThreadPoolExecutor

# Prompt xử lý văn bản
EXPERT_PROMPT = """Bạn là chuyên gia dịch thuật. Hãy dịch đoạn văn bản sau.
Yêu cầu bắt buộc:
1. Nối các từ bị ngắt quãng do lỗi PDF (ví dụ: 'impor tant' -> 'important') trước khi dịch.
2. Dịch mượt mà, văn phong học thuật tự nhiên.
3. KHÔNG trả lời hay giải thích, chỉ đưa ra bản dịch.
"""

def clean_pdf_text(text):
    """Tiền xử lý văn bản PDF"""
    if not text: return ""
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('•', 'ï').replace('impor tant', 'important').replace('scienti c', 'scientific')
    return text.strip()

# ✅ HÀM ĐÃ SỬA THEO YÊU CẦU
def split_smart_chunks(text, chunk_size=1500, max_total_chars=50000):
    """
    Chia văn bản thành chunks lớn
    ✅ THÊM: Giới hạn tổng số ký tự để tránh crash
    """
    if not text: 
        return []
    
    # ✅ CẮT BỎ PHẦN DƯ THỪA
    if len(text) > max_total_chars:
        text = text[:max_total_chars]
        st.warning(f"⚠️ File quá dài. Chỉ dịch {max_total_chars:,} ký tự đầu.")
    
    # Logic chia chunks cũ (GIỮ NGUYÊN)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

def convert_to_pinyin(text):
    if not text: return ""
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        try:
            return ' '.join([i[0] for i in pypinyin.pinyin(text, style=pypinyin.TONE)])
        except: return ""
    return ""

def process_chunk(chunk, index, translator, include_english, source, target):
    try:
        pinyin_text = convert_to_pinyin(chunk) if source == "Chinese" else ""
        main_trans = translator.translate_text(chunk, source, target, EXPERT_PROMPT)
        
        if target == "Chinese" and not pinyin_text:
            pinyin_text = convert_to_pinyin(main_trans)

        eng_trans = ""
        if include_english:
            if target == "English": eng_trans = "" 
            elif source == "English": eng_trans = chunk
            else: eng_trans = translator.translate_text(chunk, source, "English", "Translate to English.")

        return (index, chunk, pinyin_text, eng_trans, main_trans)
    except Exception as e:
        return (index, chunk, "", "[Error]", f"[System Error: {str(e)}]")

def create_html_block(results, include_english):
    index, chunk, pinyin, english, second = results
    
    speak_btn = """<button class="speak-button" onclick="speakSentence(this.parentElement.textContent.replace('🔊', ''))"><svg viewBox="0 0 24 24"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/></svg></button>"""
    
    html = f'<div class="sentence-part responsive">'
    html += f'<div class="original"><strong>[{index + 1}]</strong> {chunk}{speak_btn}</div>'
    
    if pinyin: html += f'<div class="pinyin">{pinyin}</div>'
    if include_english and english: html += f'<div class="english">{english}</div>'
    
    if "[System Busy" in second or "[API Error" in second:
        html += f'<div class="second-language" style="color: red; border: 1px solid red; padding: 5px;">⚠️ {second}</div>'
    else:
        html += f'<div class="second-language">{second}</div>'
    
    html += '</div>'
    return html

def create_interactive_html_block(processed_words):
    html = '<div class="interactive-text"><p class="interactive-paragraph">'
    for item in processed_words:
        word = item.get('word', '')
        if word == '\n':
            html += '</p><p class="interactive-paragraph">'
            continue
        
        translations = item.get('translations', [])
        meaning = translations[0] if translations else ""
        pinyin_val = item.get('pinyin', '')
        
        safe_word = word.replace("'", "\\'")
        tooltip = f"{pinyin_val}\\n{meaning}"
        
        html += f"""<span class="interactive-word" onclick="speak('{safe_word}')" data-tooltip="{tooltip}">{word}</span>"""
    html += '</p></div>'
    return html

def translate_file(input_text, status_placeholder=None, progress_bar=None, include_english=True, 
                  source_lang="Chinese", target_lang="Vietnamese", 
                  translation_mode="Standard Translation", processed_words=None):
    
    # Mapping tên ngôn ngữ sang mã ISO để trình duyệt hiểu (voice filter)
    lang_map = {
        "Chinese": "zh",
        "English": "en",
        "Vietnamese": "vi"
    }
    # Mặc định là 'zh' nếu không tìm thấy
    voice_lang_code = lang_map.get(source_lang, "zh")

    # Mode 1: Interactive
    if translation_mode == "Interactive Word-by-Word" and processed_words:
        try:
            with open('template.html', 'r', encoding='utf-8') as f: template = f.read()
        except: template = "<body>{{content}}</body>"
        
        content = create_interactive_html_block(processed_words)
        # Thay thế biến {{voice_lang}} trong template
        template = template.replace('{{voice_lang}}', voice_lang_code)
        return template.replace('{{content}}', content)

    # Mode 2: Standard
    translator = Translator()
    clean_text = clean_pdf_text(input_text)
    chunks = split_smart_chunks(clean_text)
    total = len(chunks)
    
    html_body = '<div class="translation-block">'
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            future = executor.submit(process_chunk, chunk, i, translator, include_english, source_lang, target_lang)
            futures.append(future)
        
        results = []
        for i, future in enumerate(futures):
            res = future.result()
            results.append(res)
            time.sleep(1) 
            if progress_bar: progress_bar.progress((i+1)/total)
            
    for res in results:
        html_body += create_html_block(res, include_english)
            
    html_body += '</div>'

    try:
        with open('template.html', 'r', encoding='utf-8') as f: template = f.read()
    except: template = "<body>{{content}}</body>"
    
    css_fix = """<script>
    (function(){
        function s(){document.body.setAttribute('data-theme', window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');}
        s(); window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', s);
    })();
    </script></body>"""
    
    # Thay thế biến {{voice_lang}} trước
    template = template.replace('{{voice_lang}}', voice_lang_code)
    
    full_html = template.replace('{{content}}', html_body)
    if "</body>" in full_html:
        full_html = full_html.replace("</body>", css_fix)
    else:
        full_html += css_fix
        
    return full_html
