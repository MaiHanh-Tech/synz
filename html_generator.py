import os

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "template.html")

def load_template():
    try:
        with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        # minimal fallback
        return "<html><body>{{content}}</body></html>"

def create_html_block(index, chunk, pinyin, english, second, include_english=True):
    speak_btn = """<button class="speak-button" onclick="speakSentence(this.parentElement.textContent.replace('🔊', ''))">🔊</button>"""
    html = f'<div class="sentence-part responsive"><div class="original"><strong>[{index + 1}]</strong> {chunk}{speak_btn}</div>'
    if pinyin:
        html += f'<div class="pinyin">{pinyin}</div>'
    if include_english and english:
        html += f'<div class="english">{english}</div>'
    if "[System Busy" in second or "[API Error" in second:
        html += f'<div class="second-language" style="color: red;">⚠️ {second}</div>'
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
