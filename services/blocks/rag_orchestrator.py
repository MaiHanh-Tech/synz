"""
Robust RAG orchestrator helpers used by UI modules.

Exports:
- analyze_document_streamlit(title, text, user_lang, max_chars)
- compute_similarity_with_excel(text, excel_df, encoder=None, top_k=3)
- store_history(loai, title, content)
- init_knowledge_universe()
- create_personal_rag(supabase_client, user_id)
- tai_lich_su(limit=50)

✅ NEW: Translation orchestrator
- get_translation_orchestrator()
- TranslationOrchestrator class

✅ UPDATED: Knowledge Universe seeding with selected high-quality books
"""

from typing import List, Tuple, Any, Optional, Callable
import streamlit as st
import traceback
import time

# Local blocks
from services.blocks.embedding_engine import load_encoder, encode_texts
from services.blocks import knowledge_graph_v2 as kg_module
from services.blocks import personal_rag_system as pr_module

from ai_core import AI_Core
from prompts import BOOK_ANALYSIS_PROMPT

# Translator (for translation orchestrator)
try:
    from translator import Translator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

# sklearn (optional)
try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    cosine_similarity = None

# Supabase client (try import)
try:
    from supabase import create_client
except Exception:
    create_client = None

# Helper: create supabase client from st.secrets if available
def _get_supabase_client():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    except Exception:
        return None
    if not create_client:
        return None
    try:
        return create_client(url, key)
    except Exception:
        # Some environments may expose a different factory; return None
        return None


# -------------------------
# Core helpers / wrappers
# -------------------------
def analyze_document_streamlit(title: str, text: str, user_lang: str = "vi", max_chars: int = 30000) -> str:
    """
    Thin wrapper around AI_Core.analyze_static for UI.
    Returns AI text (or error string).
    """
    try:
        ai = AI_Core()
        content = text[:max_chars]
        return ai.analyze_static(content, BOOK_ANALYSIS_PROMPT)
    except Exception as e:
        return f"❌ Lỗi phân tích: {e}"


def compute_similarity_with_excel(text: str, excel_df, encoder=None, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Encode text and dataframe of books (column 'Tên sách' and optional 'CẢM NHẬN'),
    return list of (title, similarity) top_k matches.
    """
    if excel_df is None or excel_df.empty:
        return []

    try:
        enc = encoder or load_encoder()
        if enc is None or cosine_similarity is None:
            return []

        db_texts = [f"{r['Tên sách']} {str(r.get('CẢM NHẬN',''))}" for _, r in excel_df.iterrows()]
        emb_db = encode_texts(enc, db_texts)
        q_emb = encode_texts(enc, [text[:2000]])[0]
        sims = cosine_similarity([q_emb], emb_db)[0]
        import numpy as np
        idx = np.argsort(sims)[::-1][:top_k]
        matches = []
        for i in idx:
            if sims[i] > 0.0:
                matches.append((excel_df.iloc[i]["Tên sách"], float(sims[i])))
        return matches
    except Exception:
        return []


# -------------------------
# History / Supabase helpers
# -------------------------
def _try_insert_supabase(client, table: str, data: dict) -> bool:
    """Try common supabase client APIs to insert data."""
    if not client:
        return False

    try:
        if hasattr(client, "table"):
            resp = client.table(table).insert(data).execute()
            err = getattr(resp, "error", None) or (resp.get("error") if isinstance(resp, dict) else None)
            status = getattr(resp, "status_code", None) or (resp.get("status_code") if isinstance(resp, dict) else None)
            if not err and (status is None or int(status) in (200, 201, 204)):
                return True
    except Exception:
        pass

    try:
        if hasattr(client, "from_"):
            resp = client.from_(table).insert(data).execute()
            err = getattr(resp, "error", None) or (resp.get("error") if isinstance(resp, dict) else None)
            status = getattr(resp, "status_code", None) or (resp.get("status_code") if isinstance(resp, dict) else None)
            if not err and (status is None or int(status) in (200, 201, 204)):
                return True
    except Exception:
        pass

    return False


def store_history(loai: str, title: str, content: str) -> bool:
    """Store a history record in Supabase. Returns True on success."""
    client = _get_supabase_client()
    if not client:
        return False

    user = st.session_state.get("current_user", "Unknown")
    data = {
        "type": loai,
        "title": title,
        "content": content,
        "user_name": user,
        "sentiment_score": 0.0,
        "sentiment_label": "Neutral"
    }

    table_names = ["history_logs", "History_Logs", "historylogs", "historyLogs"]
    last_exc = None
    for table in table_names:
        try:
            ok = _try_insert_supabase(client, table, data)
            if ok:
                return True
        except Exception as e:
            last_exc = e
            continue

    if last_exc:
        traceback.print_exception(type(last_exc), last_exc, last_exc.__traceback__)
    else:
        print("store_history: insert attempts failed without exception.")
    return False


def _try_select_supabase(client, table: str, limit: int = 50):
    """Try to select rows from supabase using common client APIs."""
    if not client:
        return None
    try:
        if hasattr(client, "table"):
            resp = client.table(table).select("*").order("created_at", desc=True).limit(limit).execute()
            data = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else None)
            return data
    except Exception:
        pass

    try:
        if hasattr(client, "from_"):
            resp = client.from_(table).select("*").order("created_at", desc=True).limit(limit).execute()
            data = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else None)
            return data
    except Exception:
        pass

    return None


def tai_lich_su(limit: int = 50) -> List[dict]:
    """Fetch history records from Supabase and normalize fields."""
    client = _get_supabase_client()
    if not client:
        return []

    table_names = ["history_logs", "History_Logs", "historylogs", "historyLogs"]
    raw_data = None
    for table in table_names:
        try:
            raw_data = _try_select_supabase(client, table, limit=limit)
            if raw_data:
                break
        except Exception:
            continue

    if not raw_data:
        return []

    formatted = []
    for item in raw_data:
        def get_val(keys, default=""):
            for k in keys:
                if k in item and item[k] is not None:
                    return item[k]
            return default

        raw_time = get_val(["created_at", "createdAt", "Time", "time"], "")
        clean_time = str(raw_time).replace("T", " ")[:19] if raw_time else ""

        formatted.append({
            "Time": clean_time,
            "Type": get_val(["type", "Type"], ""),
            "Title": get_val(["title", "Title"], ""),
            "Content": get_val(["content", "Content"], ""),
            "User": get_val(["user_name", "User", "user"], "Unknown"),
            "SentimentScore": get_val(["sentiment_score", "SentimentScore"], 0.0),
            "SentimentLabel": get_val(["sentiment_label", "SentimentLabel"], "Neutral")
        })

    return formatted


# -------------------------
# Re-exports / factories
# -------------------------
def init_knowledge_universe():
    """Re-export init from knowledge_graph_v2."""
    try:
        return kg_module.init_knowledge_universe()
    except Exception:
        return None


def create_personal_rag(supabase_client, user_id: str):
    """Factory to create PersonalRAG instance."""
    try:
        return pr_module.PersonalRAG(supabase_client, user_id)
    except Exception:
        return None


# ========================================
# ✅ NEW: TRANSLATION ORCHESTRATOR
# ========================================

class TranslationOrchestrator:
    """Translation orchestrator - Integrated into RAG orchestrator."""

    def __init__(self):
        if not HAS_TRANSLATOR:
            raise ImportError("Thiếu translator.py - cần có file này để dịch")

        self.translator = Translator()

        self.lang_map = {
            "Chinese": "zh",
            "English": "en",
            "Vietnamese": "vi",
            "French": "fr",
            "Japanese": "ja",
            "Korean": "ko"
        }

    def translate_document(
        self,
        input_text: str,
        source_lang: str = "Chinese",
        target_lang: str = "Vietnamese",
        include_english: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Standard translation mode."""
        if not input_text.strip():
            return self._generate_error_html("Không có nội dung để dịch")

        try:
            if progress_callback:
                progress_callback(0.1)

            prompt = f"""Translate the following text from {source_lang} to {target_lang}.
Requirements:
1. Maintain natural flow and readability.
2. Preserve technical terms when appropriate.
3. Output only the translation, no explanations.

Text:
{input_text}"""

            main_translation = self.translator.translate_text(
                input_text,
                source_lang,
                target_lang,
                prompt
            )

            if progress_callback:
                progress_callback(0.6)

            english_translation = ""
            if include_english and target_lang != "English":
                if source_lang == "English":
                    english_translation = input_text
                else:
                    try:
                        english_translation = self.translator.translate_text(
                            input_text,
                            source_lang,
                            "English",
                            "Translate to English."
                        )
                    except Exception:
                        pass

            if progress_callback:
                progress_callback(0.9)

            html = self._generate_html(
                input_text,
                main_translation,
                english_translation,
                source_lang,
                target_lang
            )

            if progress_callback:
                progress_callback(1.0)

            return html

        except Exception as e:
            return self._generate_error_html(f"Lỗi dịch: {str(e)}")

    def translate_interactive(
        self,
        input_text: str,
        source_lang: str = "Chinese",
        target_lang: str = "Vietnamese"
    ) -> str:
        """Interactive word-by-word mode - only for Chinese source."""
        if source_lang != "Chinese":
            return self._generate_error_html("Interactive mode chỉ hỗ trợ nguồn Tiếng Trung")

        try:
            words = self.translator.process_word_by_word(
                input_text,
                source_lang,
                target_lang
            )
            return self._generate_interactive_html(words, source_lang)
        except Exception as e:
            return self._generate_error_html(f"Lỗi: {str(e)}")

    def _generate_html(self, original: str, translation: str, english: str, source_lang: str, target_lang: str) -> str:
        """Generate HTML for standard mode."""
        voice_lang = self.lang_map.get(source_lang, "en")

        html = f"""<!DOCTYPE html>
<html lang="{voice_lang}">
<head>
 <meta charset="UTF-8">
 <meta name="viewport" content="width=device-width, initial-scale=1.0">
 <title>Translation: {source_lang} → {target_lang}</title>
 <style>
  body {{
   font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
   padding: 30px;
   max-width: 900px;
   margin: 0 auto;
   background: #f5f7fa;
   line-height: 1.8;
  }}
  .header {{
   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
   color: white;
   padding: 25px;
   border-radius: 12px;
   margin-bottom: 30px;
   box-shadow: 0 8px 16px rgba(102,126,234,0.2);
  }}
  .header h1 {{
   margin: 0 0 10px 0;
   font-size: 28px;
  }}
  .section {{
   background: white;
   padding: 25px;
   margin-bottom: 20px;
   border-radius: 10px;
   box-shadow: 0 2px 8px rgba(0,0,0,0.06);
   border-left: 4px solid;
  }}
  .label {{
   font-weight: 600;
   color: #4a5568;
   font-size: 13px;
   text-transform: uppercase;
   letter-spacing: 0.5px;
   margin-bottom: 12px;
  }}
  .content {{
   color: #2d3748;
   font-size: 16px;
   white-space: pre-wrap;
  }}
  .original {{ border-left-color: #667eea; }}
  .translation {{ border-left-color: #764ba2; }}
  .english {{ border-left-color: #48bb78; }}
  .footer {{
   text-align: center;
   color: #a0aec0;
   font-size: 12px;
   margin-top: 40px;
  }}
 </style>
</head>
<body>
 <div class="header">
  <h1>Translation</h1>
  <p>{source_lang} → {target_lang}</p>
 </div>

 <div class="section original">
  <div class="label">Original ({source_lang})</div>
  <div class="content">{original}</div>
 </div>

 <div class="section translation">
  <div class="label">Translation ({target_lang})</div>
  <div class="content">{translation}</div>
 </div>
"""

        if english:
            html += f"""
 <div class="section english">
  <div class="label">English Reference</div>
  <div class="content">{english}</div>
 </div>
"""

        html += """
 <div class="footer">
  Generated by AI Translator Pro
 </div>
</body>
</html>"""

        return html

    def _generate_interactive_html(self, words: list, source_lang: str) -> str:
        """Generate interactive HTML with hover tooltips."""
        voice_lang = self.lang_map.get(source_lang, "zh")

        html = f"""<!DOCTYPE html>
<html lang="{voice_lang}">
<head>
 <meta charset="UTF-8">
 <meta name="viewport" content="width=device-width, initial-scale=1.0">
 <title>Interactive Translation</title>
 <style>
  body {{
   font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
   padding: 30px;
   max-width: 900px;
   margin: 0 auto;
   background: #f5f7fa;
  }}
  .header {{
   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
   color: white;
   padding: 25px;
   border-radius: 12px;
   margin-bottom: 30px;
   text-align: center;
  }}
  .content-box {{
   background: white;
   padding: 40px;
   border-radius: 12px;
   box-shadow: 0 2px 8px rgba(0,0,0,0.06);
   line-height: 2.2;
   font-size: 18px;
  }}
  .interactive-word {{
   display: inline-block;
   padding: 2px 6px;
   margin: 2px;
   cursor: pointer;
   border-bottom: 2px dotted #667eea;
   position: relative;
   transition: all 0.2s;
  }}
  .interactive-word:hover {{
   background: #667eea;
   color: white;
   border-radius: 4px;
   transform: translateY(-2px);
  }}
  .interactive-word:hover::after {{
   content: attr(data-tooltip);
   position: absolute;
   bottom: 100%;
   left: 50%;
   transform: translateX(-50%);
   background: rgba(0,0,0,0.92);
   color: white;
   padding: 10px 15px;
   border-radius: 8px;
   white-space: pre-line;
   font-size: 14px;
   z-index: 1000;
   margin-bottom: 8px;
   min-width: 180px;
   text-align: center;
   box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }}
 </style>
</head>
<body>
 <div class="header">
  <h1>Interactive Translation</h1>
  <p>Hover over words to see meaning</p>
 </div>

 <div class="content-box">
"""

        for item in words:
            word = item.get('word', '')
            pinyin = item.get('pinyin', '')
            translations = item.get('translations', [])
            meaning = translations[0] if translations else ""

            safe_word = word.replace("'", "\\'").replace('"', '&quot;')
            tooltip = f"{pinyin}\\n{meaning}"

            html += f'<span class="interactive-word" data-tooltip="{tooltip}">{word}</span>'

        html += """
 </div>
</body>
</html>"""

        return html

    def _generate_error_html(self, error_msg: str) -> str:
        """Generate error HTML."""
        return f"""<!DOCTYPE html>
<html>
<head>
 <meta charset="UTF-8">
 <title>Error</title>
 <style>
  body {{
   font-family: sans-serif;
   padding: 40px;
   text-align: center;
  }}
  .error {{
   color: #e53e3e;
   font-size: 18px;
   padding: 20px;
   border: 2px solid #e53e3e;
   border-radius: 8px;
   background: #fff5f5;
  }}
 </style>
</head>
<body>
 <div class="error">
  <h2>❌ Lỗi</h2>
  <p>{error_msg}</p>
 </div>
</body>
</html>"""


# Factory function
@st.cache_resource
def get_translation_orchestrator():
    """Get singleton translation orchestrator instance."""
    if not HAS_TRANSLATOR:
        return None
    try:
        return TranslationOrchestrator()
    except Exception:
        return None
