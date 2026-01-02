"""
MODULE TRANSLATOR - Giao diện dịch thuật
Version: Final (FIXED - Sửa lỗi import)
"""

import streamlit as st
import streamlit.components.v1 as components
import time
from typing import Optional

# ===== IMPORTS BLOCKS (with fallback) =====
try:
    from services.blocks.rag_orchestrator import get_translation_orchestrator
    HAS_ORCHESTRATOR = True  # ✅ FIX 1: Thêm dòng này
except ImportError:
    HAS_ORCHESTRATOR = False
    # Chạy yên lặng, không warning

try:
    from services.blocks.text_processor import get_text_processor  # ✅ FIX 2: Sửa path
    HAS_TEXT_PROCESSOR = True
except ImportError:
    HAS_TEXT_PROCESSOR = False

# Fallback: AI Core nếu không có orchestrator
if not HAS_ORCHESTRATOR:
    from ai_core import AI_Core

# ===== CONSTANTS =====
LANGUAGES = {
    "Vietnamese": "Tiếng Việt",
    "English": "English",
    "Chinese": "中文",
    "French": "Français",
    "Japanese": "日本語",
    "Korean": "한국어"
}

STYLE_OPTIONS = {
    "Văn học": "Write in a literary style with rich imagery and elegant phrasing.",
    "Khoa học": "Write in a scientific/technical style, precise and formal.",
    "Đời thường": "Write in a casual, conversational everyday style.",
    "Hàn lâm": "Write in an academic style with formal tone.",
    "Thương mại": "Write in a business style, concise and professional."
}

# ===== MAIN FUNCTION =====
def run():
    """
    Hàm chính để app.py gọi
    
    Features:
    - Multi-language support
    - Style customization
    - Progress tracking
    - HTML export
    - Cost estimation
    """
    
    st.header("🌏 AI Translator Pro")
    st.caption("Dịch văn bản đa ngôn ngữ với nhiều phong cách")
    
    # ========== CONFIGURATION ==========
    st.subheader("⚙️ Cấu hình")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_lang = st.selectbox(
            "Ngôn ngữ nguồn:",
            ["Chinese", "English", "Vietnamese"],
            index=0,
            help="Ngôn ngữ của văn bản gốc"
        )
    
    with col2:
        target_lang = st.selectbox(
            "Ngôn ngữ đích:",
            list(LANGUAGES.keys()),
            index=0,
            help="Ngôn ngữ muốn dịch sang"
        )
    
    with col3:
        style = st.selectbox(
            "Phong cách dịch:",
            list(STYLE_OPTIONS.keys()),
            index=0,
            help="Chọn phong cách văn phong"
        )
    
    # Mode selection (chỉ hiện khi nguồn là Chinese)
    if source_lang == "Chinese":
        mode = st.radio(
            "Chế độ dịch:",
            ["Standard (Dịch câu)", "Interactive (Học từ)"],
            horizontal=True,
            help="Standard: Dịch cả đoạn. Interactive: Hover để xem nghĩa từng từ (chỉ Chinese)"
        )
    else:
        mode = "Standard (Dịch câu)"
        if source_lang == "English":
            st.info("💡 Chế độ Interactive chỉ hỗ trợ nguồn Tiếng Trung")
    
    include_english = st.checkbox(
        "📖 Kèm Tiếng Anh",
        value=True,
        help="Hiển thị thêm bản dịch Tiếng Anh để đối chiếu (giúp học ngôn ngữ)"
    )
    
    st.divider()
    
    # ========== INPUT ==========
    st.subheader("📝 Nhập văn bản")
    
    text_input = st.text_area(
        "Dán văn bản cần dịch:",
        height=250,
        placeholder="Nhập hoặc dán văn bản vào đây...",
        help="Hỗ trợ văn bản dài, tự động chia chunks"
    )
    
    # ========== COST ESTIMATION ==========
    if text_input and HAS_TEXT_PROCESSOR:
        text_proc = get_text_processor()
        cost_info = text_proc.estimate_translation_cost(
            text_input,
            include_english,
            target_lang
        )
        
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Số ký tự", f"{cost_info['total_chars']:,}")
        col_info2.metric("Số đoạn", cost_info['num_chunks'])
        col_info3.metric("API calls", cost_info['estimated_api_calls'])
        
        if cost_info.get('warning'):
            st.warning(cost_info['warning'])
    
    st.divider()
    
    # ========== TRANSLATE BUTTON ==========
    if st.button("🚀 Dịch Ngay", type="primary", use_container_width=True):
        
        # Validate input
        if not text_input.strip():
            st.error("❌ Chưa nhập văn bản!")
            return
        
        # Validate mode
        if mode == "Interactive (Học từ)" and source_lang != "Chinese":
            st.error("❌ Chế độ Interactive chỉ hỗ trợ nguồn Tiếng Trung")
            return
        
        # ========== TRANSLATION PROCESS ==========
        progress_bar = st.progress(0, text="Đang khởi động...")
        status_text = st.empty()
        
        html_output = None
        translated_text = None
        
        try:
            if HAS_ORCHESTRATOR:
                # ===== USE ORCHESTRATOR (Preferred) =====
                orch = get_translation_orchestrator()
                
                if mode == "Interactive (Học từ)":
                    status_text.text("🔄 Đang phân tích từ vựng...")
                    
                    html_output = orch.translate_interactive(
                        text_input,
                        source_lang,
                        target_lang
                    )
                    
                else:  # Standard mode
                    status_text.text("🔄 Đang dịch văn bản...")
                    
                    def update_progress(value):
                        progress_bar.progress(value, text=f"🔄 Đang dịch... {int(value*100)}%")
                    
                    html_output = orch.translate_document(
                        text_input,
                        source_lang,
                        target_lang,
                        include_english,
                        progress_callback=update_progress
                    )
            
            else:
                # ===== FALLBACK: Direct AI Call =====
                status_text.text("🔄 Đang dịch (chế độ fallback)...")
                
                ai = AI_Core()
                style_instr = STYLE_OPTIONS.get(style, "")
                
                prompt = f"""Translate the following text into {LANGUAGES[target_lang]}.
Style instructions: {style_instr}

Text:
{text_input}"""
                
                translated_text = ai.generate(prompt, model_type="pro")
                
                # Create simple HTML
                html_output = f"""<!DOCTYPE html>
<html lang="{target_lang.lower()[:2]}">
<head>
    <meta charset="UTF-8">
    <title>Translation - {style}</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; max-width: 900px; margin: 0 auto; }}
        h2 {{ color: #333; }}
        .translation {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
    </style>
</head>
<body>
    <h2>Translation: {source_lang} → {LANGUAGES[target_lang]}</h2>
    <p><strong>Style:</strong> {style}</p>
    <div class="translation">{translated_text}</div>
</body>
</html>"""
            
            # ========== SUCCESS ==========
            progress_bar.progress(1.0, text="✅ Hoàn thành!")
            status_text.success("🎉 Dịch xong! Cuộn xuống để xem kết quả.")
            
            st.balloons()
            
            # ========== DOWNLOAD BUTTON ==========
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"translation_{source_lang}_to_{target_lang}_{style.replace(' ', '_')}_{timestamp}.html"
            
            st.download_button(
                label="📥 Tải file HTML",
                data=html_output.encode('utf-8'),
                file_name=filename,
                mime="text/html",
                use_container_width=True
            )
            
            # ========== DISPLAY RESULTS ==========
            st.divider()
            st.subheader("📄 Kết quả dịch thuật")
            
            # Show translated text if in fallback mode
            if translated_text:
                st.markdown(translated_text)
                st.divider()
            
            # Show HTML preview
            with st.expander("🔍 Xem trước HTML (Click để mở)", expanded=True):
                components.html(html_output, height=600, scrolling=True)
            
            # ========== SAVE HISTORY (Optional) =====
            try:
                from services.blocks.rag_orchestrator import store_history
                store_history(
                    "Dịch Thuật",
                    f"{source_lang} → {target_lang} ({style})",
                    text_input[:500]
                )
            except Exception:
                # History saving is optional
                pass
        
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            
            st.error(f"❌ Lỗi dịch thuật: {str(e)}")
            
            # Show detailed error in expander
            with st.expander("🔍 Chi tiết lỗi (cho developer)"):
                st.exception(e)
            
            st.info("💡 **Gợi ý khắc phục:**\n"
                   "- Kiểm tra kết nối mạng\n"
                   "- Thử giảm độ dài văn bản\n"
                   "- Đợi 1 phút rồi thử lại (có thể API quá tải)")


# ===== HELPER FUNCTIONS =====

def _estimate_api_calls(text: str, include_english: bool, target_lang: str) -> dict:
    """
    [Inference] Ước tính API calls khi không có text_processor
    
    Fallback estimation khi block chưa có
    """
    char_count = len(text.replace(" ", ""))
    
    # Giả sử mỗi chunk 1500 chars
    num_chunks = max(1, char_count // 1500)
    
    api_calls = num_chunks
    if include_english and target_lang != "English":
        api_calls *= 2
    
    return {
        "total_chars": char_count,
        "num_chunks": num_chunks,
        "estimated_api_calls": api_calls,
        "warning": "⚠️ Văn bản dài, có thể tốn thời gian" if api_calls > 20 else None
    }
