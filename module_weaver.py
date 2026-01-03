import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import time
import json

# Local imports
from services.blocks.file_processor import doc_file
from services.blocks.embedding_engine import load_encoder
from services.blocks.html_generator import load_template, create_html_block, create_interactive_html_block

# ✅ SỬA: Import đầy đủ từ rag_orchestrator
from services.blocks.rag_orchestrator import (
    analyze_document_streamlit, 
    compute_similarity_with_excel, 
    store_history, 
    init_knowledge_universe, 
    create_personal_rag, 
    tai_lich_su,
    get_translation_orchestrator  # ✅ Hàm này có ở dòng 356 trong rag_orchestrator.py
)

# KG module cho upgrade
from services.blocks import knowledge_graph_v2 as kg_module

# Core engines
from ai_core import AI_Core
from voice_block import Voice_Engine
from prompts import DEBATE_PERSONAS, BOOK_ANALYSIS_PROMPT
from services.blocks import knowledge_graph_v2 as kg_module

# Optional supabase import (don't fail app if missing)
try:
    from supabase import create_client, Client
except ImportError:
    pass

# TRANSLATIONS / UI TEXT
TRANS = {
    "vi": {
        "lang_select": "Ngôn ngữ / Language / 语言",
        "tab1": "📚 Phân Tích Sách",
        "tab2": "✍️ Dịch Giả",
        "tab3": "🗣️ Tranh Biện",
        "tab4": "🎙️ Phòng Thu AI",
        "tab5": "⏳ Nhật Ký",
        "t1_header": "Trợ lý Nghiên cứu & Knowledge Graph",
        "t1_up_excel": "1. Kết nối Kho Sách (Excel)",
        "t1_up_doc": "2. Tài liệu mới (PDF/Docx)",
        "t1_btn": "🚀 PHÂN TÍCH NGAY",
        "t1_analyzing": "Đang phân tích {name}...",
        "t1_connect_ok": "✅ Đã kết nối {n} cuốn sách.",
        "t1_graph_title": "🪐 Vũ trụ Sách",
        "t1_seed_books": "✅ Đã tải {n} sách tinh hoa vào Knowledge Graph (18 sách bao trùm 4 tầng triết học)",
        "t2_header": "Dịch Thuật Đa Chiều",
        "t2_input": "Nhập văn bản cần dịch:",
        "t2_target": "Dịch sang:",
        "t2_style": "Phong cách:",
        "t2_btn": "✍️ Dịch Ngay",
        "t3_header": "Đấu Trường Tư Duy",
        "t3_persona_label": "Chọn Đối Thủ:",
        "t3_input": "Nhập chủ đề tranh luận...",
        "t3_clear": "🗑️ Xóa Chat",
        "t4_header": "🎙️ Phòng Thu AI Đa Ngôn Ngữ",
        "t4_voice": "Chọn Giọng:",
        "t4_speed": "Tốc độ:",
        "t4_btn": "🔊 TẠO AUDIO",
        "t5_header": "Nhật Ký & Lịch Sử",
        "t5_refresh": "🔄 Tải lại Lịch sử",
        "t5_empty": "Chưa có dữ liệu lịch sử.",
    },
    "en": {
        "lang_select": "Language",
        "tab1": "📚 Book Analysis",
        "tab2": "✍️ Translator",
        "tab3": "🗣️ Debater",
        "tab4": "🎙️ AI Studio",
        "tab5": "⏳ History",
        "t1_header": "Research Assistant & Knowledge Graph",
        "t1_up_excel": "1. Connect Book Database (Excel)",
        "t1_up_doc": "2. New Documents (PDF/Docx)",
        "t1_btn": "🚀 ANALYZE NOW",
        "t1_analyzing": "Analyzing {name}...",
        "t1_connect_ok": "✅ Connected {n} books.",
        "t1_graph_title": "🪐 Book Universe",
        "t1_seed_books": "✅ Loaded {n} foundational books into Knowledge Graph (18 books covering 4 philosophy layers)",
        "t2_header": "Multidimensional Translator",
        "t2_input": "Enter text to translate:",
        "t2_target": "Translate to:",
        "t2_style": "Style:",
        "t2_btn": "✍️ Translate",
        "t3_header": "Thinking Arena",
        "t3_persona_label": "Choose Opponent:",
        "t3_input": "Enter debate topic...",
        "t3_clear": "🗑️ Clear Chat",
        "t4_header": "🎙️ Multilingual AI Studio",
        "t4_voice": "Select Voice:",
        "t4_speed": "Speed:",
        "t4_btn": "🔊 GENERATE AUDIO",
        "t5_header": "Logs & History",
        "t5_refresh": "🔄 Refresh History",
        "t5_empty": "No history data found.",
    },
    "zh": {
        "lang_select": "语言",
        "tab1": "📚 书籍分析",
        "tab2": "✍️ 翻译专家",
        "tab3": "🗣️ 辩论场",
        "tab4": "🎙️ AI 录音室",
        "tab5": "⏳ 历史记录",
        "t1_header": "研究助手 & 知识图谱",
        "t1_up_excel": "1. 连接书库 (Excel)",
        "t1_up_doc": "2. 上传新文档 (PDF/Docx)",
        "t1_btn": "🚀 立即分析",
        "t1_analyzing": "正在分析 {name}...",
        "t1_connect_ok": "✅ 已连接 {n} 本书。",
        "t1_graph_title": "🪐 书籍宇宙",
        "t1_seed_books": "✅ 已加载 {n} 本精华书籍到知识图谱 (18本书覆盖4层哲学)",
        "t2_header": "多维翻译",
        "t2_input": "输入文本:",
        "t2_target": "翻译成:",
        "t2_style": "风格:",
        "t2_btn": "✍️ 翻译",
        "t3_header": "思维竞技场",
        "t3_persona_label": "选择对手:",
        "t3_input": "输入辩论主题...",
        "t3_clear": "🗑️ 清除聊天",
        "t4_header": "🎙️ AI 多语言录音室",
        "t4_voice": "选择声音:",
        "t4_speed": "语速:",
        "t4_btn": "🔊 生成音频",
        "t5_header": "日志 & 历史",
        "t5_refresh": "🔄 刷新历史",
        "t5_empty": "暂无历史数据。",
    }
}

def T(key):
    lang = st.session_state.get('weaver_lang', 'vi')
    return TRANS.get(lang, TRANS['vi']).get(key, key)

@st.cache_resource
def load_models():
    try:
        model = load_encoder()
        return model
    except Exception:
        return None

def check_model_available():
    model = load_models()
    if model is None:
        st.warning("⚠️ Chức năng Knowledge Graph tạm thời không khả dụng (thiếu RAM)")
        return False
    return True

def doc_file_safe(uploaded_file):
    return doc_file(uploaded_file)

# ✅ SỬA: Helper để init KnowledgeUniverse với sách tinh hoa + Excel upgrade
@st.cache_resource
def get_knowledge_universe(excel_file=None):
    """Khởi tạo Knowledge Graph với sách tinh hoa (18 sách) + optional Excel upgrade"""
    try:
        # BƯỚC 1: Tạo KG cơ bản (đã có 18 sách tinh hoa từ knowledge_graph_v2.py)
        ku = init_knowledge_universe()
        if not ku:
            st.warning("⚠️ Không thể khởi tạo Knowledge Graph")
            return None
        
        # BƯỚC 2: Nếu có Excel, upgrade thêm sách từ Excel
        if excel_file:
            try:
                # Đọc Excel để lấy danh sách sách
                df_excel = pd.read_excel(excel_file).dropna(subset=["Tên sách"])
                st.success(f"✅ Đã kết nối {len(df_excel)} cuốn sách từ Excel")
                
                # Upgrade KG với sách từ Excel
                ku = kg_module.upgrade_existing_database(excel_file, ku)
                
                # Hiển thị thông báo thành công
                total_books = len(ku.graph.nodes)
                st.success(f"✅ Đã tải {total_books} sách vào Knowledge Graph (bao gồm 18 sách tinh hoa + {len(df_excel)} từ Excel)")
                
            except Exception as e:
                st.warning(f"⚠️ Không thể upgrade từ Excel: {e}")
        else:
            # Chỉ có sách tinh hoa
            total_books = len(ku.graph.nodes)
            st.info(f"📚 Đã tải {total_books} sách tinh hoa vào Knowledge Graph (18 sách bao trùm 4 tầng triết học)")
        
        return ku
        
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo Knowledge Graph: {e}")
        return None

# --- RUN ---
def run():
    ai = AI_Core()
    voice = Voice_Engine()

    # ✅ THAY ĐỔI: Khởi tạo KG với thông báo rõ ràng về sách tinh hoa
    knowledge_universe = get_knowledge_universe()

    with st.sidebar:
        st.markdown("---")
        lang_choice = st.selectbox("🌐 " + TRANS['vi']['lang_select'], ["Tiếng Việt", "English", "中文"], key="weaver_lang_selector")
        if lang_choice == "Tiếng Việt":
            st.session_state.weaver_lang = 'vi'
        elif lang_choice == "English":
            st.session_state.weaver_lang = 'en'
        else:
            st.session_state.weaver_lang = 'zh'

    st.header(f"🧠 The Cognitive Weaver")
    
    # ✅ HIỂN THỊ TRẠNG THÁI KG (MỚI)
    if knowledge_universe:
        summary = knowledge_universe.get_episteme_summary()
        col1, col2, col3, col4 = st.columns(4)
        for layer, data in summary.items():
            with eval(f"col{1+list(summary.keys()).index(layer)}"):
                st.metric(layer[:15], f"{data['count']} sách", delta=f"{len(data['recent'])} recent")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([T("tab1"), T("tab2"), T("tab3"), T("tab4"), T("tab5")])

    # TAB 1: RAG (CẢI TIẾN với KG integration)
    with tab1:
        st.header(T("t1_header"))
        with st.container():
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                file_excel = st.file_uploader(T("t1_up_excel"), type="xlsx", key="t1_excel")
            with c2:
                uploaded_files = st.file_uploader(T("t1_up_doc"), type=["pdf", "docx", "txt", "md", "html"], accept_multiple_files=True, key="t1_files")
            with c3:
                st.write("")
                st.write("")
                btn_run = st.button(T("t1_btn"), type="primary", use_container_width=True)

        # ✅ RELOAD KG nếu có Excel mới
        if file_excel and btn_run:
            knowledge_universe = get_knowledge_universe(file_excel)

        if btn_run and uploaded_files:
            vec = load_encoder()
            has_db_excel = bool(file_excel)

            for f in uploaded_files:
                text = doc_file_safe(f)
                if not text:
                    st.warning(f"⚠️ Không đọc được file {f.name}")
                    continue

                link = ""
                if has_db_excel and vec is not None:
                    try:
                        matches = compute_similarity_with_excel(text, pd.read_excel(file_excel).dropna(subset=["Tên sách"]), vec)
                        if matches:
                            link = "\n".join([f"- {m[0]} ({m[1]*100:.0f}%)" for m in matches])
                    except Exception as e:
                        st.warning(f"Không thể tính similarity: {e}")

                # ✅ TÌM SÁCH LIÊN QUAN TỪ KG (ưu tiên)
                related = []
                if knowledge_universe:
                    try:
                        related = knowledge_universe.find_related_books(text[:2000], top_k=5)
                    except Exception as e:
                        st.warning(f"Lỗi KG search: {e}")

                with st.spinner(T("t1_analyzing").format(name=f.name)):
                    res = analyze_document_streamlit(f.name, text, user_lang=st.session_state.get('weaver_lang', 'vi'))
                    if res and "Lỗi" not in res:
                        st.markdown(f"### 📄 {f.name}")
                        if link:
                            st.markdown("**🔗 Sách tương tự từ Excel:**")
                            st.markdown(link)
                        if related:
                            st.markdown("**🪐 Sách liên quan từ Knowledge Graph (18 sách tinh hoa):**")
                            for node_id, title, score, explanation in related:
                                fp = knowledge_universe.graph.nodes[node_id].get("first_principles", "")
                                st.markdown(f"- **{title}** ({score:.2f}) — {explanation}" + (f"\n  *First Principles:* {fp}" if fp else ""))
                        st.markdown(res)
                        st.markdown("---")
                        store_history("Phân Tích Sách", f.name, res)
                    else:
                        st.error(f"❌ Không thể phân tích file {f.name}: {res}")

        # Graph visualization (cải tiến với KG export)
        if knowledge_universe:
            with st.expander(T("t1_graph_title"), expanded=False):
                try:
                    nodes, edges = knowledge_universe.export_for_visualization()
                    if nodes:
                        from streamlit_agraph import agraph, Node, Edge, Config
                        config = Config(width=1000, height=600, directed=True, physics=True)
                        agraph(nodes[:50], edges[:100], config)  # Limit để tránh lag
                except Exception as e:
                    st.info("📊 Graph visualization tạm thời không khả dụng.")
        elif file_excel:
            # Fallback Excel graph (giữ nguyên logic cũ)
            try:
                if "df_viz" not in st.session_state:
                    st.session_state.df_viz = pd.read_excel(file_excel).dropna(subset=["Tên sách"])
                # ... (code graph Excel cũ giữ nguyên)
            except:
                pass

    
    # TAB 2: Dịch Thuật Đa Chiều (NÂNG CẤP - dùng TranslationOrchestrator)
    with tab2:
        st.subheader(T("t2_header"))
        st.markdown("**Dịch văn bản đa ngôn ngữ với phong cách chuyên nghiệp, hỗ trợ interactive (tiếng Trung) và tải file HTML.**")

        # Input text
        input_text = st.text_area("Nhập văn bản cần dịch:", height=200, key="translator_input")

        col1, col2, col3 = st.columns(3)
        with col1:
            source_lang = st.selectbox(
                "Ngôn ngữ nguồn:",
                ["Chinese", "English", "Vietnamese", "French", "Japanese", "Korean"],
                index=0
            )
        with col2:
            target_lang = st.selectbox(
                "Ngôn ngữ đích:",
                ["Vietnamese", "English", "Chinese", "French", "Japanese", "Korean"],
                index=0
            )
        with col3:
            mode = st.radio("Chế độ dịch:", ["Standard (HTML đẹp)", "Interactive (chỉ tiếng Trung → Việt)"], horizontal=True)

        include_english = st.checkbox("Thêm bản dịch tiếng Anh làm tham chiếu (nếu cần)", value=True)

        if st.button("🚀 Dịch ngay", type="primary", use_container_width=True):
            if not input_text.strip():
                st.warning("Vui lòng nhập văn bản cần dịch.")
            else:
                # Lấy orchestrator
                orchestrator = get_translation_orchestrator()
                if not orchestrator:
                    st.error("⚠️ Không tải được bộ dịch. Kiểm tra translator.py và API key.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(value):
                        progress_bar.progress(value)
                        status_text.text(f"Đang dịch... {int(value*100)}%")

                    try:
                        if mode == "Interactive (chỉ tiếng Trung → Việt)":
                            if source_lang != "Chinese":
                                st.error("Interactive mode chỉ hỗ trợ nguồn tiếng Trung.")
                            else:
                                status_text.text("Đang xử lý interactive translation...")
                                html_output = orchestrator.translate_interactive(
                                    input_text,
                                    source_lang="Chinese",
                                    target_lang=target_lang
                                )
                        else:
                            status_text.text("Đang dịch và tạo HTML...")
                            html_output = orchestrator.translate_document(
                                input_text,
                                source_lang=source_lang,
                                target_lang=target_lang,
                                include_english=include_english and target_lang != "English",
                                progress_callback=update_progress
                            )

                        # Thành công
                        progress_bar.progress(1.0)
                        status_text.success("✅ Hoàn thành!")

                        # Nút tải HTML
                        st.download_button(
                            label="📥 Tải file HTML kết quả",
                            data=html_output.encode('utf-8'),
                            file_name=f"translation_{source_lang}_to_{target_lang}.html",
                            mime="text/html"
                        )

                        # Preview
                        with st.expander("👀 Xem trước kết quả", expanded=True):
                            st.components.v1.html(html_output, height=800, scrolling=True)

                        # Lưu lịch sử
                        store_history(
                            "Dịch Thuật",
                            f"{source_lang} → {target_lang} ({mode})",
                            input_text[:300]
                        )

                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Lỗi trong quá trình dịch: {str(e)}")
                        with st.expander("Chi tiết lỗi"):
                            st.exception(e)

    # TAB 3: Đấu trường
    with tab3:
        st.subheader(T("t3_header"))
        mode = st.radio("Mode:", ["👤 Solo", "⚔️ Multi-Agent"], horizontal=True, key="w_t3_mode")
        if "weaver_chat" not in st.session_state:
            st.session_state.weaver_chat = []

        if mode == "👤 Solo":
            c1, c2 = st.columns([3, 1])
            with c1:
                persona = st.selectbox(T("t3_persona_label"), list(DEBATE_PERSONAS.keys()), key="w_t3_solo_p")
            with c2:
                if st.button(T("t3_clear"), key="w_t3_clr"):
                    st.session_state.weaver_chat = []
                    st.rerun()
            for msg in st.session_state.weaver_chat:
                st.chat_message(msg["role"]).write(msg["content"])
            if prompt := st.chat_input(T("t3_input")):
                st.chat_message("user").write(prompt)
                st.session_state.weaver_chat.append({"role": "user", "content": prompt})
                recent_history = st.session_state.weaver_chat[-10:]
                context_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent_history])
                full_prompt = f"LỊCH SỬ:\n{context_text}\n\nNHIỆM VỤ: Trả lời câu hỏi mới nhất của USER."
                with st.chat_message("assistant"):
                    with st.spinner("..."):
                        res = ai.generate(full_prompt, model_type="flash", system_instruction=DEBATE_PERSONAS[persona])
                        if res:
                            st.write(res)
                            st.session_state.weaver_chat.append({"role": "assistant", "content": res})
                            store_history("Tranh Biện Solo", f"{persona} - {prompt[:50]}...", f"Q: {prompt}\nA: {res}")
        else:
            participants = st.multiselect("Chọn Hội Đồng:", list(DEBATE_PERSONAS.keys()),
                                          default=[list(DEBATE_PERSONAS.keys())[0], list(DEBATE_PERSONAS.keys())[1]],
                                          max_selections=3)
            topic = st.text_input("Chủ đề:", key="w_t3_topic")
            if st.button("🔥 KHAI CHIẾN", disabled=(len(participants) < 2 or not topic)):
                st.session_state.weaver_chat = []
                start_msg = f"📢 **CHỦ TỌA:** Khai mạc tranh luận về: *'{topic}'*"
                st.session_state.weaver_chat.append({"role": "system", "content": start_msg})
                st.info(start_msg)
                full_transcript = [start_msg]

                MAX_DEBATE_TIME = 600
                start_time = time.time()

                with st.status("🔥 Cuộc chiến đang diễn ra (3 vòng)...") as status:
                    try:
                        for round_num in range(1, 4):
                            if time.time() - start_time > MAX_DEBATE_TIME:
                                st.warning("⏰ Hết giờ! Cuộc tranh luận kết thúc sớm.")
                                break

                            status.update(label=f"🔄 Vòng {round_num}/3 đang diễn ra...")

                            for i, p_name in enumerate(participants):
                                if time.time() - start_time > MAX_DEBATE_TIME:
                                    break

                                context_str = topic
                                if len(st.session_state.weaver_chat) > 1:
                                    recent_msgs = st.session_state.weaver_chat[-4:]
                                    context_str = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])

                                length_instruction = " (BẮT BUỘC: Trả lời ngắn gọn khoảng 150-200 từ. Đi thẳng vào trọng tâm, không lan man.)"

                                if round_num == 1:
                                    p_prompt = f"CHỦ ĐỀ: {topic}\nNHIỆM VỤ (Vòng 1 - Mở đầu): Nêu quan điểm chính và 2-3 lý lẽ. {length_instruction}"
                                else:
                                    p_prompt = f"CHỦ ĐỀ: {topic}\nBỐI CẢNH MỚI NHẤT:\n{context_str}\n\nNHIỆM VỤ (Vòng {round_num} - Phản biện): Phản biện sắc bén quan điểm đối thủ và củng cố lập trường của mình. {length_instruction}"

                                try:
                                    res = ai.generate(
                                        p_prompt,
                                        model_type="pro",
                                        system_instruction=DEBATE_PERSONAS[p_name]
                                    )

                                    if res:
                                        clean_res = res.replace(f"{p_name}:", "").strip()
                                        clean_res = clean_res.replace(f"**{p_name}:**", "").strip()
                                        icons = {"Kẻ Phản Biện": "😈", "Shushu": "🎩", "Phật Tổ": "🙏", "Socrates": "🏛️"}
                                        icon = icons.get(p_name, "🤖")
                                        content_fmt = f"### {icon} {p_name}\n\n{clean_res}"
                                        st.session_state.weaver_chat.append({"role": "assistant", "content": content_fmt})
                                        full_transcript.append(content_fmt)
                                        with st.chat_message("assistant", avatar=icon):
                                            st.markdown(content_fmt)
                                        time.sleep(3)
                                except Exception as e:
                                    st.error(f"Lỗi khi gọi AI cho {p_name}: {e}")
                                    continue
                        status.update(label="✅ Tranh luận kết thúc!", state="complete")
                    except Exception as e:
                        st.error(f"Lỗi trong quá trình tranh luận: {e}")

                full_log = "\n\n".join(full_transcript)
                store_history("Hội Đồng Tranh Biện", f"Chủ đề: {topic}", full_log)

    # TAB 4: VOICE
    with tab4:
        st.subheader(T("t4_header"))
        
        # 1. Chọn Giọng (Lấy từ Voice Engine)
        if voice and hasattr(voice, 'VOICE_OPTIONS'):
            voice_opts = list(voice.VOICE_OPTIONS.keys())
            selected_voice = st.selectbox(T("t4_voice"), voice_opts, index=0)
        else:
            selected_voice = None
            st.warning("⚠️ Chưa tải được module giọng nói.")

        # 2. Chọn Tốc độ
        speed = st.slider(T("t4_speed"), -50, 50, 0, format="%d%%")

        # 3. Nhập văn bản
        inp_v = st.text_area("Text:", height=150)
        
        if st.button(T("t4_btn")) and inp_v:
            with st.spinner("Đang tạo âm thanh..."):
                # Truyền giọng và tốc độ vào hàm speak
                path = voice.speak(inp_v, voice_key=selected_voice, speed=speed)
                if path: 
                    st.audio(path)

    # TAB 5: NHẬT KÝ (CÓ PHẦN BAYES)
    with tab5:
        st.subheader("⏳ Nhật Ký & Phản Chiếu Tư Duy")
        if st.button("🔄 Tải lại", key="w_t5_refresh"):
            st.session_state.history_cloud = tai_lich_su()
            st.rerun()

        data = st.session_state.get("history_cloud", tai_lich_su())

        if data:
            df_h = pd.DataFrame(data)

            if "SentimentScore" in df_h.columns:
                try:
                    df_h["score"] = pd.to_numeric(df_h["SentimentScore"], errors='coerce').fillna(0)
                    import plotly.express as px
                    fig = px.line(df_h, x="Time", y="score", markers=True, color_discrete_sequence=["#76FF03"])
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass

            with st.expander("🔮 Phân tích Tư duy theo xác suất Bayes (E.T. Jaynes)", expanded=False):
                st.info("AI sẽ coi Lịch sử hoạt động của chị là 'Dữ liệu quan sát' (Evidence) để suy luận ra 'Hàm mục tiêu' (Objective Function) và sự dịch chuyển niềm tin của chị.")

                if st.button("🧠 Chạy Mô hình Bayes ngay"):
                    with st.spinner("Đang tính toán xác suất hậu nghiệm (Posterior)..."):
                        recent_logs = df_h.tail(10).to_dict(orient="records")
                        logs_text = json.dumps(recent_logs, ensure_ascii=False)

                        bayes_prompt = f"""
                        Đóng vai một nhà khoa học tư duy theo trường phái E.T. Jaynes (sách 'Probability Theory: The Logic of Science').

                        DỮ LIỆU QUAN SÁT (EVIDENCE):
                        Đây là nhật ký hoạt động của tôi:
                        {logs_text}

                        NHIỆM VỤ:
                        Hãy phân tích chuỗi hành động này như một bài toán suy luận Bayes.
                        1. **Xác định Priors (Niềm tin tiên nghiệm):** Dựa trên các hành động đầu, tôi đang quan tâm/tin tưởng điều gì?
                        2. **Cập nhật Likelihood (Khả năng):** Các hành động tiếp theo củng cố hay làm yếu đi niềm tin đó?
                        3. **Kết luận Posterior (Hậu nghiệm):** Trạng thái tư duy hiện tại của tôi đang hội tụ về đâu? Có mâu thuẫn (Inconsistency) nào trong logic hành động không?

                        Trả lời ngắn gọn, sâu sắc, dùng thuật ngữ xác suất nhưng dễ hiểu.
                        """

                        analysis = ai.generate(bayes_prompt, model_type="pro")
                        st.markdown(analysis)

            st.divider()
            for index, item in df_h.iterrows():
                t = str(item.get('Time', ''))
                tp = str(item.get('Type', ''))
                ti = str(item.get('Title', ''))
                ct = str(item.get('Content', ''))

                icon = "📝"
                if "Tranh Biện" in tp:
                    icon = "🗣️"
                elif "Dịch" in tp:
                    icon = "✍️"

                with st.expander(f"{icon} {t} | {tp} | {ti}"):
                    st.markdown(ct)
        else:
            st.info(T("t5_empty"))
