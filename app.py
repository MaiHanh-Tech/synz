import streamlit as st
import google.generativeai as genai
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import numpy as np
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="Mai Hanh Strategy (Pro)", layout="wide", page_icon="💎")
st.title("💎 The Mai Hanh Analyzer (Unlimited Context)")

# --- QUẢN LÝ BẢO MẬT (SECRETS) ---
if 'GOOGLE_API_KEY' in st.secrets:
    api_key = st.secrets['GOOGLE_API_KEY']
else:
    api_key = st.sidebar.text_input("Nhập Google API Key:", type="password")

if not api_key:
    st.warning("⚠️ Vui lòng nhập API Key để tiếp tục.")
    st.stop()

genai.configure(api_key=api_key)

# *** CẤU HÌNH MODEL MẠNH NHẤT (LONG CONTEXT) ***
# Sử dụng 1.5 Pro vì đây là bản hỗ trợ 2 TRIỆU tokens (đọc nguyên cuốn sách)
# Google chưa có API 2.5 Pro, 1.5 Pro hiện là bản SOTA (State-of-the-art)
try:
    model = genai.GenerativeModel('gemini-2.5-pro')
except:
    st.error("Tài khoản chưa hỗ trợ Pro, chuyển về Flash.")
    model = genai.GenerativeModel('gemini-2.5-flash')

# --- HÀM XỬ LÝ ---
@st.cache_resource
def load_models():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def doc_noi_dung_file(uploaded_file):
    if not uploaded_file: return ""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        # Đọc PDF (Toàn bộ các trang)
        if ext == '.pdf':
            reader = PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in reader.pages])
        # Đọc Word
        elif ext == '.docx':
            doc = Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        # Đọc Text/Markdown
        elif ext in ['.txt', '.md']:
            return str(uploaded_file.read(), "utf-8")
        # Đọc Web/HTML
        elif ext in ['.html', '.htm']:
            soup = BeautifulSoup(uploaded_file, 'html.parser')
            text = soup.get_text()
            return text
    except Exception as e: return f"Lỗi đọc file: {e}"
    return ""

# --- GIAO DIỆN ---
with st.sidebar:
    st.header("1. Kết Nối Kho Sách")
    file_excel = st.file_uploader("Upload Book146.xlsx", type="xlsx")
    
    vec_model = None
    db_vec = None
    df = None
    
    if file_excel:
        try:
            df = pd.read_excel(file_excel).dropna(subset=['Tên sách'])
            vec_model = load_models()
            # Tạo dữ liệu vector cho kho sách
            content = [f"{r['Tên sách']} {r['CẢM NHẬN']}" for i,r in df.iterrows()]
            db_vec = vec_model.encode(content)
            st.success(f"✅ Đã nạp {len(df)} cuốn sách cũ.")
        except: st.error("Lỗi file Excel")

st.header("2. Upload Tài Liệu (Hỗ trợ sách dài)")
uploaded_files = st.file_uploader(
    "Kéo thả các file cần phân tích vào đây", 
    type=["pdf","docx","txt","md","html"], 
    accept_multiple_files=True 
)

if st.button("🚀 PHÂN TÍCH & TỔNG HỢP CHIẾN LƯỢC", type="primary"):
    if not uploaded_files:
        st.warning("Chưa có file nào!")
    else:
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        danh_sach_tom_tat = [] 

        st.subheader("📝 I. Phân Tích Chi Tiết")
        
        for i, file_doc in enumerate(uploaded_files):
            with st.spinner(f"Gemini Pro đang đọc toàn bộ file {i+1}/{total_files}: {file_doc.name}..."):
                # 1. Đọc nội dung (FULL - Không cắt)
                text = doc_noi_dung_file(file_doc)
                do_dai = len(text)
                
                # 2. RAG (Tìm liên kết với kho sách cũ)
                lien_ket = ""
                if file_excel and len(text) > 100:
                    try:
                        # Chỉ lấy 2000 ký tự đầu để tìm kiếm vector cho nhanh
                        query_vec = vec_model.encode([text[:2000]])
                        scores = cosine_similarity(query_vec, db_vec)[0]
                        top = np.argsort(scores)[::-1][:3]
                        for idx in top:
                            if scores[idx] > 0.35: # Ngưỡng giống nhau > 35%
                                lien_ket += f"- {df.iloc[idx]['Tên sách']} (Tác giả: {df.iloc[idx]['Tác giả']})\n"
                    except: pass

                if not lien_ket: lien_ket = "Không tìm thấy liên kết rõ ràng với kho sách cũ."

                # 3. Prompt (Gửi toàn bộ nội dung sách)
                # Dùng model Pro nên ta tự tin gửi cả text dài
                prompt = f'''
                Bạn là Trợ lý Nghiên cứu Chiến lược (Sử dụng Model Gemini Pro - Long Context).
                
                NHIỆM VỤ: Phân tích tài liệu: '{file_doc.name}' (Độ dài: {do_dai} ký tự).
                
                THÔNG TIN THAM KHẢO TỪ KHO SÁCH CŨ CỦA CHỊ HẠNH:
                {lien_ket}
                
                YÊU CẦU: 
                1. **Tóm tắt cốt lõi:** Những luận điểm chính yếu nhất của sách/tài liệu này.
                2. **Phân tích chiều sâu:** Đánh giá tư duy tác giả, điểm mạnh/yếu của lập luận.
                3. **Kết nối tri thức:** Tài liệu này bổ sung hay phản biện gì với các cuốn sách cũ trong danh sách tham khảo ở trên?
                4. **Trích dẫn đắt giá:** 1 câu trích dẫn hay nhất.
                5. **Phương pháp luâj:** Tác giả đã dùng phương pháp gì để đi đến kết luận này? Giả định ngầm của họ là gì? Nếu CÁC tác giả khác  phân tích cùng một vấn đề, HỌ sẽ nói gì? 
                
                NỘI DUNG TÀI LIỆU (FULL TEXT):
                {text}
                '''
                
                try:
                    res = model.generate_content(prompt)
                    danh_sach_tom_tat.append(f"=== TÀI LIỆU {i+1}: {file_doc.name} ===\n{res.text}\n")
                    
                    with st.expander(f"📄 Kết quả: {file_doc.name} (Đã đọc {do_dai} ký tự)", expanded=False):
                        st.markdown(res.text)
                except Exception as e:
                    st.error(f"Lỗi AI khi đọc file này: {e}")
            
            progress_bar.progress((i + 1) / total_files)

        st.divider()
        st.header("🏆 II. BÁO CÁO TỔNG QUAN CHIẾN LƯỢC")
        
        if len(danh_sach_tom_tat) > 0:
            with st.spinner("🧠 Brain Pro đang tổng hợp chiến lược..."):
                du_lieu_tong_hop = "\n".join(danh_sach_tom_tat)
                
                prompt_tong_hop = f'''
                Bạn là Cố vấn Chiến lược cấp cao.
                Dưới đây là các bản phân tích của {total_files} tài liệu.
                
                DỮ LIỆU ĐẦU VÀO:
                {du_lieu_tong_hop}
                
                NHIỆM VỤ: Viết BÁO CÁO TỔNG HỢP (SYNTHESIS).
                1. **Mẫu hình chung (Patterns):** Các tài liệu này có điểm gì tương đồng về tư duy?
                2. **Góc nhìn đa chiều:** Các tài liệu bổ sung hay mâu thuẫn nhau?
                3. **Kết luận chiến lược:** Bài học cốt lõi rút ra là gì?
                
                Hãy viết sâu sắc, logic.
                '''
                
                try:
                    res_tong_hop = model.generate_content(prompt_tong_hop)
                    st.success("Đã hoàn thành tổng hợp!")
                    st.markdown(res_tong_hop.text)
                    st.download_button("💾 Tải Báo Cáo Tổng Hợp (.txt)", res_tong_hop.text, file_name="Bao_Cao_Tong_Hop.txt")
                except: st.error("Lỗi tổng hợp.")
