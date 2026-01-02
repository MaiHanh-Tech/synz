import google.generativeai as genai
import streamlit as st
import time
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError, InvalidArgument

class AI_Core:
    def __init__(self):
        self.api_ready = False
        try:
            # Kiểm tra key tồn tại trước khi lấy
            if "api_keys" in st.secrets and "gemini_api_key" in st.secrets["api_keys"]:
                api_key = st.secrets["api_keys"]["gemini_api_key"]
                genai.configure(api_key=api_key)
                self.api_ready = True
            else:
                st.error("⚠️ Chưa cấu hình API Key trong secrets.toml")
                return

            # Cấu hình Safety (Chặn nội dung độc hại)
            self.safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            # Cấu hình Generation Config (Tối ưu cho 2.5 Pro)
            self.gen_config = genai.GenerationConfig(
                temperature=0.8,
                max_output_tokens= 8000,  
                top_p=0.95,
                top_k=40
            )

        except Exception as e:
            st.error(f"❌ Lỗi khởi tạo AI Core: {e}")

    def _get_model(self, model_name, system_instr=None):
        """Hàm helper để khởi tạo model đúng phiên bản"""
        # ✅ DANH SÁCH MODEL MỚI NHẤT (Cập nhật 2025)
        valid_names = {
            "flash": "gemini-2.5-flash",         # Nhanh, rẻ
            "pro": "gemini-2.5-pro",             # Thông minh nhất (Dùng cho tranh biện)
            "exp": "gemini-2.5-flash-lite"        # Bản thử nghiệm
        }
        
        # Mặc định fallback về 2.5 Flash nếu tên sai
        target_name = valid_names.get(model_name, "gemini-2.5-flash")
        
        try:
            return genai.GenerativeModel(
                model_name=target_name,
                safety_settings=self.safety_settings,
                generation_config=self.gen_config,
                system_instruction=system_instr
            )
        except Exception as e:
            # st.warning(f"⚠️ Không thể khởi tạo model {target_name}: {e}")
            return None

    def generate(self, prompt, model_type="flash", system_instruction=None):
        """
        Hàm gọi AI chính: Tự động chuyển model nếu lỗi (Fallback Strategy)
        """
        if not self.api_ready:
            return "⚠️ API Key chưa sẵn sàng."

        # ✅ CHIẾN THUẬT ƯU TIÊN: Pro -> Flash -> Exp
        if model_type == "pro":
            # Với task khó (Tranh biện): Ưu tiên 2.5 Pro
            plan = [
                ("pro", "Gemini 2.5 pro", 6), 
                ("flash", "Gemini 2.5 Flash", 3), 
                ("exp", "gemini-2.5-flash-lite", 3)
            ]
        else:
            # Với task thường: Ưu tiên Flash cho nhanh
            plan = [
                ("flash", "Gemini 2.5 Flash", 2), 
                ("exp", "gemini-2.5-flash-lite", 2),
                ("pro", "Gemini 2.5 Pro", 6)
            ]

        last_errors = []
        quota_exhausted_count = 0

        for m_type, m_name, base_wait_time in plan:
            try:
                # Khởi tạo model
                model = self._get_model(m_type, system_instr=system_instruction)
                if not model: continue
                
                # Gọi API
                response = model.generate_content(prompt)
                
                # Kiểm tra kết quả
                if response and hasattr(response, 'text') and response.text:
                    return response.text
                
                # Xử lý các lý do bị chặn (Safety, Token...)
                if response and hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        reason = candidate.finish_reason.name
                        if reason == "SAFETY":
                            last_errors.append(f"{m_name}: Bị chặn (Safety)")
                            continue
                        elif reason == "MAX_TOKENS":
                            last_errors.append(f"{m_name}: Quá dài (Max Tokens)")
                            continue
                
                last_errors.append(f"{m_name}: Trả về rỗng")
                continue
            
            except ResourceExhausted:
                # Lỗi hết tiền/quota -> Chờ lâu hơn một chút rồi thử model khác
                quota_exhausted_count += 1
                error_msg = f"{m_name}: Hết Quota (429)"
                last_errors.append(error_msg)
                time.sleep(base_wait_time * quota_exhausted_count)
                
            except (ServiceUnavailable, InternalServerError):
                # Lỗi Server Google -> Chờ ngắn
                last_errors.append(f"{m_name}: Lỗi Server (5xx)")
                time.sleep(2)
            
            except InvalidArgument as e:
                # Lỗi Input -> Dừng luôn, không thử lại
                return f"⚠️ Lỗi Input (Prompt không hợp lệ): {str(e)[:200]}"
                
            except Exception as e:
                last_errors.append(f"{m_name}: Lỗi lạ ({str(e)[:50]})")
                time.sleep(1)

        # Nếu thử hết các model mà vẫn lỗi
        error_summary = "\n".join(f"- {e}" for e in last_errors[-3:])
        return f"⚠️ Hệ thống đang bận hoặc gặp lỗi:\n{error_summary}\n\n💡 Vui lòng thử lại sau 1 phút."

    @staticmethod
    @st.cache_data(show_spinner=False, ttl=3600)
    def analyze_static(text, instruction):
        """
        Hàm dùng riêng cho RAG (Đọc tài liệu) - Có Cache để tiết kiệm tiền
        """
        try:
            api_key = st.secrets["api_keys"]["gemini_api_key"]
            genai.configure(api_key=api_key)
            
            # Luôn dùng Flash cho RAG vì nó đọc context dài tốt và rẻ
            model = genai.GenerativeModel(
                "gemini-2.5-flash",
                system_instruction=instruction
            )
            
            # Cắt bớt nếu text quá dài (tránh lỗi quá tải)
            max_chars = 200000 
            truncated_text = text[:max_chars]
            
            if len(text) > max_chars:
                st.warning(f"⚠️ Tài liệu quá dài, chỉ phân tích {max_chars:,} ký tự đầu.")
            
            response = model.generate_content(truncated_text)
            
            if response and hasattr(response, 'text') and response.text:
                return response.text
            else:
                return "⚠️ Không có phản hồi từ AI."
                
        except Exception as e:
            return f"❌ Lỗi phân tích tĩnh: {str(e)[:200]}"
