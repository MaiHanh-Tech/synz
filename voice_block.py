import edge_tts
import asyncio
import tempfile
import streamlit as st
import unicodedata
import re

# 👇 IMPORT MỚI: Để đồng bộ cấu hình và ghi log lỗi
from services.blocks.config import AppConfig
from services.blocks.logger import AppLogger

class Voice_Engine:
    def __init__(self):
        self.logger = AppLogger() # ✅ Khởi tạo Logger
        
        # ✅ LẤY GIỌNG TỪ CONFIG (Thay vì hardcode, giúp dễ quản lý tập trung)
        voices = AppConfig.TTS_VOICES
        self.VOICE_OPTIONS = {
            "🇻🇳 VN - Nữ (Hoài My)": voices["vi"]["female"],
            "🇻🇳 VN - Nam (Nam Minh)": voices["vi"]["male"],
            "🇺🇸 US - Nữ (Emma)": voices["en"]["female"],
            "🇺🇸 US - Nam (Andrew)": voices["en"]["male"],
            "🇨🇳 CN - Nữ (Xiaoyi)": voices["zh"]["female"],
            "🇨🇳 CN - Nam (Yunjian)": voices["zh"]["male"]
        }

    async def _gen(self, text, voice, rate):
        """Generate audio file asynchronously"""
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            # Tạo file tạm thời để tránh lỗi quyền ghi file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                await communicate.save(fp.name)
                return fp.name
        except Exception as e:
            st.error(f"Lỗi tạo audio: {e}")
            self.logger.log_error("Voice_Engine", str(e), "") # ✅ Ghi log lỗi
            return None

    def _clean_text_for_speech(self, text, voice_code):
        """
        ✅ LỌC VÀ CHUẨN HÓA VĂN BẢN CHO TỪ GIỌNG NÓI
        """
        if not text or not text.strip():
            return None
        
        # 1. ✅ SỬA LỖI REGEX (QUAN TRỌNG):
        # Code cũ dùng dải ký tự quá rộng làm mất chữ Hán.
        # Code này chỉ xóa đúng các Emoji và ký tự đặc biệt vô nghĩa.
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
            "\U0001F680-\U0001F6FF"  # Transport
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U0001F900-\U0001F9FF"  # Supplemental
            "]+", 
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # 2. ✅ XỬ LÝ THEO NGÔN NGỮ (Giữ nguyên logic của chị)
        if "vi-VN" in voice_code:
            text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
            
        elif "zh-CN" in voice_code:
            # Tiếng Trung: GIỮ NGUYÊN (Không xóa ký tự dựa trên category 'Lo' nữa)
            text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
            
        elif "en-US" in voice_code:
            try:
                text = unicodedata.normalize('NFKD', text)
                text = text.encode('ascii', 'ignore').decode('ascii')
            except:
                pass
        
        # 3. ✅ DỌN DẸP CUỐI CÙNG
        text = re.sub(r'\s+', ' ', text).strip()
        text = ''.join(char for char in text 
                      if char.isprintable() or char.isspace())
        
        # 4. ✅ GIỚI HẠN ĐỘ DÀI
        MAX_LENGTH = 4500
        if len(text) > MAX_LENGTH:
            text = text[:MAX_LENGTH]
            st.warning(f"⚠️ Văn bản quá dài. Chỉ đọc {MAX_LENGTH} ký tự đầu.")
        
        return text if text.strip() else None

    def speak(self, text, voice_key=None, speed=0):
        """
        Chuyển văn bản thành Audio Path
        """
        if not text: 
            return None
        
        # Fallback về giọng mặc định nếu key không tìm thấy
        default_voice = AppConfig.TTS_VOICES["vi"]["female"]
        voice_code = self.VOICE_OPTIONS.get(voice_key, default_voice)
        
        # ✅ LỌC VÀ CHUẨN HÓA VĂN BẢN
        cleaned_text = self._clean_text_for_speech(text, voice_code)
        
        if not cleaned_text:
            st.warning("⚠️ Văn bản không hợp lệ hoặc chỉ chứa ký tự đặc biệt")
            return None
        
        rate_str = f"{'+' if speed >= 0 else ''}{speed}%"

        try:
            # Chạy Async trong môi trường Sync của Streamlit
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            path = loop.run_until_complete(
                self._gen(cleaned_text, voice_code, rate_str)
            )
            loop.close()
            
            return path
            
        except Exception as e:
            st.error(f"❌ Lỗi tạo giọng nói: {e}")
            self.logger.log_error("Voice_Speak", str(e), "") # ✅ Ghi log lỗi
            return None
        finally:
            try:
                loop.close()
            except:
                pass
