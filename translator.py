import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
import json
import jieba
from pypinyin import pinyin, Style
from pydantic import BaseModel, Field
from typing import List
from collections import OrderedDict

class WordDefinition(BaseModel):
    word: str
    pinyin: str
    translation: str

class InteractiveTranslation(BaseModel):
    words: List[WordDefinition]

class Translator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            # ✅ SỬA: Đọc đúng cấu trúc secrets giống AI_Core
            try:
                if "api_keys" in st.secrets and "gemini_api_key" in st.secrets["api_keys"]:
                    self.api_key = st.secrets["api_keys"]["gemini_api_key"]
                elif "google_genai" in st.secrets and "api_key" in st.secrets["google_genai"]:
                    # Fallback cho cấu trúc cũ
                    self.api_key = st.secrets["google_genai"]["api_key"]
                else:
                    self.api_key = ""
                
                if self.api_key:
                    genai.configure(api_key=self.api_key)
            except Exception as e:
                st.error(f"❌ Lỗi đọc API Key: {e}")
                self.api_key = ""
            
            # 2. Cấu hình Model (Ưu tiên Flash 1.5 ổn định)
            self.model_flash = "gemini-2.5-flash" 
            self.model_pro = "gemini-2.5-pro"
            
            # ✅ SỬA: Đọc model config an toàn hơn
            try:
                if "google_genai" in st.secrets:
                    self.model_flash = st.secrets["google_genai"].get("model_flash", self.model_flash)
                    self.model_pro = st.secrets["google_genai"].get("model_pro", self.model_pro)
            except Exception:
                pass  # Giữ mặc định
                
            self.safety = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
            
            # ✅ DÙNG OrderedDict VỚI GIỚI HẠN
            self.cache = OrderedDict()
            self.MAX_CACHE_SIZE = 100  # Chỉ lưu 100 bản dịch gần nhất
            
            self.initialized = True

    def _generate(self, model_name, prompt, structured_output=None):
        if not self.api_key: 
            return "Error: Chưa nhập API Key"
        
        gen_config = {"temperature": 0.3}
        if structured_output:
            gen_config.update({
                "response_mime_type": "application/json",
                "response_schema": structured_output
            })

        model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=self.safety,
            generation_config=gen_config
        )

        # Thử tối đa 2 lần, chờ ngắn
        for attempt in range(2):
            try:
                response = model.generate_content(prompt)
                if response.text: 
                    return response.text
            except Exception as e:
                err = str(e)
                # Nếu sai model (404) -> Báo lỗi ngay để đổi model
                if "404" in err or "Not Found" in err:
                    return f"[Model Error: Model {model_name} không tồn tại. Hãy dùng gemini-2.5-flash]"
                
                # Nếu quá tải (429) -> Chờ 2s rồi thử lại 1 lần
                if "429" in err or "exhausted" in err:
                    time.sleep(2)
                    continue
                
                return f"[API Error: {err}]"
        
        return "[System Busy: Quá tải, vui lòng thử lại sau vài giây]"

    def _add_to_cache(self, key, value):
        """Thêm vào cache với giới hạn kích thước"""
        if len(self.cache) >= self.MAX_CACHE_SIZE:
            # Xóa item cũ nhất (FIFO)
            self.cache.popitem(last=False)
        self.cache[key] = value

    def translate_text(self, text, source, target, prompt_template=None):
        if not text.strip(): 
            return ""
        
        # ✅ SỬA LOGIC CACHE KEY (Cắt ngắn để tiết kiệm bộ nhớ key)
        cache_key = f"{text[:200]}|{source}|{target}"
        if cache_key in self.cache: 
            return self.cache[cache_key]

        full_prompt = f"{prompt_template}\n\nNguồn: {source}\nĐích: {target}\nVăn bản: {text}"
        
        # Luôn dùng Flash trước vì nó nhanh và rẻ
        res = self._generate(self.model_flash, full_prompt)
        
        # Nếu Flash lỗi model, tự động fallback sang hardcode 1.5-flash
        if "Model Error" in res:
            res = self._generate("gemini-2.5-flash", full_prompt)

        if "API Error" not in res and "System Busy" not in res:
            # ✅ SỬA CÁCH LƯU CACHE
            self._add_to_cache(cache_key, res.strip())
            
        return res.strip()

    def process_word_by_word(self, text, source, target):
        prompt = f"Phân tích từ vựng: '{text}' ({source}->{target})."
        res = self._generate(self.model_flash, prompt, structured_output=InteractiveTranslation)
        
        try:
            parsed = InteractiveTranslation.model_validate_json(res)
            return [w.model_dump() for w in parsed.words]
        except Exception:
            # Fallback: Dùng jieba cut cơ bản
            return [
                {
                    'word': w,
                    'pinyin': '',
                    'translations': []
                } 
                for w in jieba.cut(text)
            ]
