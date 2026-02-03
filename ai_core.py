import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading

# Exceptions
from google.api_core.exceptions import ResourceExhausted as GeminiResourceExhausted
from google.api_core.exceptions import ServiceUnavailable as GeminiServiceUnavailable, InternalServerError as GeminiInternalError
from openai import RateLimitError, APIError, AuthenticationError, Timeout

class AI_Core:
    def __init__(self):
        self.status_container = st.container()
        self.status_message = st.empty()
        self.grok_ready = False
        self.gemini_ready = False
        self.deepseek_ready = False
        self.grok_client = None
        self.deepseek_client = None

        # ✅ TIMEOUT PHÂN CẤP (Jaynes: "Adapt prior based on context complexity")
        self.TIMEOUT_FAST = 15      # Solo debate, simple query
        self.TIMEOUT_NORMAL = 30    # Standard RAG
        self.TIMEOUT_COMPLEX = 60   # Multi-turn debate, Bayes analysis
        
        # ✅ PARALLEL EXECUTOR cho multi-persona
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # 1. DEEPSEEK
        try:
            if "deepseek" in st.secrets and "api_key" in st.secrets["deepseek"]:
                self.deepseek_client = OpenAI(
                    api_key=st.secrets["deepseek"]["api_key"],
                    base_url="https://api.deepseek.com/v1",
                    timeout=self.TIMEOUT_NORMAL
                )
                self.deepseek_ready = True
        except Exception:
            pass
        
        # 2. GEMINI (GIỮ NGUYÊN - ĐÃ TỐI ƯU)
        try:
            if "api_keys" in st.secrets and "gemini_api_key" in st.secrets["api_keys"]:
                genai.configure(api_key=st.secrets["api_keys"]["gemini_api_key"])
                self.safety_settings = [
                    {"category": c, "threshold": "BLOCK_NONE"} for c in [
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT"
                    ]
                ]
                self.gen_config = genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=7000,  
                    top_p=0.9,
                    top_k=40
                )
                self.gemini_ready = True
        except Exception:
            pass

        # 3. GROK
        try:
            if "xai" in st.secrets and "api_key" in st.secrets["xai"]:
                self.grok_client = OpenAI(
                    api_key=st.secrets["xai"]["api_key"],
                    base_url="https://api.x.ai/v1",
                    timeout=self.TIMEOUT_NORMAL
                )
                self.grok_ready = True
        except Exception:
            pass

        # Status
        with self.status_container:
            status_parts = []
            if self.deepseek_ready: status_parts.append("🟣 DeepSeek")
            if self.gemini_ready: status_parts.append("🟡 Gemini")
            if self.grok_ready: status_parts.append("🟢 Grok")
            if not status_parts:
                st.error("🔴 Không có API nào sẵn sàng")
            else:
                st.caption(f"**AI Engine:** {' → '.join(status_parts)}")

    # ===========================
    # ✅ PHƯƠNG PHÁP 1: PARALLEL RACING (Wittgenstein: "Don't wait for certainty, act on best evidence")
    # ===========================
    def _parallel_race(self, prompt, system_instruction=None, max_tokens=2000, timeout=30):
        """
        Gọi TẤT CẢ API song song, trả về kết quả đầu tiên thành công.
        Áp dụng cho: Debate 2-3 nhân cách, Bayes analysis
        
        Triết lý: Thay vì fallback tuần tự (Gemini → DeepSeek → Grok), 
        ta "đua" tất cả cùng lúc như thí nghiệm vật lý đo nhiều detector.
        """
        results = {}
        lock = threading.Lock()
        
        def _try_gemini():
            try:
                if self.gemini_ready:
                    res = self._gemini_generate(prompt, "flash", system_instruction)
                    if res:
                        with lock:
                            results['gemini'] = res
            except:
                pass
        
        def _try_deepseek():
            try:
                if self.deepseek_ready:
                    res = self._deepseek_generate(prompt, system_instruction, max_tokens, timeout)
                    if res:
                        with lock:
                            results['deepseek'] = res
            except:
                pass
        
        def _try_grok():
            try:
                if self.grok_ready:
                    res = self._grok_generate(prompt, system_instruction, max_tokens, timeout)
                    if res:
                        with lock:
                            results['grok'] = res
            except:
                pass
        
        # ✅ BẮN ĐỒNG THỜI
        futures = []
        if self.gemini_ready:
            futures.append(self.executor.submit(_try_gemini))
        if self.deepseek_ready:
            futures.append(self.executor.submit(_try_deepseek))
        if self.grok_ready:
            futures.append(self.executor.submit(_try_grok))
        
        # ✅ ĐỢI với timeout
        start = time.time()
        while time.time() - start < timeout:
            with lock:
                if results:
                    # Ưu tiên: Gemini > DeepSeek > Grok
                    if 'gemini' in results:
                        return results['gemini'], 'gemini'
                    elif 'deepseek' in results:
                        return results['deepseek'], 'deepseek'
                    elif 'grok' in results:
                        return results['grok'], 'grok'
            time.sleep(0.1)
        
        return None, None

    # ===========================
    # ✅ PHƯƠNG PHÁP 2: SMART FALLBACK với timeout động
    # ===========================
    def _deepseek_generate(self, prompt, system_instruction=None, max_tokens=2000, timeout=30):
        """DeepSeek với timeout tùy chỉnh"""
        if not self.deepseek_ready: 
            return None

        models = ["deepseek-chat"]
        messages = [{"role": "user", "content": prompt}]
        if system_instruction:
            messages.insert(0, {"role": "system", "content": system_instruction})

        for model in models:
            try:
                resp = self.deepseek_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=max_tokens,
                    timeout=timeout  # ✅ TIMEOUT ĐỘNG
                )
                return resp.choices[0].message.content.strip()
            except (RateLimitError, APIError):
                time.sleep(2)
                continue
            except Exception:
                continue
        return None

    def _gemini_generate(self, prompt, model_type="flash", system_instruction=None):
        """Gemini (GIỮ NGUYÊN - ĐÃ TỐI ƯU)"""
        if not self.gemini_ready: 
            return None
        
        valid_models = {
            "flash": "gemini-2.5-flash",
            "pro": "gemini-2.5-pro"  
        }
        model_name = valid_models.get(model_type, "gemini-2.5-flash")

        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings=self.safety_settings,
                generation_config=self.gen_config,
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            return None
        except (GeminiResourceExhausted, GeminiServiceUnavailable, GeminiInternalError):
            return None
        except Exception:
            return None

    def _grok_generate(self, prompt, system_instruction=None, max_tokens=2000, timeout=30):
        """Grok với timeout tùy chỉnh"""
        if not self.grok_ready: 
            return None

        models = ["grok-beta"]
        messages = [{"role": "user", "content": prompt}]
        if system_instruction:
            messages.insert(0, {"role": "system", "content": system_instruction})

        for model in models:
            try:
                resp = self.grok_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=max_tokens,
                    timeout=timeout  # ✅ TIMEOUT ĐỘNG
                )
                return resp.choices[0].message.content.strip()
            except (RateLimitError, APIError):
                time.sleep(2)
                continue
            except Exception:
                continue
        return None

    # ===========================
    # ✅ API CHÍNH - ADAPTIVE STRATEGY
    # ===========================
    def generate(self, prompt, model_type="pro", system_instruction=None, max_tokens=4000, use_parallel=False):
        """
        ✅ ADAPTIVE STRATEGY:
        - use_parallel=True: Dùng parallel racing (cho debate multi-turn, Bayes)
        - use_parallel=False: Dùng smart fallback (cho RAG, simple query)
        
        Triết lý: "Measure what is measurable, and make measurable what is not" (Galileo)
        """
        self.status_message.info("🤖 Đang gọi AI...")

        # ✅ STRATEGY 1: PARALLEL RACING (cho debate 2-3 nhân cách, Bayes)
        if use_parallel:
            result, source = self._parallel_race(
                prompt, 
                system_instruction, 
                max_tokens, 
                timeout=self.TIMEOUT_COMPLEX
            )
            if result:
                icons = {'gemini': '⚡', 'deepseek': '💰', 'grok': '🎯'}
                self.status_message.success(f"{icons.get(source, '✅')} {source.title()}")
                return result
        
        # ✅ STRATEGY 2: SMART FALLBACK (cho RAG, simple query)
        else:
            # 1. GEMINI FIRST (Nhanh nhất)
            if self.gemini_ready:
                result = self._gemini_generate(prompt, model_type, system_instruction)
                if result:
                    self.status_message.success("⚡ Gemini")
                    return result
            
            # 2. DEEPSEEK (Nếu Gemini fail)
            if self.deepseek_ready:
                result = self._deepseek_generate(
                    prompt, system_instruction, max_tokens, 
                    timeout=self.TIMEOUT_NORMAL
                )
                if result:
                    self.status_message.success("💰 DeepSeek")
                    return result

            # 3. GROK (Cuối cùng)
            if self.grok_ready:
                result = self._grok_generate(
                    prompt, system_instruction, max_tokens,
                    timeout=self.TIMEOUT_NORMAL
                )
                if result:
                    self.status_message.success("🎯 Grok")
                    return result
                
        self.status_message.error("⚠️ Tất cả API bận")
        return "⚠️ Hệ thống bận. Thử lại sau!"

    # ===========================
    # ✅ PHƯƠNG PHÁP ĐẶC BIỆT: BATCH GENERATION cho Multi-Persona
    # ===========================
    def generate_batch(self, prompts_dict, system_instructions_dict=None, max_tokens=3000):
        """
        Gọi NHIỀU prompt song song (cho 2-3 nhân cách debate).
        
        Input:
            prompts_dict: {"Persona1": "prompt1", "Persona2": "prompt2"}
            system_instructions_dict: {"Persona1": "sys1", "Persona2": "sys2"}
        
        Output:
            {"Persona1": "response1", "Persona2": "response2"}
        
        Triết lý: "In complex systems, parallel paths reveal truth faster than serial search" (Prigogine)
        """
        results = {}
        lock = threading.Lock()
        
        def _call_for_persona(persona_name, prompt):
            sys_inst = system_instructions_dict.get(persona_name) if system_instructions_dict else None
            
            # ✅ MỖI persona GỌI PARALLEL RACE
            result, source = self._parallel_race(
                prompt, 
                sys_inst, 
                max_tokens,
                timeout=self.TIMEOUT_COMPLEX
            )
            
            if result:
                with lock:
                    results[persona_name] = {
                        'content': result,
                        'source': source
                    }
        
        # ✅ TẠO THREAD cho mỗi persona
        threads = []
        for persona, prompt in prompts_dict.items():
            t = threading.Thread(target=_call_for_persona, args=(persona, prompt))
            t.start()
            threads.append(t)
        
        # ✅ ĐỢI tất cả threads (max 60s)
        for t in threads:
            t.join(timeout=self.TIMEOUT_COMPLEX)
        
        return results

    @staticmethod
    @st.cache_data(ttl=3600)
    def analyze_static(text, instruction):
        """✅ RAG dùng Gemini (có cache, nhanh) - GIỮ NGUYÊN"""
        try:
            # ✅ Ưu tiên Gemini cho RAG (có cache)
            if "api_keys" in st.secrets and "gemini_api_key" in st.secrets["api_keys"]:
                genai.configure(api_key=st.secrets["api_keys"]["gemini_api_key"])
                model = genai.GenerativeModel("gemini-2.5-flash")
                text = text[:150000]
                response = model.generate_content(f"{instruction}\n\n{text}")
                if response and response.text:
                    return response.text.strip()
            
            # Fallback DeepSeek
            if "deepseek" in st.secrets:
                client = OpenAI(
                    api_key=st.secrets["deepseek"]["api_key"],
                    base_url="https://api.deepseek.com/v1",
                    timeout=60
                )
                text = text[:180000]
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=4000,
                    temperature=0.3
                )
                return resp.choices[0].message.content.strip()
                
            return "❌ Không có API khả dụng"
        except Exception as e:
            return f"❌ RAG lỗi: {str(e)[:150]}"
