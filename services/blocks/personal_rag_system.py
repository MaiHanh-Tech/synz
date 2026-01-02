# (Nội dung nguyên bản personal_rag_system.py được đưa vào đây)
import streamlit as st
from datetime import datetime
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class PersonalRAG:
    def __init__(self, supabase_client, user_id):
        self.db = supabase_client
        self.user_id = user_id
        self.encoder = self._load_encoder()
        self.profile = self._load_user_profile()
    @st.cache_resource
    def _load_encoder(_self):
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device='cpu')
    def _load_user_profile(self):
        try:
            response = self.db.table("user_profiles").select("*").eq("user_id", self.user_id).execute()
            if response.data:
                return json.loads(response.data[0]["profile_json"])
            else:
                default_profile = {"user_id": self.user_id, "thinking_style": {}, "knowledge_interests": [], "interaction_history_embeddings": [], "last_updated": datetime.now().isoformat()}
                return default_profile
        except Exception as e:
            st.warning(f"Không load được profile: {e}")
            return {}
    def record_interaction(self, interaction_type, content, context=None):
        if not content or len(content.strip()) < 10:
            return
        embedding = self.encoder.encode([content])[0].tolist()
        data = {"user_id": self.user_id, "type": interaction_type, "content": content, "embedding": json.dumps(embedding), "context": json.dumps(context or {}), "timestamp": datetime.now().isoformat()}
        try:
            self.db.table("user_interactions").insert(data).execute()
        except Exception as e:
            st.warning(f"Không ghi được interaction: {e}")
    def update_profile(self, force=False):
        if not force:
            last_update = datetime.fromisoformat(self.profile.get("last_updated", "2020-01-01"))
            if (datetime.now() - last_update).days < 1:
                return
        try:
            response = self.db.table("user_interactions").select("*").eq("user_id", self.user_id).order("timestamp", desc=True).limit(100).execute()
            if not response.data or len(response.data) < 10:
                st.info("Chưa đủ dữ liệu để xây dựng profile (cần ít nhất 10 tương tác)")
                return
            interactions = response.data
            contents = [item["content"] for item in interactions]
            embeddings = [json.loads(item["embedding"]) for item in interactions]
            avg_embedding = np.mean(embeddings, axis=0).tolist()
            from collections import Counter
            all_words = " ".join(contents).lower().split()
            common_words = [word for word, count in Counter(all_words).most_common(20) if len(word) > 4]
            type_counts = Counter([item["type"] for item in interactions])
            dominant_type = type_counts.most_common(1)[0][0]
            tone_map = {"debate": "analytical, argumentative, logical", "translation": "multilingual, literary", "book_analysis": "scholarly, reflective, interdisciplinary", "query": "curious, information-seeking"}
            self.profile.update({
                "thinking_style": {"favorite_keywords": common_words[:10], "writing_tone": tone_map.get(dominant_type, "neutral"), "debate_strategy": "First-principles reasoning"},
                "interaction_history_embeddings": avg_embedding,
                "last_updated": datetime.now().isoformat()
            })
            profile_json = json.dumps(self.profile, ensure_ascii=False)
            self.db.table("user_profiles").upsert({"user_id": self.user_id, "profile_json": profile_json}).execute()
            st.success("✅ Đã cập nhật AI Profile!")
        except Exception as e:
            st.error(f"Lỗi update profile: {e}")
    def get_personalized_context(self, query, top_k=5):
        if not self.profile.get("interaction_history_embeddings"):
            return ""
        query_emb = self.encoder.encode([query])[0]
        try:
            response = self.db.table("user_interactions").select("*").eq("user_id", self.user_id).order("timestamp", desc=True).limit(50).execute()
            if not response.data:
                return ""
            interactions = response.data
            embeddings = [json.loads(item["embedding"]) for item in interactions]
            similarities = cosine_similarity([query_emb], embeddings)[0]
            import numpy as np
            top_indices = np.argsort(similarities)[::-1][:top_k]
            context_parts = []
            for idx in top_indices:
                item = interactions[idx]
                context_parts.append(f"[{item['type']}] {item['content'][:200]}")
            context = f"=== USER CONTEXT ===\nKeywords: {', '.join(self.profile.get('thinking_style', {}).get('favorite_keywords', []))}\nTone: {self.profile.get('thinking_style', {}).get('writing_tone', 'neutral')}\n=== RELEVANT PAST INTERACTIONS ===\n{chr(10).join(context_parts)}\n==="
            return context
        except Exception as e:
            st.warning(f"Không lấy được context: {e}")
            return ""
    def generate_persona_prompt(self):
        if not self.profile:
            return None
        style = self.profile.get("thinking_style", {})
        keywords = ", ".join(style.get("favorite_keywords", []))
        tone = style.get("writing_tone", "neutral")
        prompt = f'BẠN ĐANG MÔ PHỎNG PHONG CÁCH TƯ DUY CỦA USER "{self.user_id}".\\n- Từ ngữ ưa dùng: {keywords}\\n- Phong cách viết: {tone}\\n- Chiến lược lập luận: First-principles reasoning\\nNhiệm vụ: Trả lời theo phong cách này.'
        return prompt
