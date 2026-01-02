import streamlit as st
from supabase import create_client, Client

class DBBlock:
    """Wrapper đơn giản cho Supabase client + helper methods."""
    def __init__(self):
        self.client = None
        self.connected = False
        try:
            url = st.secrets["supabase"]["url"]
            key = st.secrets["supabase"]["key"]
            self.client: Client = create_client(url, key)
            self.connected = True
        except Exception:
            self.connected = False

    def insert_history(self, loai, title, content, user_name):
        if not self.connected: return False
        data = {
            "type": loai,
            "title": title,
            "content": content,
            "user_name": user_name,
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral"
        }
        try:
            self.client.table("history_logs").insert(data).execute()
            return True
        except Exception:
            # Try fallback table name
            try:
                self.client.table("History_Logs").insert(data).execute()
                return True
            except Exception:
                return False

    def get_history(self, limit=50):
        if not self.connected: return []
        try:
            resp = self.client.table("history_logs").select("*").order("created_at", desc=True).limit(limit).execute()
        except Exception:
            try:
                resp = self.client.table("History_Logs").select("*").order("created_at", desc=True).limit(limit).execute()
            except Exception:
                return []
        return resp.data or []

    # Helpers for user_profiles / interactions used by PersonalRAG
    def upsert_user_profile(self, user_id, profile_json):
        if not self.connected: return False
        try:
            self.client.table("user_profiles").upsert({"user_id": user_id, "profile_json": profile_json}).execute()
            return True
        except Exception:
            return False

    def insert_user_interaction(self, data):
        if not self.connected: return False
        try:
            self.client.table("user_interactions").insert(data).execute()
            return True
        except Exception:
            return False

    def query_user_interactions(self, user_id, limit=100):
        if not self.connected: return []
        try:
            resp = self.client.table("user_interactions").select("*").eq("user_id", user_id).order("timestamp", desc=True).limit(limit).execute()
            return resp.data or []
        except Exception:
            return []
