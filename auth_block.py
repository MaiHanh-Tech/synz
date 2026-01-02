import streamlit as st
import hashlib
from datetime import datetime
try:
    from supabase import create_client, Client
except ImportError:
    st.error("⚠️ Thiếu thư viện supabase. Hãy thêm 'supabase' vào requirements.txt")

class AuthBlock:
    def __init__(self):
        # 1. Kết nối Supabase
        try:
            url = st.secrets["supabase"]["url"]
            key = st.secrets["supabase"]["key"]
            self.supabase: Client = create_client(url, key)
            self.db_connected = True
        except Exception as e:
            self.db_connected = False
            # st.error(f"Lỗi kết nối DB: {e}")

        # 2. Backdoor (Admin cứng trong secrets để phòng hộ)
        self.hard_admin_hash = st.secrets.get("admin_password_hash", "")

        # Init Session
        if 'user_logged_in' not in st.session_state: 
            st.session_state.user_logged_in = False

    def _hash_password(self, password):
        return hashlib.sha256(str(password).encode()).hexdigest()

    def login(self, password):
        """Logic đăng nhập: Ưu tiên DB, nếu DB sập thì dùng Hard Admin"""
        if not password: return False
        input_hash = self._hash_password(password)

        # CÁCH 1: Check Admin cứng (Phòng khi DB lỗi hoặc quên pass DB)
        if input_hash == self.hard_admin_hash:
            self._set_session("SuperAdmin", True, True)
            return True

        # CÁCH 2: Check Database Supabase (Chỉ check user = admin vì đây là form login tổng)
        if self.db_connected:
            try:
                # Lấy tất cả user đang active
                response = self.supabase.table("users").select("*").eq("is_active", True).execute()
                users = response.data
                
                for user in users:
                    if user['password_hash'] == input_hash:
                        is_admin = (user['role'] == 'admin')
                        self._set_session(user['username'], is_admin, True)
                        return True
            except Exception:
                pass # Lỗi DB thì thôi, trả về False
        
        return False

    def _set_session(self, u, admin, vip):
        st.session_state.user_logged_in = True
        st.session_state.current_user = u
        st.session_state.is_admin = admin
        st.session_state.is_vip = vip

    # --- CÁC HÀM QUẢN LÝ USER (CHO ADMIN) ---
    def create_user(self, username, password, role="user"):
        if not self.db_connected: return False, "Mất kết nối DB"
        try:
            p_hash = self._hash_password(password)
            data = {"username": username, "password_hash": p_hash, "role": role}
            self.supabase.table("users").insert(data).execute()
            return True, "Tạo thành công!"
        except Exception as e:
            return False, f"Lỗi: {str(e)}"

    def delete_user(self, username):
        if not self.db_connected: return False, "Mất kết nối DB"
        try:
            self.supabase.table("users").delete().eq("username", username).execute()
            return True, "Đã xóa!"
        except Exception as e:
            return False, f"Lỗi: {str(e)}"
    
    def get_all_users(self):
        if not self.db_connected: return []
        try:
            res = self.supabase.table("users").select("*").execute()
            return res.data
        except: return []
