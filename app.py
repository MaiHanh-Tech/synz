import streamlit as st

# 1. CẤU HÌNH TRANG
st.set_page_config(page_title="Cognitive Weaver", layout="wide", page_icon="🏢")

# 2. LOAD AUTH (CORE)
try:
    from auth_block import AuthBlock
    auth = AuthBlock()
except ImportError:
    st.error("❌ Thiếu file 'auth_block.py'. Hãy tạo file này trước!")
    st.stop()
except Exception as e:
    st.error(f"❌ Lỗi khởi tạo Auth: {e}")
    st.stop()

# SIMPLE SAFE WRAPPER
def safe_run_module(module_func, module_name, components):
    try:
        module_func(components)  # Pass components
    except Exception as e:
        st.error(f"❌ Module {module_name} gặp lỗi:")
        st.exception(e)
        st.info("💡 Hãy reload trang hoặc chọn module khác")

# 3. LOGIN UI
if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False

if not st.session_state.user_logged_in:
    st.title("🔐 Đăng Nhập Hệ Thống/Log in")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        pwd = st.text_input("Nhập mật khẩu:", type="password", placeholder="Nhập mật khẩu của bạn")
        if st.button("Truy cập", use_container_width=True):
            if auth.login(pwd):
                st.success("✅ Đăng nhập thành công!")
                st.rerun()
            else:
                st.error("❌ Sai mật khẩu!")
                attempts = st.session_state.get('login_attempts', {}).get('global', [])
                remaining = 5 - len(attempts)
                if remaining > 0:
                    st.warning(f"⚠️ Còn {remaining} lần thử")
    st.stop()

# 4. SIDEBAR & NAVIGATION
with st.sidebar:
    st.title("🗂️ DANH MỤC ỨNG DỤNG/LIST APP")
    user_name = st.session_state.current_user.replace("Super", "")
    st.info(f"👤 Hello: **{user_name.strip()}**")
    app_choice = st.radio("Chọn công việc:", [
        "💰 1. Cognitive Weaver",
        "🧠 2. CFO Controller"
    ])
    st.divider()
    if st.button("Đăng Xuất/Log out"):
        st.session_state.user_logged_in = False
        st.rerun()

    # Admin panel (nếu có)
    if st.session_state.get("is_admin"):
        st.divider()
        st.write("👑 **Admin Panel**")
        try:
            all_users = auth.get_all_users()
            if all_users:
                import pandas as pd
                df_users = pd.DataFrame(all_users)
                display_cols = [col for col in ['username', 'role', 'is_active', 'created_at'] if col in df_users.columns]
                st.dataframe(df_users[display_cols], hide_index=True)
            with st.expander("Quản lý Người dùng"):
                new_u = st.text_input("Username:")
                new_p = st.text_input("Password:", type="password")
                new_role = st.selectbox("Role:", ["user", "admin"])
                if st.button("Tạo User"):
                    if new_u and new_p:
                        ok, msg = auth.create_user(new_u, new_p, new_role)
                        if ok:
                            st.success(msg); st.rerun()
                        else:
                            st.error(msg)
        except Exception:
            st.warning("Không thể tải danh sách user từ DB")

# 5. LOAD UI MODULES AN TOÀN WITH ORCHESTRATOR
from orchestrator import CognitiveApp  # New import

try:
    app_orch = (CognitiveApp()
                .with_ai()
                .with_translator()
                .with_voice()
                .with_logger()
                .with_db()
                .with_knowledge_graph()
                .with_personal_rag(st.session_state.current_user)  # Pass user_id
                .build())

    if app_choice == "💰 1. Cognitive Weaver":
        import module_weaver as mw
        safe_run_module(mw.run, "Cognitive Weaver", app_orch.components)
    elif app_choice == "🧠 2. CFO Controller":
        import module_cfo as mc
        safe_run_module(mc.run, "CFO Controller", app_orch.components)
except ImportError as e:
    st.error(f"⚠️ Lỗi: Không tìm thấy module tương ứng!\nChi tiết: {e}")
    st.info("👉 Hãy đảm bảo đã có các file UI: module_cfo.py, module_translator.py, module_weaver.py")
except Exception as e:
    st.error(f"❌ Lỗi nghiêm trọng: {e}")
    st.exception(e)
