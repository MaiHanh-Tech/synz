import streamlit as st
import pandas as pd
from services.blocks.cfo_data_manager import tao_data_full_kpi, validate_uploaded_data, tinh_chi_so, phat_hien_gian_lan
from ai_core import AI_Core

def run():
    ai = AI_Core()
    st.header("💰 CFO Controller Dashboard")
    with st.sidebar:
        st.markdown("---")
        st.write("📊 **Nguồn dữ liệu**")
        data_source = st.radio("Chọn nguồn:", ["Demo (Giả)", "Upload Excel"])
        if data_source == "Upload Excel":
            uploaded = st.file_uploader("Upload file Excel", type="xlsx")
            if uploaded:
                try:
                    df_raw = pd.read_excel(uploaded)
                    is_valid, msg = validate_uploaded_data(df_raw)
                    if is_valid:
                        st.session_state.df_fin = df_raw
                        st.success("✅ Tải data thành công!")
                    else:
                        st.error(f"❌ Lỗi data: {msg}")
                except Exception as e:
                    st.error(f"Lỗi đọc file: {e}")
        if st.button("🔄 Tạo data demo mới"):
            st.session_state.df_fin = tao_data_full_kpi(seed=int(st.time()))
            st.rerun()

    if 'df_fin' not in st.session_state:
        st.session_state.df_fin = tao_data_full_kpi(seed=42)

    df = tinh_chi_so(st.session_state.df_fin.copy())
    last = df.iloc[-1]

    t1, t2, t3, t4 = st.tabs(["📊 KPIs & Sức Khỏe", "📉 Phân Tích Chi Phí", "🕵️ Rủi Ro & Check", "🔮 Dự Báo & What-If"])

    with t1:
        st.subheader("Sức khỏe Tài chính Tháng gần nhất")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Doanh Thu", f"{last['Doanh Thu']/1e9:.1f} tỷ")
        k2.metric("Lợi Nhuận ST", f"{last['Lợi Nhuận ST']/1e9:.1f} tỷ")
        k3.metric("ROS", f"{last.get('ROS',0):.1f}%")
        k4.metric("Dòng Tiền", f"{last['Dòng Tiền Thực']/1e9:.1f} tỷ")
        st.line_chart(df.set_index("Tháng")[["Doanh Thu", "Lợi Nhuận ST"]])

    with t2:
        c1, c2 = st.columns([2,1])
        with c1:
            if "Giá Vốn" in df.columns and "Chi Phí VH" in df.columns:
                st.bar_chart(df.set_index("Tháng")[["Giá Vốn", "Chi Phí VH"]])
            else:
                st.info("Chưa có đủ cột dữ liệu chi phí để vẽ biểu đồ.")
        with c2:
            st.write("🤖 **Trợ lý Phân tích:**")
            q = st.text_input("Hỏi về chi phí...")
            if q:
                with st.spinner("AI đang soi số liệu..."):
                    context = f"Dữ liệu tháng cuối: Doanh thu {last['Doanh Thu']}, Lợi nhuận {last['Lợi Nhuận ST']}."
                    res = ai.generate(q, system_instruction=f"Bạn là Kế toán trưởng. Phân tích dựa trên: {context}")
                    st.write(res)

    with t3:
        c_risk, c_check = st.columns(2)
        with c_risk:
            st.subheader("Quét Gian Lận (ML)")
            if st.button("🔍 Quét ngay"):
                bad = phat_hien_gian_lan(df)
                if not bad.empty:
                    st.error(f"Phát hiện {len(bad)} tháng bất thường!")
                    st.dataframe(bad)
                else:
                    st.success("Dữ liệu sạch.")
        with c_check:
            st.subheader("Cross-Check (Đối chiếu)")
            val_a = st.number_input("Số liệu Thuế (Tờ khai):", value=100.0)
            val_b = st.number_input("Số liệu Sổ cái (ERP):", value=105.0)
            if st.button("So khớp"):
                diff = val_b - val_a
                if diff != 0:
                    st.warning(f"Lệch: {diff}. Rủi ro truy thu thuế!")
                else:
                    st.success("Khớp!")

    with t4:
        st.subheader("🎛️ What-If Analysis")
        base_rev = last['Doanh Thu']
        base_profit = last['Lợi Nhuận ST']
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            delta_price = st.slider("Tăng/Giảm Giá Bán (%)", -20, 20, 0)
        with c_s2:
            delta_cost = st.slider("Tăng/Giảm Chi Phí (%)", -20, 20, 0)
        new_rev = base_rev * (1 + delta_price/100)
        base_fixed_cost = last.get('Chi Phí VH', 0)
        new_profit = base_profit + (new_rev - base_rev) - (base_fixed_cost * delta_cost/100)
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Lợi Nhuận Gốc", f"{base_profit/1e9:.2f} tỷ")
        col_res2.metric("Lợi Nhuận Mới", f"{new_profit/1e9:.2f} tỷ", delta=f"{(new_profit - base_profit)/1e9:.2f} tỷ")
