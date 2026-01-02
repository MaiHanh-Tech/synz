import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

def tao_data_full_kpi(start_date=None, months=24, seed=None):
    if seed:
        np.random.seed(seed)
    if not start_date:
        start_date = datetime.now() - timedelta(days=months*30)
    dates = pd.date_range(start=start_date, periods=months, freq="ME")
    df = pd.DataFrame({"Tháng": dates})
    base_revenue = 6000000000
    growth_rate = 0.05 / 12
    revenues = []
    for i in range(months):
        trend = base_revenue * (1 + growth_rate) ** i
        month = (i % 12) + 1
        if month in [11, 12]:
            seasonal = 1.15
        elif month in [1,2,3]:
            seasonal = 0.95
        else:
            seasonal = 1.0
        noise = np.random.uniform(0.9, 1.1)
        revenues.append(trend * seasonal * noise)
    df["Doanh Thu"] = revenues
    df["Giá Vốn"] = df["Doanh Thu"] * np.random.uniform(0.58, 0.62, months)
    base_salary = 700000000
    df["CP Lương"] = base_salary * np.random.uniform(0.95, 1.05, months)
    df["CP Marketing"] = df["Doanh Thu"] * np.random.uniform(0.08, 0.12, months)
    df["CP Khác"] = np.random.randint(100, 200, months) * 1000000
    df["Chi Phí VH"] = df["CP Lương"] + df["CP Marketing"] + df["CP Khác"]
    anomaly_months = np.random.choice(range(12, months-2), size=2, replace=False)
    for m in anomaly_months:
        anomaly_type = np.random.choice(['chi_phi_dot_bien', 'mat_khach_hang'])
        if anomaly_type == 'chi_phi_dot_bien':
            df.loc[m, "Chi Phí VH"] *= 1.8
        else:
            df.loc[m, "Doanh Thu"] *= 0.7
    df["Lợi Nhuận ST"] = df["Doanh Thu"] - df["Giá Vốn"] - df["Chi Phí VH"]
    df["Dòng Tiền Thực"] = df["Lợi Nhuận ST"] * np.random.uniform(0.75, 0.85, months)
    df["Công Nợ Phải Thu"] = df["Doanh Thu"] * np.random.uniform(0.15, 0.25, months)
    df["Hàng Tồn Kho Tổng"] = df["Giá Vốn"] * np.random.uniform(0.2, 0.3, months)
    df["TS Ngắn Hạn"] = (df["Công Nợ Phải Thu"] + df["Hàng Tồn Kho Tổng"] + np.random.randint(500, 1000, months) * 1000000)
    df["Nợ Ngắn Hạn"] = df["TS Ngắn Hạn"] * np.random.uniform(0.4, 0.6, months)
    df["Vốn Chủ Sở Hữu"] = np.random.randint(5000, 6000, months) * 1000000
    return df

def validate_uploaded_data(df):
    required_columns = ["Tháng", "Doanh Thu", "Chi Phí VH", "Lợi Nhuận ST"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return False, f"Thiếu cột: {', '.join(missing)}"
    if (df["Doanh Thu"] < 0).any():
        return False, "Doanh thu không được âm"
    for col in ["Doanh Thu", "Chi Phí VH"]:
        if col in df.columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ((df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)).sum()
            if outliers > len(df) * 0.1:
                return False, f"Cột {col} có quá nhiều giá trị bất thường ({outliers}/{len(df)})"
    return True, "OK"

def tinh_chi_so(df):
    try:
        df["Current Ratio"] = df["TS Ngắn Hạn"] / df["Nợ Ngắn Hạn"].replace(0, 1)
        df["Gross Margin"] = (df["Doanh Thu"] - df["Giá Vốn"]) / df["Doanh Thu"].replace(0, 1) * 100
        df["ROS"] = df["Lợi Nhuận ST"] / df["Doanh Thu"].replace(0, 1) * 100
    except:
        pass
    return df

def phat_hien_gian_lan(df):
    iso = IsolationForest(contamination=0.05, random_state=42)
    col = "Chi Phí VH" if "Chi Phí VH" in df.columns else df.columns[1]
    data_clean = df[[col]].fillna(0)
    df['Anomaly'] = iso.fit_predict(data_clean)
    return df[df['Anomaly'] == -1]
