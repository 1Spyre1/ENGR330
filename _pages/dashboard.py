import streamlit as st
from utils import read_txt

DATA_PATH = r"C:\Users\Yunus Emre\Documents\GitHub\ENGR330\data.txt"


def render():
    st.empty()

    # ---------- CSS ----------
    st.markdown("""
<style>

/* 🌑 ANA ARKA PLAN */
.stApp {
    background-color: #0e1117;
    color: #e6e6e6;
}

/* 🚫 STREAMLIT ÜST BEYAZ BAR (HEADER) KALDIR */
header {
    background: rgba(0,0,0,0) !important;
    height: 0px !important;
}

/* Toolbar (Deploy, Stop vs.) */
div[data-testid="stToolbar"] {
    background: #0e1117 !important;
}

/* ÜST BOŞLUĞU AZALT */
.block-container {
    padding-top: 1rem !important;
}

/* BAŞLIKLAR */
h1, h2, h3, h4 {
    color: #ffffff;
}

/* METRIC CARD */
.metric-card {
    border-radius: 16px;
    padding: 26px;
    color: white;
    text-align: center;
    font-weight: 600;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.45);
}

/* BUTONLAR (Previous / Next) */
button {
    background-color: #1f2937 !important;
    color: white !important;
    border-radius: 8px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #0e1117;
}
::-webkit-scrollbar-thumb {
    background: #374151;
    border-radius: 4px;
}

</style>
""", unsafe_allow_html=True)


    # ---------- DATA ----------
    readings = read_txt(DATA_PATH)

    if not readings:
        st.warning("No sensor data available yet")
        return

    last = readings[-1]

    temp = float(last["temperature"])
    hum = float(last["humidity"])
    pres = float(last["pres"])

    # ---------- LAST 50 AVERAGES ----------
    last_50 = readings[-50:]

    avg_temp = sum(float(r["temperature"]) for r in last_50) / len(last_50)
    avg_hum = sum(float(r["humidity"]) for r in last_50) / len(last_50)

    # ---------- DATA CHANGE CHECK ----------
    prev = st.session_state.get("prev_last")
    is_updated = prev != last
    st.session_state.prev_last = last

    # ---------- COLOR LOGIC ----------
    def temp_color(v):
        if v < 10: return "#3498db"
        if v < 20: return "#2ecc71"
        if v < 25: return "#f1c40f"
        if v < 30: return "#e67e22"
        return "#e74c3c"

    def hum_color(v):
        if v < 20: return "#e74c3c"
        if v < 40: return "#e67e22"
        if v < 60: return "#2ecc71"
        if v < 80: return "#3498db"
        return "#9b59b6"

    def pres_color(v):
        return "#3498db"

    # ---------- CARD ----------
    def metric_card(title, value, unit, color, animate):
        cls = "metric-card animate" if animate else "metric-card"
        st.markdown(f"""
        <div class="{cls}" style="background:{color}">
            <div style="font-size:20px; margin-bottom:10px;">{title}</div>
            <div style="font-size:40px;">{value} {unit}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- UI ----------
    st.markdown('<div class="section-title">Live Last Data</div>', unsafe_allow_html=True)

    # 🔹 ÜST SATIR: Temperature | Pressure | Humidity
    row1 = st.columns(3, gap="large")

    with row1[0]:
        metric_card("Temperature", f"{temp:.2f}", "°C", temp_color(temp), is_updated)

    with row1[1]:
        metric_card("Pressure", f"{pres:.2f}", "hPa", pres_color(pres), is_updated)

    with row1[2]:
        metric_card("Humidity", f"{hum:.2f}", "%", hum_color(hum), is_updated)

    # 🔹 ALT SATIR: Avg Temp | boş | Avg Humidity
    row2 = st.columns(3, gap="large")

    with row2[0]:
        metric_card(
            "Avg Temperature (Last 50)",
            f"{avg_temp:.2f}",
            "°C",
            temp_color(avg_temp),
            True
        )

    with row2[2]:
        metric_card(
            "Avg Humidity (Last 50)",
            f"{avg_hum:.2f}",
            "%",
            hum_color(avg_hum),
            True
        )






