import streamlit as st
from utils import read_txt

DATA_PATH = r"C:\Users\Yunus Emre\Documents\GitHub\ENGR330\data.txt"


def render():
    st.empty()
    # ---------- CSS ----------
    st.markdown("""
    <style>
    .metric-card {
        border-radius: 16px;
        padding: 26px;
        color: white;
        text-align: center;
        font-weight: 600;
        transition: transform 0.3s ease;
    }

    .metric-card.animate {
        animation: pulse 0.8s ease-in-out;
    }

    @keyframes pulse {
        0%   { transform: scale(1); box-shadow: 0 0 0 rgba(0,0,0,0); }
        50%  { transform: scale(1.05); box-shadow: 0 0 18px rgba(255,255,255,0.6); }
        100% { transform: scale(1); box-shadow: 0 0 0 rgba(0,0,0,0); }
    }

    .section-title {
        margin-top: 10px;
        margin-bottom: 20px;
        font-size: 26px;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- DATA ----------
    readings = read_txt(DATA_PATH)

    if not readings:
        st.warning("No sensor data available yet")
        return

    last = readings[-1]

    # String → float (CRITICAL)
    temp = float(last["temperature"])
    hum = float(last["humidity"])
    pres = float(last["pres"])

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
        if v < 980: return "#e74c3c"
        if v < 990: return "#e67e22"
        if v < 1000: return "#2ecc71"
        if v < 1010: return "#3498db"
        return "#9b59b6"

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

    row1 = st.columns(3)
    row2 = st.columns(3)  # şimdilik boş

    with row1[0]:
        metric_card(
            "Temperature",
            f"{temp:.2f}",
            "°C",
            temp_color(temp),
            is_updated
        )

    with row1[1]:
        metric_card(
            "Humidity",
            f"{hum:.2f}",
            "%",
            hum_color(hum),
            is_updated
        )

    with row1[2]:
        metric_card(
            "Pressure",
            f"{pres:.2f}",
            "hPa",
            pres_color(pres),
            is_updated
        )




