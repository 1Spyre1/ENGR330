import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import read_txt

DATA_PATH = r"C:\Users\Yunus Emre\Documents\GitHub\ENGR330\data.txt"


def render():
    # ===== ÖNCE EKRANI TEMİZLE =====
    st.empty()
    
    # ===== DARK MODE + BAR KALDIRMA =====
    st.markdown(
        """
        <style>
        header {visibility: hidden;}
        .stApp {
            background: radial-gradient(circle at top, #121826, #05080f);
            color: white;
        }
        
        /* DİĞER SAYFALARIN KARTLARINI GİZLE */
        .metric-card {
            display: none !important;
        }
        
        /* BUTONLARI DÜZELT */
        button[kind="secondary"] {
            background-color: #1f2937 !important;
            color: #e5e7eb !important;
            border: 1px solid #374151 !important;
            border-radius: 8px !important;
        }
        
        button[kind="secondary"]:hover {
            background-color: #374151 !important;
            border-color: #4b5563 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## 📈 Sensor Trends (Last 50 Samples)")

    readings = read_txt(DATA_PATH)
    if not readings:
        st.warning("No data available yet")
        return

    # ---- SON 50 DATA ----
    df = pd.DataFrame(readings[-50:])

    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
    df["pres"] = pd.to_numeric(df["pres"], errors="coerce")

    df = df.dropna()


    avg_temp = df["temperature"].mean()
    avg_hum = df["humidity"].mean()
    avg_pres = df["pres"].mean()

    # =========================
    # ÜSTTE 2 GRAFİK
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_line_chart(
                df,
                "temperature",
                "🌡 Temperature Trend (°C)",
                "#e74c3c",
                avg_temp
            ),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            create_line_chart(
                df,
                "humidity",
                "💧 Humidity Trend (%)",
                "#3498db",
                avg_hum
            ),
            use_container_width=True
        )

    # =========================
    # ALTTA PRESSURE
    # =========================
    st.plotly_chart(
        create_line_chart(
            df,
            "pres",
            "🧭 Pressure Trend (hPa)",
            "#2ecc71",
            avg_pres
        ),
        use_container_width=True
    )


# ==================================================
# GRAFİK FONKSİYONU
# ==================================================
def create_line_chart(df, column, title, color, avg_value):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=df[column],
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=5),
            name=column
        )
    )

    # Ortalama çizgisi
    fig.add_hline(
        y=avg_value,
        line_dash="dash",
        line_color="#9ca3af",
        annotation_text=f"Avg (last 50): {avg_value:.2f}",
        annotation_position="bottom left",
        annotation_font=dict(color="#e5e7eb")
    )

    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=40, r=40, t=60, b=40),
        title=dict(text=title, x=0.02, font=dict(color="#ffffff")),
        xaxis_title="Sample Index",
        yaxis_title=column,
        paper_bgcolor="#1f2937",
        plot_bgcolor="#111827",
        font=dict(color="#e5e7eb"),
        xaxis=dict(
            gridcolor="#374151",
            zerolinecolor="#374151"
        ),
        yaxis=dict(
            gridcolor="#374151",
            zerolinecolor="#374151"
        )
    )

    return fig






