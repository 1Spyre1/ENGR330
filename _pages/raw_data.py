import streamlit as st
import pandas as pd
from utils import read_txt

DATA_PATH = r"C:\Users\Yunus Emre\Documents\GitHub\ENGR330\data.txt"


def render():
    # ===== ÖNCE EKRANI TEMİZLE =====
    st.empty()
    
    # ===== DARK MODE + HEADER KALDIR =====
    st.markdown("""
    <style>

    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }

    header {
        background: rgba(0,0,0,0) !important;
        height: 0px !important;
    }

    div[data-testid="stToolbar"] {
        background: #0e1117 !important;
    }

    .block-container {
        padding-top: 1rem !important;
    }

    h1, h2, h3 {
        color: #ffffff;
    }
    
    /* DİĞER SAYFALARIN KARTLARINI GİZLE */
    .metric-card {
        display: none !important;
    }
    
    /* GRAFİKLERİ GİZLE (Raw Data sayfasında) */
    div[data-testid="stVerticalBlock"] > div > div > div > iframe {
        display: none !important;
    }
    
    /* Plotly chartları gizle */
    .js-plotly-plot {
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

    /* DATAFRAME BEYAZLIKLARINI DÜZELT */
    div[data-testid="stDataFrame"] {
        background-color: #0e1117 !important;
    }
    
    div[data-testid="stDataFrame"] > div {
        background-color: #111827 !important;
    }
    
    /* Tablo başlıkları beyaz */
    div[data-testid="stDataFrame"] table thead th {
        background-color: #0e1117 !important;
        color: #0e1117 !important;
        border-color: #374151 !important;
        font-weight: 600 !important;
    }
    
    /* Row indexleri (soldaki numaralar) */
    div[data-testid="stDataFrame"] table tbody th {
        background-color: #1f2937 !important;
        color: #9ca3af !important;
        border-color: #374151 !important;
    }
    
    /* Tablo hücreleri */
    div[data-testid="stDataFrame"] table tbody td {
        background-color: #111827 !important;
        color: #e5e7eb !important;
        border-color: #374151 !important;
    }
    
    /* Dataframe scrollbar */
    div[data-testid="stDataFrame"] ::-webkit-scrollbar {
        background: #1f2937;
    }
    
    div[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {
        background: #374151;
    }

    </style>
    """, unsafe_allow_html=True)

    # ===== TITLE =====
    st.markdown("### 📋 Raw Sensor Data (Last 50 Samples)")

    readings = read_txt(DATA_PATH)

    if not readings:
        st.warning("No data available yet")
        return

    # ===== LAST 50 =====
    last_50 = readings[-50:]
    df = pd.DataFrame(last_50)
    
    # Index'i sıfırla (soldaki row numaraları için)
    df = df.reset_index(drop=True)

    # ===== REAL DARK TABLE (Pandas Styler) =====
    styled_df = df.style \
        .set_properties(**{
            "background-color": "#111827",
            "color": "#e5e7eb",
            "border-color": "#374151"
        }) \
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#1f2937"),
                    ("color", "#ffffff"),
                    ("border-color", "#374151"),
                    ("font-weight", "600")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("border-color", "#374151")
                ]
            },
            {
                "selector": "tbody tr",
                "props": [
                    ("background-color", "#111827")
                ]
            },
            {
                "selector": "tbody tr:hover",
                "props": [
                    ("background-color", "#1f2937")
                ]
            },
            {
                "selector": ".row_heading",
                "props": [
                    ("background-color", "#1f2937"),
                    ("color", "#9ca3af"),
                    ("border-color", "#374151")
                ]
            },
            {
                "selector": ".blank",
                "props": [
                    ("background-color", "#1f2937"),
                    ("border-color", "#374151")
                ]
            }
        ])

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=550
    )





