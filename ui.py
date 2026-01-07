import streamlit as st
import time

from _pages import dashboard, raw_data, charts

# Sayfa ayarları
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        /* Ana container üst padding azalt */
        .block-container {
            padding-top: 1rem;
        }

        /* Dashboard başlığı yukarı al */
        h2 {
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }

        /* Divider boşluğunu azalt */
        hr {
            margin-top: 0.5rem;
            margin-bottom: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar tamamen gizle
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sayfa listesi
PAGES = [
    ("Dashboard", dashboard.render),
    ("Raw Data", raw_data.render),
    ("Charts", charts.render),
]

# Session state
if "page_index" not in st.session_state:
    st.session_state.page_index = 0

# Navigation
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Sayfa ismini al
page_name, page_func = PAGES[st.session_state.page_index]

# Butonları ve başlığı aynı satıra al
col_left, col_center, col_right = st.columns([1, 8, 1])

with col_left:
    if st.button("⬅️ Previous", use_container_width=True):
        if st.session_state.page_index > 0:
            st.session_state.page_index -= 1
            st.rerun()

with col_center:
    st.markdown(
        f"<h2 style='text-align:center; margin-top:0px; margin-bottom:0px;'>{page_name}</h2>",
        unsafe_allow_html=True
    )

with col_right:
    if st.button("Next ➡️", use_container_width=True):
        if st.session_state.page_index < len(PAGES) - 1:
            st.session_state.page_index += 1
            st.rerun()

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
st.divider()

# ---- 🔥 SADECE SEÇİLİ SAYFAYI RENDER ET ----
PAGES[st.session_state.page_index][1]()

# ---- AUTO REFRESH ----
REFRESH_INTERVAL = 1  # saniye

time.sleep(REFRESH_INTERVAL)
st.rerun()




