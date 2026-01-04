import streamlit as st
import time
from _pages import dashboard, raw_data

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="BLE Sensor Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- GLOBAL CSS ----------------
st.markdown(
    """
    <style>
        /* üst padding azalt */
        .block-container {
            padding-top: 1rem;
        }

        h2 {
            margin-top: 0rem !important;
            margin-bottom: 0.5rem !important;
        }

        hr {
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- PAGES ----------------
PAGES = [
    ("Dashboard", dashboard.render),
    ("Raw Data", raw_data.render),
]

if "page_index" not in st.session_state:
    st.session_state.page_index = 0

# ---------------- NAVIGATION ----------------
st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

left, center, right = st.columns([2, 6, 2])

with left:
    if st.button("⬅️ Previous"):
        st.session_state.page_index = max(0, st.session_state.page_index - 1)
        st.rerun()

with center:
    st.markdown(
        f"<h2 style='text-align:center;'>{PAGES[st.session_state.page_index][0]}</h2>",
        unsafe_allow_html=True
    )

with right:
    if st.button("Next ➡️"):
        st.session_state.page_index = min(len(PAGES) - 1, st.session_state.page_index + 1)
        st.rerun()

st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------- 🔥 SINGLE PLACEHOLDER ----------------
page_placeholder = st.empty()

with page_placeholder.container():
    # SADECE SEÇİLİ SAYFA ÇİZİLİR
    PAGES[st.session_state.page_index][1]()

# ---------------- AUTO REFRESH (SAFE) ----------------
REFRESH_INTERVAL = 1  # saniye

time.sleep(REFRESH_INTERVAL)
st.rerun()





