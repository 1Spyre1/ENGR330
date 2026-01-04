import streamlit as st
import pandas as pd
from utils import read_txt

DATA_PATH = r"C:\Users\Yunus Emre\Documents\GitHub\ENGR330\data.txt"

def render():
    st.subheader("Raw Sensor Data (Last 50 Samples)")

    readings = read_txt(DATA_PATH)

    if not readings:
        st.warning("No data available yet")
        return

    # 🔥 SADECE SON 50 SATIR
    last_50 = readings[-50:]

    df = pd.DataFrame(last_50)

    st.dataframe(
        df,
        use_container_width=True,
        height=550
    )



