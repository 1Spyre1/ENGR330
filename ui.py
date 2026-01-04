import streamlit as st
import os
import time
from utils import loading_fire_prediction_model, predict_fire, read_txt

DATA_PATH = "data.txt"

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;'>Fire Prediction System</h1>", unsafe_allow_html=True)

readings = read_txt(DATA_PATH)
st.dataframe(readings)
model = loading_fire_prediction_model()
predict_fire(model, x)
