import streamlit as st
import requests
import numpy as np

st.title("🩺 Medical AI Diagnosis (Demo)")
use_demo = st.checkbox("Use Demo Vectors", value=True)

if use_demo:
    img_emb = np.random.rand(2048).tolist()
    txt_emb = np.random.rand(768).tolist()
    lab_vec = np.random.rand(10).tolist()
    diagnosis = ["pneumonia", "asthma"]
    indicators = ["infiltrate_xray"]
    labs = {"WBC": 13000}
else:
    st.warning("Custom input uploading not enabled yet.")
    st.stop()

if st.button("🧠 Predict"):
    data = {
        "img_emb": img_emb,
        "txt_emb": txt_emb,
        "lab_vec": lab_vec,
        "diagnosis": diagnosis,
        "indicators": indicators,
        "labs": labs
    }
    res = requests.post("http://localhost:8000/predict", json=data)

    if res.status_code == 200:
        output = res.json()
        st.success(f"Prediction: {'Positive' if output['prediction'] else 'Negative'}")
        st.write("Confidence:", output["confidence"])
        st.json(output["rules_triggered"])
    else:
        st.error("Prediction failed.")

