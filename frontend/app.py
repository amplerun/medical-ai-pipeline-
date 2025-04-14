import streamlit as st
import requests

st.title("Medical AI Diagnostic Assistant")

img_emb = st.text_area("Image Embedding (2048-dim)", "")
txt_emb = st.text_area("Text Embedding (768-dim)", "")
lab_vec = st.text_area("Lab Vector (10-dim)", "")
diagnosis = st.text_input("Initial Diagnoses (comma-separated)", "")
labs = st.text_area("Lab Dict (e.g. {\"WBC\": 8.0})", "")

if st.button("Submit"):
    payload = {
        "img_emb": list(map(float, img_emb.strip().split())),
        "txt_emb": list(map(float, txt_emb.strip().split())),
        "lab_vec": list(map(float, lab_vec.strip().split())),
        "diagnosis": diagnosis.split(","),
        "labs": eval(labs),
        "indicators": []
    }

    res = requests.post("https://your-backend-url.onrender.com/predict", json=payload)
    st.json(res.json())
