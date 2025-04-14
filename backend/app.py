from fastapi import FastAPI
from pydantic import BaseModel
import requests
from modules.logic.apply_rules import RuleEngine

app = FastAPI()
rule_engine = RuleEngine()

# Store your API key here (you can also read from env var)
LAMA_API_KEY = "1015506f-fb94-4ca4-9f07-af03ea3b1ca0"
LAMA_API_URL = "https://lama-api.com/predict"

class PatientInput(BaseModel):
    img_emb: list
    txt_emb: list
    lab_vec: list
    diagnosis: list = []
    indicators: list = []
    labs: dict = {}

@app.post("/predict")
def predict(input: PatientInput):
    # Prepare data for Lama API
    lama_payload = {
        "img_emb": input.img_emb,
        "txt_emb": input.txt_emb,
        "lab_vec": input.lab_vec
    }

    headers = {
        "Authorization": f"Bearer {LAMA_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(LAMA_API_URL, json=lama_payload, headers=headers)

    if response.status_code != 200:
        return {"error": "Failed to get response from Lama API", "status": response.status_code}

    lama_result = response.json()
    rules = rule_engine.run_all(input.diagnosis, input.labs, input.indicators)

    return {
        "prediction": lama_result["prediction"],
        "confidence": round(lama_result["confidence"], 4),
        "rules_triggered": rules
    }
