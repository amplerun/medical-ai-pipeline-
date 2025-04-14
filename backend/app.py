from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from modules.fusion.fusion_model import FusionModel
from modules.logic.apply_rules import RuleEngine

app = FastAPI()
model = FusionModel()
model.load_state_dict(torch.load("models/fusion_model.pth", map_location=torch.device("cpu")))
model.eval()

rule_engine = RuleEngine()

class PatientInput(BaseModel):
    img_emb: list
    txt_emb: list
    lab_vec: list
    diagnosis: list = []
    indicators: list = []
    labs: dict = {}

@app.post("/predict")
def predict(input: PatientInput):
    x_img = torch.tensor([input.img_emb], dtype=torch.float32)
    x_txt = torch.tensor([input.txt_emb], dtype=torch.float32)
    x_lab = torch.tensor([input.lab_vec], dtype=torch.float32)

    with torch.no_grad():
        output = model(x_img, x_txt, x_lab)
        prob = torch.sigmoid(output).item()
        label = int(prob > 0.5)

    rules = rule_engine.run_all(input.diagnosis, input.labs, input.indicators)

    return {
        "prediction": label,
        "confidence": round(prob, 4),
        "rules_triggered": rules
    }

