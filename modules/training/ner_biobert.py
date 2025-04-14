# FILE: modules/training/ner_biobert.py

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class ClinicalNER:
    def __init__(self, model_name="d4data/biobert_ner", device=-1):  # -1 = CPU
        self.nlp = pipeline("ner", model=model_name, tokenizer=model_name, grouped_entities=True, device=device)

    def extract_entities(self, text: str):
        return self.nlp(text)

if __name__ == "__main__":
    text = "The patient reports shortness of breath and a history of asthma. Elevated WBC count noted."
    ner = ClinicalNER()
    entities = ner.extract_entities(text)
    print("Entities:", entities)
