# FILE: modules/preprocessing/text_embedding.py

from transformers import AutoTokenizer, AutoModel
import torch

class BioBERTEmbedder:
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_embedding.cpu().numpy()

if __name__ == "__main__":
    embedder = BioBERTEmbedder()
    text = "Patient presents with fever and respiratory distress."
    emb = embedder.get_embedding(text)
    print(f"Text Embedding Shape: {emb.shape}")
