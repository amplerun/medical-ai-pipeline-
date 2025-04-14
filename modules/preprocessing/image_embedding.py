# FILE: modules/preprocessing/image_embedding.py

import torch
from monai.networks.nets import densenet121
from monai.transforms import (
    ScaleIntensity, Resize, EnsureChannelFirst, Compose
)
import numpy as np

class ImageEmbedder:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = densenet121(pretrained=True, spatial_dims=2, in_channels=1, out_channels=14).features.to(device)
        self.model.eval()

        self.transform = Compose([
            EnsureChannelFirst(),
            ScaleIntensity(),
            Resize(spatial_size=(224, 224))
        ])

    def get_embedding(self, img: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            img_tensor = torch.tensor(img).unsqueeze(0)
            img_tensor = self.transform(img_tensor).float().to(self.device)
            features = self.model(img_tensor)
            pooled = torch.mean(features, dim=[2, 3])  # Global average pooling
            return pooled.cpu().numpy()

if __name__ == "__main__":
    dummy_image = np.random.rand(1024, 1024).astype(np.float32)
    embedder = ImageEmbedder()
    emb = embedder.get_embedding(dummy_image)
    print(f"Image Embedding Shape: {emb.shape}")
