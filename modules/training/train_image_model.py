# FILE: modules/training/train_image_model.py

import os
import torch
import numpy as np
from monai.networks.nets import densenet121
from monai.transforms import (
    Compose, LoadImage, Resize, EnsureChannelFirst, ScaleIntensity, ToTensor
)
from monai.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import pandas as pd

DEVICE = "cpu"

def get_transforms():
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize((224, 224)),
        ToTensor()
    ])

def load_dataset(csv_file, image_dir, max_samples=1000):
    df = pd.read_csv(csv_file)
    df = df[df['Finding Labels'].str.contains("Pneumonia") | (df['Finding Labels'] == "No Finding")]
    df = df.sample(n=min(max_samples, len(df)))
    df['label'] = df['Finding Labels'].apply(lambda x: 1 if "Pneumonia" in x else 0)

    data = []
    for _, row in df.iterrows():
        image_path = os.path.join(image_dir, row["Image Index"])
        data.append({"image": image_path, "label": row["label"]})
    return data

def train_model():
    data = load_dataset("data/NIH_Labels.csv", "data/NIH_Images", max_samples=500)
    train_data, val_data = train_test_split(data, test_size=0.2)

    train_ds = Dataset(train_data, transform=get_transforms())
    val_ds = Dataset(val_data, transform=get_transforms())

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    model = densenet121(spatial_dims=2, in_channels=1, out_channels=1).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_fn = BCEWithLogitsLoss()

    for epoch in range(5):  # lightweight training
        model.train()
        total_loss = 0
        for batch in train_loader:
            x, y = batch["image"].to(DEVICE), batch["label"].float().unsqueeze(1).to(DEVICE)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "models/pneumonia_classifier.pth")
    print("✅ Model saved as 'pneumonia_classifier.pth'")

if __name__ == "__main__":
    train_model()
