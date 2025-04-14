# FILE: modules/preprocessing/image_transforms.py

from monai.transforms import (
    LoadImage, ScaleIntensity, Resize, EnsureChannelFirst, Compose
)
import numpy as np

def get_image_transforms():
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize(spatial_size=(224, 224))
    ])

if __name__ == "__main__":
    import torch
    from monai.data import DataLoader, Dataset

    # Dummy image for test
    dummy_img = np.random.rand(1024, 1024).astype(np.float32)
    dataset = Dataset(data=[{"image": dummy_img}], transform=get_image_transforms())
    loader = DataLoader(dataset, batch_size=1)

    for batch in loader:
        print(f"Transformed Image Shape: {batch['image'].shape}")
