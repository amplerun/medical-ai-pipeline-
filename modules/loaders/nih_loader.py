# FILE: modules/loaders/nih_loader.py

import os
import pandas as pd
import pydicom
import numpy as np
from glob import glob

def load_nih_images(image_dir: str):
    dicom_files = glob(os.path.join(image_dir, "*.dcm"))
    images = []
    for file in dicom_files:
        dicom = pydicom.dcmread(file)
        image = dicom.pixel_array.astype(np.float32)
        images.append(image)
    return images

if __name__ == "__main__":
    images = load_nih_images("data/NIH_DICOM/")
    print(f"Loaded {len(images)} NIH images.")
