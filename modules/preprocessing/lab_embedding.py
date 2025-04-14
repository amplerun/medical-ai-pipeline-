# FILE: modules/preprocessing/lab_embedding.py

import pandas as pd
import numpy as np

LAB_FEATURES = [50868, 50862, 50882, 50902, 50912, 50931, 50960, 50983, 51006, 51265]  # example ITEMIDs

def structure_labs(df: pd.DataFrame, patient_id: int) -> np.ndarray:
    df_patient = df[df['subject_id'] == patient_id]
    lab_vector = []

    for itemid in LAB_FEATURES:
        values = df_patient[df_patient['itemid'] == itemid]['valuenum']
        if len(values) > 0:
            lab_vector.append(values.mean())
        else:
            lab_vector.append(0.0)  # default missing

    return np.array(lab_vector).reshape(1, -1)

if __name__ == "__main__":
    # Simulated lab DataFrame
    data = {
        "subject_id": [1]*4 + [2]*3,
        "itemid": [50868, 50862, 50902, 50931, 50868, 50862, 50902],
        "valuenum": [8.1, 145, 36.6, 7.3, 7.9, 143, 37.1]
    }
    df = pd.DataFrame(data)
    lab_vec = structure_labs(df, patient_id=1)
    print(f"Lab Vector Shape: {lab_vec.shape}")
