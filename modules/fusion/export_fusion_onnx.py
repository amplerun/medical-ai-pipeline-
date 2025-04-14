# FILE: modules/fusion/export_fusion_onnx.py

import torch
from fusion_model import FusionModel

model = FusionModel()
model.eval()

# Dummy inputs for export
x_img = torch.randn(1, 2048)
x_txt = torch.randn(1, 768)
x_lab = torch.randn(1, 10)

torch.onnx.export(
    model,
    (x_img, x_txt, x_lab),
    "models/fusion_model.onnx",
    input_names=["img_emb", "txt_emb", "lab_vec"],
    output_names=["prediction"],
    dynamic_axes={"img_emb": {0: "batch"}, "txt_emb": {0: "batch"}, "lab_vec": {0: "batch"}},
    opset_version=11
)

print("✅ Fusion model exported to 'models/fusion_model.onnx'")
