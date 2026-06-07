import sys
import os
import yaml
from pathlib import Path
root_path = str(Path(__file__).resolve().parents[3])

if root_path not in sys.path:
    sys.path.append(root_path)

import torch
import torch.onnx
import torchvision.models as models
from E_Segmentation.Unet.TransUNet import TransUNet

model_path = "D:/0. Model_Save_Folder/Model_Save_Folder_HA"

model = TransUNet(num_classes = 3, img_size=900)
model.load_state_dict(torch.load(model_path + "/Best_Accuracy_Validation.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 900, 900)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dummy_input = dummy_input.to(device)

onnx_file_path = model_path + "/model.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None
)

print("Finish")