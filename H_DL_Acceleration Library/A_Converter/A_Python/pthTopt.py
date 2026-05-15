import torch
import yaml
import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_path not in sys.path:
    sys.path.append(root_path)

from E_Segmentation.Unet.TransUNet import TransUNet

yaml_path = os.path.join(root_path, 'E_Segmentation', 'Util', 'config.yaml')
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

num_classes = config['num_class']
input_size  = config['input_size']
pth_path    = os.path.join(config['save_path'], "Best_Accuracy_Validation.pth")
out_path    = os.path.join(config['save_path'], "model_cuda.pt")

device = torch.device("cuda")
model = TransUNet(num_classes=num_classes, img_size=input_size)
model.load_state_dict(torch.load(pth_path, map_location=device))
model.eval()

dummy = torch.zeros(1, 3, input_size, input_size).to(device)

with torch.no_grad():
    try:
        scripted = torch.jit.script(model)
    except Exception as e:
        print("Fail")
        scripted = torch.jit.trace(model, dummy, strict=False)

scripted.save(out_path)
print("Finish")
