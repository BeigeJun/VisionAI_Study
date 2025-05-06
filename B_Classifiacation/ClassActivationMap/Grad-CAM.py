import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        #forward 호출 후에 forward output 계산 후 걸어두는 hook
        self.target_layer.register_forward_hook(forward_hook)
        #module input에 대한 gradient가 계산될 때마다 hook이 호출됨
        self.target_layer.register_full_backward_hook(backward_hook)

    def forward(self, input, target_index=None):
        output = self.model(input)

        if target_index is None:
            target_index = output.argmax(dim=1)

        self.model.zero_grad()
        #output[A,B]일때  A는 샘플의 순서(0이면 첫번째 샘플) 그리고 B는 라벨을 지정해줘서 그에대한 기대치를 알려주는 방법이라 한다
        #그 후 backward는 그 기대치에 대한 역전파
        output[0, target_index].backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        #한 채널에 해당되는 가로 세로의 평균을 계산
        #이는 각 채널이 얼마나 영향력을 행사하는지 알기 위해서
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        #브로드캐스팅으로 곱셈 후 덧셈 진행
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        #양의 값만 남기고 음은 없앰
        cam = F.relu(cam)
        #크기 조정
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        #정규화
        cam = cam - cam.min()
        cam = cam / cam.max()

        #크기가 1인 차원 제거
        return cam.squeeze().cpu().detach().numpy()


def find_last_layer(model):
    for layer in model.named_modules():
        if isinstance(layer[1], nn.Conv2d):
            return layer[1]
    return None


def view_cam(img_tensor, model, grad_cam, class_labels, num_classes=3):
    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean

    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    input_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax().item()

    fig, axes = plt.subplots(1, num_classes + 1, figsize=(20, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for class_idx in range(num_classes):
        mask = grad_cam.forward(input_tensor, target_index=class_idx)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)

        superimposed_img = cv2.addWeighted(img_np, 0.7, heatmap_resized, 0.3, 0)

        axes[class_idx + 1].imshow(superimposed_img)
        axes[class_idx + 1].set_title(f"{class_labels[class_idx]}")
        axes[class_idx + 1].axis('off')

    plt.suptitle(f"Predicted Label: {class_labels[predicted_class]}")
    plt.tight_layout()
    plt.show()