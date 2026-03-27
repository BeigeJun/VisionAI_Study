import torch
import torchvision
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(Image.fromarray(image)).unsqueeze(0)
    return image


def generate_cam(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    return cv2.resize(cam_img, size_upsample)


def visualize_cam(image_path):
    image_tensor = preprocess_image(image_path)

    features = []

    def hook(module, input, output):
        features.append(output)

    model.layer4.register_forward_hook(hook)

    with torch.no_grad():
        output = model(image_tensor)

    _, predicted = torch.max(output.data, 1)

    weight_softmax = model.fc.weight.data.numpy()
    cam = generate_cam(features[0].cpu().data.numpy(), weight_softmax, predicted)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    result = heatmap * 0.3 + img * 0.5

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(heatmap)
    plt.title('Heatmap')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(result.astype(np.uint8))
    plt.title('CAM Result')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = 'C:/Temp/Bird.png'
    visualize_cam(image_path)
