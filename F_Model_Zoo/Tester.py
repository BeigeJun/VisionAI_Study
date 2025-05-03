import torch
from torchvision.datasets import ImageFolder
from Models.Classification.Grad_CAM import GradCam, find_last_layer, view_cam
from Models.Classification.MobileNetV1 import MobileNetV1
from Models.Classification.MobileNetV2 import MobileNetV2
from Models.Classification.MobileNetV3 import MobileNetV3
from Models.Classification.ResNet import ResNet
from Models.Classification.EfficientNet import EfficientNet


def data_loader(str_path, info):
    transform_info = info

    test_dataset = ImageFolder(root=str_path + "//test", transform=transform_info)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)
    return test_loader


def main():
    load_path = "D:/Image_Data/FishData"
    model = EfficientNet(num_class=3)
    transform_info = model.return_transform_info()
    test_loader = data_loader(load_path, transform_info)

    state_dict = torch.load('D:/Model_Save/Test/FishDataEffi/Bottom_Loss_Validation_MLP.pth')
    model.load_state_dict(state_dict)
    model.eval()

    target_layer = find_last_layer(model)
    grad_cam = GradCam(model=model, target_layer=target_layer)

    class_labels = test_loader.dataset.classes
    correct = 0
    for images, labels in test_loader:
        with torch.no_grad():
            outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)
        predicted_labels = torch.argmax(outputs, dim=1)
        correct += 1 if class_labels[labels.item()] == class_labels[predicted_labels.item()] else 0
        print("Actual Label:", class_labels[labels.item()])
        print("Predicted Label:", class_labels[predicted_labels.item()])
        print("Class Probabilities:")
        for i in range(probabilities.size(1)):
            print(f"{class_labels[i]}: {probabilities[0][i].item() * 100:.2f}%")
        print("-------------------")

        view_cam(images[0], model, grad_cam, class_labels, num_classes=len(class_labels))

    print(f"Acc : {correct / len(test_loader):.5f}%")


if __name__ == "__main__":
    main()
