import os
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from Model_Zoo.Models.Util.Draw_Graph import Draw_Graph
from Model_Zoo.Models.ObjectDetection.Util.Utils import calculate_IoU, mAP, get_bboxes, YoloLoss
from DataLoader import Classification_data_loader, Object_Detection_data_loader, Compose
from torchvision.datasets import ImageFolder
from Models.Classification.MobileNetV1 import MobileNetV1
from Models.Classification.MobileNetV2 import MobileNetV2
from Models.Classification.MobileNetV3 import MobileNetV3
from Models.Classification.ResNet import ResNet
from Models.Classification.EfficientNet import EfficientNet
from Models.ObjectDetection.YoloV1 import Yolov1


def train_model(device, model, model_type, epochs, validation_epoch, learning_rate, patience, train_loader, validation_loader,
                test_loader, save_path):
    graph = Draw_Graph(save_path, patience)

    patience_count = 0

    model.train()
    #criterion, optimizer 선택 가능하게 변경 필요
    if model_type == "Classification":
        criterion = torch.nn.CrossEntropyLoss()
    elif model_type == "Object_Detection":
        criterion = YoloLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pbar = tqdm(range(epochs), desc="Epoch Progress")

    for epoch in pbar:
        if model_type == "Classification":
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            if patience_count >= patience:
                graph.save_plt()
                break

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_accuracy = correct_train / total_train

            pbar.set_postfix({'Loss': f'{train_loss:.4f}', 'Accuracy': f'{train_accuracy * 100:.2f}%'})

            patience_count += 1
            train_accuracy = correct_train / total_train

            graph.save_train_best_model_info(model, epoch, train_accuracy, train_loss)

            if (epoch + 1) % validation_epoch == 0:
                graph.append_train_losses_and_acc(train_loss, train_accuracy)
                graph.update_graph(model, device, validation_loader, criterion, epoch, model_type)
                patience_count = graph.save_validation_best_model_info(model, epoch, patience_count)
                graph.save_train_info(patience_count)

        elif model_type == "Object_Detection":
            model.train()
            running_loss = 0.0
            train_mAP = 0.0

            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

            # Calculate train mAP
            model.eval()
            pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, device=device)
            train_mAP = mAP(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")

            pbar.set_postfix({'Train Loss': f'{train_loss:.4f}', 'Train mAP': f'{train_mAP:.4f}'})

            patience_count += 1
            graph.save_train_best_model_info(model, epoch, train_mAP, train_loss)

            if (epoch + 1) % validation_epoch == 0:
                graph.append_train_losses_and_acc(train_loss, train_mAP)
                graph.update_graph(model, device, validation_loader, criterion, epoch, model_type)
                patience_count = graph.save_validation_best_model_info(model, epoch, patience_count)
                graph.save_train_info(patience_count)

            model.train()

    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

    graph.save_test_info(total, correct, accuracy)


def main():
    num_class = 20
    epoch = 10000
    validation_epoch = 10
    patience = 1000
    batch_size = 32
    learning_rate = 0.001
    save_path = "D:/Model_Save/Test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet(num_class=num_class).to(device)
    # model = Yolov1(split_size=7, num_boxes=2, num_classes=num_class).to(device)


    model_name = model.__class__.__name__
    model_type = "Object_Detection" if model_name == 'Yolov1' else "Classification"

    transform_info = model.return_transform_info()
    if model_type == "Classification":
        load_path = "D:/Image_Data/FishData"
        train_loader, validation_loader, test_loader = Classification_data_loader(load_path, batch_size, transform_info)
    else:
        traincsvfile_path = "D:/Image_Data/Pascal/100examples.csv"
        testcsvfile_path = "D:/Image_Data/Pascal/test.csv"
        IMG_DIR = "D:/Image_Data/Pascal/images"
        LABEL_DIR = "D:/Image_Data/Pascal/labels"
        transform = Compose(transform_info)
        train_validation_set = Object_Detection_data_loader(traincsvfile_path, transform=transform,
                                                             img_dir=IMG_DIR, label_dir=LABEL_DIR)
        test_set = Object_Detection_data_loader(testcsvfile_path, transform=transform,
                                                             img_dir=IMG_DIR, label_dir=LABEL_DIR)

        train_set_num = int(0.8 * len(train_validation_set))
        validation_set_num = len(train_validation_set) - train_set_num

        train_set, validation_set = random_split(train_validation_set, [train_set_num, validation_set_num])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    train_model(device=device, model=model, model_type=model_type, epochs=epoch, validation_epoch=validation_epoch,
                learning_rate=learning_rate, patience=patience, train_loader=train_loader,
                validation_loader=validation_loader, test_loader=test_loader, save_path=save_path)


if __name__ == '__main__':
     main()
