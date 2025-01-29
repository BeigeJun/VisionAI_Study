import os
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from Models.Classification.MobileNetV1 import MobileNetV1
from Models.Classification.MobileNetV2 import MobileNetV2
from Models.Classification.MobileNetV3 import MobileNetV3
from Models.Classification.ResNet import ResNet
from Models.Classification.EfficientNet import EfficientNet


def data_loader(str_path, info):
    transform_info = info

    train_dataset = ImageFolder(root=str_path + "//train", transform=transform_info)
    validation_dataset = ImageFolder(root=str_path + "//validation", transform=transform_info)
    test_dataset = ImageFolder(root=str_path + "//test", transform=transform_info)

    #num_workers는 데이터를 불러올 때 사용할 프로세스 수. 기본값은 0이고 커질수록 데이터를 불러오는 속도가 빨라짐.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, num_workers=4, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    return train_loader, validation_loader, test_loader


def train_model(device, model, epochs, patience, train_loader, validation_loader, test_loader, save_path):
    patience_count = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    top_accuracy_train = 0
    top_accuracy_validation = 0
    top_accuracy_train_epoch = 0
    top_accuracy_validation_epoch = 0

    bottom_loss_train = float('inf')
    bottom_loss_validation = float('inf')
    bottom_loss_train_epoch = 0
    bottom_loss_validation_epoch = 0

    os.makedirs(save_path, exist_ok=True)

    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4))

    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    pbar = tqdm(range(epochs), desc="Epoch Progress")
    for epoch in pbar:
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        if patience_count >= patience:
            plt.savefig(save_path + '//training_validation_graphs.png')
            plt.close()
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

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_train / total_train
        pbar.set_postfix({'Loss': f'{epoch_loss:.4f}', 'Accuracy': f'{epoch_accuracy:.2f}%'})

        patience_count += 1
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        if top_accuracy_train < train_accuracy:
            top_accuracy_train = train_accuracy
            model_path_make = save_path + '//Best_Accuracy_Train_MLP.pth'
            torch.save(model.state_dict(), model_path_make)
            top_accuracy_train_epoch = epoch

        if bottom_loss_train > running_loss:
            bottom_loss_train = running_loss
            model_path_make = save_path + '//Bottom_Loss_Train_MLP.pth'
            torch.save(model.state_dict(), model_path_make)
            bottom_loss_train_epoch = epoch

        if (epoch + 1) % 10 == 0:
            train_losses.append(running_loss / len(train_loader))
            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)

            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_losses.append(val_loss / len(validation_loader))
            val_accuracy = correct_val / total_val
            val_accuracies.append(val_accuracy)

            axs[0].clear()
            axs[1].clear()

            axs[0].plot(range(10, epoch + 2, 10), train_accuracies, label='Train Accuracy', color='red', linewidth=0.5)
            axs[0].plot(range(10, epoch + 2, 10), val_accuracies, label='Validation Accuracy', color='blue',
                        linewidth=0.5)
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('Accuracy')
            axs[0].set_title('Training and Validation Accuracy')
            axs[0].legend()

            axs[1].plot(range(10, epoch + 2, 10), train_losses, label='Train Loss', color='red', linewidth=0.5)
            axs[1].plot(range(10, epoch + 2, 10), val_losses, label='Validation Loss', color='blue', linewidth=0.5)
            axs[1].set_xlabel('Epochs')
            axs[1].set_ylabel('Loss')
            axs[1].set_title('Training and Validation Loss')
            axs[1].legend()

            for tick in axs[0].get_xticks():
                axs[0].axvline(x=tick, color='gray', linestyle='-', linewidth=0.1)

            for tick in axs[0].get_yticks():
                axs[0].axhline(y=tick, color='gray', linestyle='-', linewidth=0.1)

            for tick in axs[1].get_xticks():
                axs[1].axvline(x=tick, color='gray', linestyle='-', linewidth=0.1)

            for tick in axs[1].get_yticks():
                axs[1].axhline(y=tick, color='gray', linestyle='-', linewidth=0.1)

            plt.draw()
            plt.pause(0.1)

            if bottom_loss_validation > val_loss:
                bottom_loss_validation = val_loss
                model_path_make = save_path + '//Bottom_Loss_Validation_MLP.pth'
                torch.save(model.state_dict(), model_path_make)
                bottom_loss_validation_epoch = epoch
                patience_count = 0

            if top_accuracy_validation < val_accuracy:
                top_accuracy_validation = val_accuracy
                model_path_make = save_path + '//Best_Accuracy_Validation_MLP.pth'
                torch.save(model.state_dict(), model_path_make)
                top_accuracy_validation_epoch = epoch

            if (epoch + 1) % 50 == 0:
                plt.savefig(save_path + '//training_validation_graphs.png')

                with open(save_path + '//numbers.txt', "w") as file:
                    file.write(
                        f"Top Accuracy Train Epoch : {top_accuracy_train_epoch} Accuracy : {top_accuracy_train}\n"
                        f"Top Accuracy Validation Epoch : {top_accuracy_validation_epoch} Accuracy : {top_accuracy_validation}\n"
                        f"Bottom Loss Train Epoch : {bottom_loss_train_epoch} Loss : {bottom_loss_train}\n"
                        f"Bottom Loss Validation Epoch : {bottom_loss_validation_epoch} Loss : {bottom_loss_validation}\n"
                        f"Patience Count : {patience_count}/{patience}\n")

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

    with open(save_path + '//Test_Result.txt', "w") as file:
        file.write(f"Total Num : {total}, Correct Num : {correct}\n"
                   f"Accuracy : {accuracy}")


def main():
    num_class = 3
    epoch = 10000
    patience = 1000
    load_path = "D:/Image_Data/FishData"
    save_path = "D:/Model_Save/Test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(model_type='50', num_class=num_class).to(device)
    transform_info = model.return_transform_info()
    train_loader, validation_loader, test_loader = data_loader(load_path, transform_info)
    train_model(device=device, model=model, epochs=epoch, patience=patience, train_loader=train_loader,
                validation_loader=validation_loader, test_loader=test_loader, save_path=save_path)


if __name__ == '__main__':
     main()
