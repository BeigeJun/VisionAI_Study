import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from F_Model_Zoo.Models.Util.Draw_Graph import Draw_Graph

def classification_data_loader(str_path, batch_size, info):
    transform_info = info

    train_dataset = ImageFolder(root=str_path + "//train", transform=transform_info)
    validation_dataset = ImageFolder(root=str_path + "//validation", transform=transform_info)
    test_dataset = ImageFolder(root=str_path + "//test", transform=transform_info)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    return train_loader, validation_loader, test_loader

def train_model(device, model, model_type, epochs, validation_epoch, learning_rate, patience, train_loader, validation_loader,
                test_loader, save_path):
    graph = Draw_Graph(save_path, patience)

    patience_count = 0

    model.train()
    criterion = nn.CrossEntropyLoss()

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
