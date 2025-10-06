import os
import yaml
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from D_AnomalyDetection.Util.Loss import SSIM

def save_batch_images(images, folder_path, prefix='output', batch_num=0, n=4):
    os.makedirs(folder_path, exist_ok=True)
    to_pil = transforms.ToPILImage()

    images = images[:n].cpu().detach()
    for i, img_tensor in enumerate(images):
        pil_img = to_pil(img_tensor)
        save_path = os.path.join(folder_path, f'{batch_num}_{prefix}_{i}.png')
        pil_img.save(save_path)

def anomalydetection_data_loader(str_path, batch_size, info):
    transform_info = info

    train_dataset = ImageFolder(root=str_path + "//train", transform=transform_info)
    validation_dataset = ImageFolder(root=str_path + "//validation", transform=transform_info)
    test_dataset = ImageFolder(root=str_path + "//test", transform=transform_info)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    return train_loader, validation_loader, test_loader


def train_model(device, model, train_loader, val_loader, test_loader, graph, epochs=20, lr=0.01, patience=5, graph_update_epoch = 1):
    if model.model_name == "AutoEncoder":
        criterion = SSIM()
    else :
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    Best_val_loss = 100
    patience_count = 0

    pbar = tqdm(total=epochs, desc='Total Progress', position=0)

    lstTrainLosses = []
    lstValLosses = []
    lstValPredictLabel = []
    fMaxLoss = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        lstTrainLosses.clear()
        lstValLosses.clear()
        lstValPredictLabel.clear()

        for inputs, labels in train_loader:
            nCount = 0
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(inputs, outputs)
            loss.backward()
            optimizer.step()

            save_batch_images(outputs, folder_path='D://0. Model_Save_Folder//output_train_images',
                              prefix=f'epoch_{epoch}')
            lstTrainLosses.append(loss.item())
            nCount += 1

        train_loss = sum(lstTrainLosses) / len(lstTrainLosses)
        fMaxLoss = max(lstTrainLosses)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            nCount = 0
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(inputs, outputs)
                lstValLosses.append(loss.item())

                if fMaxLoss > lstValLosses[nCount]:
                    lstValPredictLabel.append(0)
                else:
                    lstValPredictLabel.append(1)

                save_batch_images(outputs, folder_path='D://0. Model_Save_Folder//output_validation_images',
                                  prefix=f'epoch_{epoch}')
                nCount += 1

        val_acc = 100 * lstValPredictLabel.count(0) / len(lstValPredictLabel)
        val_loss = sum(lstValLosses) / len(lstValLosses)

        if val_loss < Best_val_loss:
            Best_val_loss = val_loss
            patience_count = 0

        else:
            patience_count += 1
            if patience_count >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        if (epoch + 1) % graph_update_epoch == 0:
            graph.update_graph(
                train_acc=100,
                train_loss=train_loss,
                val_acc=val_acc,
                val_loss=val_loss,
                epoch=epoch,
                patience_count=patience_count
            )
            graph.save_plt()

        pbar.set_postfix({
            'Epoch': epoch + 1,
            # 'Train Acc': f'{100:.2f}%',
            'Train Loss': f'{train_loss:.4f}',
            'Val Acc': f'{val_acc:.2f}%',
            'Val Loss': f'{val_loss:.4f}',
            'Best Val Loss': f'{Best_val_loss:.8f}'
        })
        pbar.update(1)

    graph.save_plt()
    graph.save_train_info(patience_count)

    best_model_path = os.path.join(graph.save_path, 'Bottom_Loss_Train.pth')
    model.load_state_dict(torch.load(best_model_path))

    model.eval()

    lstLosses = []
    model.eval()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            for i in range(inputs.size(0)):
                loss = criterion(inputs[i:i + 1], outputs[i:i + 1])
                lstLosses.append(loss.item())

    model.Threshold = max(lstLosses)

    correct_test = 0
    total_test = 0
    Predict_results = []

    listPredict = ["TRUE", "FALSE"]
    with torch.no_grad():
        for inputs, labels in test_loader:
            nCount = 0
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            for i in range(inputs.size(0)):
                loss = criterion(inputs[i:i + 1], outputs[i:i + 1])
                pred = 0 if loss > model.Threshold else 1
                Predict_results.append(pred)
                correct_test += (pred == labels[i].item())
                total_test += 1

                save_batch_images(inputs, folder_path='D://0. Model_Save_Folder//output_test_images',
                                  batch_num=total_test, prefix=f'input_' + listPredict[pred])
                save_batch_images(outputs, folder_path='D://0. Model_Save_Folder//output_test_images',
                                  batch_num = total_test, prefix=f'output_' + listPredict[pred])
            nCount += 1

    test_acc = 100 * correct_test / total_test
    print(f'Final Test Accuracy: {test_acc:.2f}%')

    graph.save_test_info(
        total=total_test,
        correct=correct_test,
        accuracy=test_acc
    )