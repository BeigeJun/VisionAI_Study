import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder

def classification_data_loader(str_path, batch_size, info):
    transform_info = info

    train_dataset = ImageFolder(root=str_path + "//train", transform=transform_info)
    validation_dataset = ImageFolder(root=str_path + "//validation", transform=transform_info)
    test_dataset = ImageFolder(root=str_path + "//test", transform=transform_info)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    return train_loader, validation_loader, test_loader


def train_model(device, model, train_loader, val_loader, test_loader, graph, optimizer_name = 'Adam',
                criterion_name = 'CrossEntropyLoss', epochs=20, lr=0.001, patience=5, graph_update_epoch = 10):

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    if criterion_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience_count = 0

    pbar = tqdm(total=epochs, desc='Total Progress', position=0)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        val_acc = 100 * correct_val / total_val
        val_loss = val_loss / len(val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0

        else:
            patience_count += 1
            if patience_count >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        graph.update_graph(train_acc, train_loss, val_acc, val_loss, epoch, patience_count)
        pbar.set_postfix({
            'Epoch': epoch + 1,
            'Train Acc': f'{train_acc:.2f}%',
            'Train Loss': f'{train_loss:.4f}',
            'Val Acc': f'{val_acc:.2f}%',
            'Val Loss': f'{val_loss:.4f}',
            'Best Val Acc': f'{best_val_acc:.2f}%'
        })
        pbar.update(1)

    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()

    test_acc = 100 * correct_test / total_test
    print(f'Final Test Accuracy: {test_acc:.2f}%')
