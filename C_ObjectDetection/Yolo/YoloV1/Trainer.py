import os
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
from F_Model_Zoo.Models.Util.Draw_Graph import Draw_Graph
from F_Model_Zoo.Models.ObjectDetection.Util.Utils import calculate_IoU, mAP, get_bboxes, YoloLoss

torch.backends.cudnn.enabled = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def train_model(device, model, model_type, epochs, validation_epoch, learning_rate, patience, train_loader, validation_loader,
                test_loader, save_path, image_count):

    graph = Draw_Graph(save_path, patience)

    patience_count = 0
    model.train()
    criterion = YoloLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    pbar = tqdm(range(epochs), desc="Epoch Progress")

    for epoch in pbar:

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
