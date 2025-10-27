from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
from G_Model_Zoo.Models.ObjectDetection.Util.Utils import calculate_IoU, mAP, get_bboxes, YoloLoss

def train_model(device, model, train_loader, val_loader, test_loader, graph, epochs=20, lr=0.001,
                patience=5, graph_update_epoch = 10):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = YoloLoss()

    best_val_acc = 0
    patience_count = 0

    pbar = tqdm(range(epochs), desc="Epoch Progress")

    for epoch in pbar:

        model.train()
        running_loss = 0.0

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
        pred_boxes, target_boxes = get_bboxes(val_loader, model, iou_threshold=0.5, threshold=0.4, device=device)
        train_mAP = mAP(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")

        pbar.set_postfix({'Train Loss': f'{train_loss:.4f}', 'Train mAP': f'{train_mAP:.4f}'})

        graph.update_graph(train_acc=train_mAP, train_loss=train_loss, val_acc=train_mAP, val_loss=0.0, epoch=epoch, patience_count=patience_count)

        patience_count += 1

    graph.save_model()

    # model.eval()
    # with torch.no_grad():
    #     total = 0
    #     correct = 0
    #     for inputs, labels in test_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = model(inputs)
    #
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    #     accuracy = 100 * correct / total
