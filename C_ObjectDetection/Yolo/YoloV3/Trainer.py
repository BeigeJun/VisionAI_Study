from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
from C_ObjectDetection.Yolo.YoloV3.Util import YoloV3Loss, calculate_IoU, mAP, get_bboxes

def train_model(device, model, train_loader, val_loader, test_loader, graph, epochs=20, lr=0.001,
                patience=5, graph_update_epoch = 10):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = YoloV3Loss()

    best_val_acc = 0
    patience_count = 0

    pbar = tqdm(range(epochs), desc="Epoch Progress")
    losses = []

    IMAGE_SIZE = 416
    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]
    scaler = torch.cuda.amp.GradScaler()

    scaled_anchors = (
            torch.tensor(ANCHORS)
            * torch.tensor([IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    for epoch in pbar:

        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y0, y1, y2 = (
                y[0].to(device),
                y[1].to(device),
                y[2].to(device),
            )

            with torch.cuda.amp.autocast():
                out = model(x)
                loss = (
                        loss_fn(out[0], y0, scaled_anchors[0])
                        + loss_fn(out[1], y1, scaled_anchors[1])
                        + loss_fn(out[2], y2, scaled_anchors[2])
                )

            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss = sum(losses) / len(losses)

        train_loss = running_loss

        # Calculate train mAP
        model.eval()
        pred_boxes, target_boxes = get_bboxes(val_loader, model, anchors = ANCHORS, iou_threshold=0.5, threshold=0.4, device=device)
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
