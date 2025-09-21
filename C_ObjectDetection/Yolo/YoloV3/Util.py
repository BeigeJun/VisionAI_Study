import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        #구조에 맞게 변형
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        #IoU 생성
        iou_b1 = calculate_IoU(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = calculate_IoU(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        #더 높은 IoU 채택
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        #(1 - bestbox)을 통해 더 높은 상자만 남김
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )
        #신뢰도 계산
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss


def calculate_IoU(boxes_predict, boxes_labels, box_format="midpoint"):
    #박스 중심 기준
    if box_format == "midpoint":
        box1_x1 = boxes_predict[..., 0:1] - boxes_predict[..., 2:3] / 2
        box1_y1 = boxes_predict[..., 1:2] - boxes_predict[..., 3:4] / 2
        box1_x2 = boxes_predict[..., 0:1] + boxes_predict[..., 2:3] / 2
        box1_y2 = boxes_predict[..., 1:2] + boxes_predict[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    #박스 꼭짓점 기준
    if box_format == "corners":
        box1_x1 = boxes_predict[..., 0:1]
        box1_y1 = boxes_predict[..., 1:2]
        box1_x2 = boxes_predict[..., 2:3]
        box1_y2 = boxes_predict[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    #뒤집힘 방지
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    #교차 구역 넓이 구하기 .clamp(0)는 음수를 0으로 만듬 이는 안곂칠때를 위한 것
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    #두 구역 넓이 구하기
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    #IoU
    return intersection / (box1_area + box2_area - intersection + 1e-6)


def nms(bboxes, iou_threshold, threshold, box_format="corners"):
    assert type(bboxes) == list

    #임계값보다 높은 값만 유지
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        #IoU가 크다는 것은 사진상에서 바운딩 박스가 많이 곂친다는것. 따라서 임계값으로 정렬된 것에서 Iou 임계값보다 낮은것만 살림.
        #이를 통해 다른 객체에 대한 바운딩 박스는 살릴수 있음.
        bboxes = [box for box in bboxes
                  if box[0] != chosen_box[0]
                  or calculate_IoU(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]),
                                   box_format=box_format) < iou_threshold]

        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms


def mAP(predict_boxes, gt_boxes, iou_threshold, box_format="corners", num_class=20):
    average_precisions = []
    epsilon = 1e-6
    for class_ in range(num_class):
        detections = []
        ground_truths = []

        #[train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        #테이블을 만들기 위해 각 라벨에 대한 정답, 오답 정보 추출
        for detection in predict_boxes:
            if detection[1] == class_:
                detections.append(detection)

        for true_box in gt_boxes:
            if true_box[1] == class_:
                ground_truths.append(true_box)

        #해당 클래스의 개수를 딕셔너리 형태로 저장
        amount_bboxes = {}
        for gt in ground_truths:
            image_id = gt[0]
            if image_id in amount_bboxes:
                amount_bboxes[image_id] += 1
            else:
                amount_bboxes[image_id] = 1

        #결과 기록을 위한 탠서 생성
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        #TP, FP 판단
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = []
            #이미지 ID가 같으면 우선 추가
            for bbox in ground_truths:
                if bbox[0] == detection[0]:
                    ground_truth_img.append(bbox)

            best_iou = 0
            #IoU를 계산하고 임계값과 비교하여 TP, FP 분류
            for idx, gt in enumerate(ground_truth_img):
                iou = calculate_IoU(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        #cumsum은 누적 합
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        #recall이 0일때 Precision은 1인 경우를 추가하여 그래프를 보강
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        #trapz는 사다리꼴 규칙을 사용하여 주어진 데이터 포인트들에 대한 수치적분을 수행
        average_precisions.append(torch.trapz(precisions, recalls))
    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
    plt.show()


def convert_cellboxes(predictions, S=7):
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)

    #두 바운딩박스 추출
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    #그냥 예시... 갑자기 머리 과부화
    # bboxes1 = torch.tensor([
    #     [  # 첫 번째 (유일한) 이미지(3*3그리드)
    #         [  # 첫 번째 행
    #             [0.2, 0.3, 0.5, 0.6],  # (0,0) 그리드 셀의 바운딩 박스
    #             [0.1, 0.2, 0.3, 0.4],  # (0,1) 그리드 셀의 바운딩 박스
    #             [0.4, 0.5, 0.2, 0.3]  # (0,2) 그리드 셀의 바운딩 박스
    #         ],
    #         [  # 두 번째 행
    #             [0.3, 0.4, 0.4, 0.5],  # (1,0) 그리드 셀의 바운딩 박스
    #             [0.5, 0.6, 0.3, 0.2],  # (1,1) 그리드 셀의 바운딩 박스
    #             [0.2, 0.1, 0.4, 0.6]  # (1,2) 그리드 셀의 바운딩 박스
    #         ],
    #         [  # 세 번째 행
    #             [0.1, 0.3, 0.5, 0.4],  # (2,0) 그리드 셀의 바운딩 박스
    #             [0.4, 0.2, 0.3, 0.5],  # (2,1) 그리드 셀의 바운딩 박스
    #             [0.3, 0.5, 0.2, 0.4]  # (2,2) 그리드 셀의 바운딩 박스
    #         ]
    #     ]
    # ])

    #점수 추출 및 합병
    scores = torch.cat((predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0)
    #더 정확도 높은 박스 선택
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2

    #x,y 상대좌표 생성 및 w,h 전체이미지에대한 비율 생성
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_h = 1 / S * best_boxes[..., 2:4]

    #[class, confidence, x, y, w, h] 생성
    converted_bboxes = torch.cat((x, y, w_h), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    #예측 텐서의 첫번째 요소 정수형으로 변환
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def get_bboxes(loader, model, iou_threshold, threshold, pred_format="cells", box_format="midpoint", device="cuda",):
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        #실제, 예측 바운딩 박스를 리스트 형식으로 반환
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        #이미지 마다 처리
        for idx in range(batch_size):
            #중복 박스 삭제
            nms_boxes = nms(bboxes[idx], iou_threshold=iou_threshold, threshold=threshold, box_format=box_format,)

            for nms_box in nms_boxes:
                #이미지 라벨 추가(식별자?)
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                #제거 가능
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def iou_width_height(boxes1, boxes2):
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union