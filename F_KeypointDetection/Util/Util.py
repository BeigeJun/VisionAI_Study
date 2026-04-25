import os
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def generate_heatmap(heatmap_size, pt, sigma=2):
    heatmap = np.zeros((heatmap_size[0], heatmap_size[1]), dtype=np.float32)

    mu_x = int(pt[0] + 0.5)
    mu_y = int(pt[1] + 0.5)

    if mu_x < 0 or mu_y < 0 or mu_x >= heatmap_size[1] or mu_y >= heatmap_size[0]:
        return heatmap

    tmp_size = sigma * 3
    
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

    img_x = max(0, ul[0]), min(br[0], heatmap_size[1])
    img_y = max(0, ul[1]), min(br[1], heatmap_size[0])

    g_x = max(0, -ul[0]), min(br[0], heatmap_size[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], heatmap_size[0]) - ul[1]

    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    g_sliced = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g_sliced
    )
    
    return heatmap


class COCOKeypointDataset(Dataset):
    def __init__(self, root_dir, json_file, input_size=(256, 192), output_size=(64, 48), num_keypoints=17, transform=None):
        self.root_dir = root_dir
        self.input_size = input_size
        self.output_size = output_size
        self.num_keypoints = num_keypoints
        
        self.transform = transform if transform != None else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        self.images = {img['id']: img for img in data['images']}
        self.annotations = [ann for ann in data['annotations'] if ann['num_keypoints'] > 0]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_info = self.images[ann['image_id']]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        image = Image.open(img_path).convert('RGB')
        
        bbox = ann['bbox']
        x, y, w, h = bbox
        image = image.crop((x, y, x + w, y + h))
        image = image.resize((self.input_size[1], self.input_size[0]))
        
        if self.transform:
            image = self.transform(image)
            
        scale_x = self.input_size[1] / w
        scale_y = self.input_size[0] / h
        
        heatmaps = np.zeros((self.num_keypoints, self.output_size[0], self.output_size[1]), dtype=np.float32)
        keypoints = ann['keypoints']
        
        for i in range(self.num_keypoints):
            kx = keypoints[i * 3]
            ky = keypoints[i * 3 + 1]
            v = keypoints[i * 3 + 2]
            
            if v > 0:
                kx_cropped = (kx - x) * scale_x * (self.output_size[1] / self.input_size[1])
                ky_cropped = (ky - y) * scale_y * (self.output_size[0] / self.input_size[0])
                heatmaps[i] = generate_heatmap(self.output_size, (kx_cropped, ky_cropped))
                
        return image, torch.from_numpy(heatmaps)


def keypointdetection_data_loader(str_path, batch_size):
    val_split = 'validation' if os.path.exists(os.path.join(str_path, 'validation')) else 'test'

    str_path_train = str_path + "/coco2017/train2017"
    json_file_train = str_path + "/coco2017/annotations/person_keypoints_train2017.json"
    train_ds = COCOKeypointDataset(str_path_train, json_file=json_file_train)

    str_path_validation = str_path + "/coco2017/val2017"
    json_file_validation = str_path + "/coco2017/annotations/person_keypoints_val2017.json"
    val_ds = COCOKeypointDataset(str_path_validation, json_file=json_file_validation)

    test_ds = COCOKeypointDataset(str_path_validation, json_file=json_file_validation)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def train_model(device, model, train_loader, val_loader, graph, epochs=100, lr=1e-4, patience=10, save_path="checkpoints"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_count = 0
    Best_val_loss = float('inf')
    pbar = tqdm(total=epochs, desc='Total Progress', position=0)

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct, train_total = 0, 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            B, K, H, W = outputs.shape
            pred_peaks = outputs.detach().reshape(B, K, -1).argmax(dim=-1)  # (B, 17)
            gt_peaks   = targets.reshape(B, K, -1).argmax(dim=-1)           # (B, 17)
            train_correct += (pred_peaks == gt_peaks).sum().item()
            train_total   += B * K

        avg_train_loss = train_loss / len(train_loader)
        train_acc = (train_correct / train_total) * 100

        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                B, K, H, W = outputs.shape
                pred_peaks = outputs.reshape(B, K, -1).argmax(dim=-1)
                gt_peaks   = targets.reshape(B, K, -1).argmax(dim=-1)
                val_correct += (pred_peaks == gt_peaks).sum().item()
                val_total   += B * K

        avg_val_loss = val_loss / len(val_loader)
        val_acc = (val_correct / val_total) * 100

        if val_loss < Best_val_loss:
            Best_val_loss = val_loss
            patience_count = 0

        graph.update_graph(train_acc, avg_train_loss, val_acc, avg_val_loss, epoch, patience_count)

        pbar.set_postfix({
            'Epoch': epoch + 1,
            'Train Loss': f'{avg_train_loss:.4f}',
            'Train Acc':  f'{train_acc:.2f}%',
            'Val Acc':    f'{val_acc:.2f}%',
            'Val Loss': f'{val_loss:.4f}',
            'Best Val Loss': f'{Best_val_loss:.8f}'
        })
        pbar.update(1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print("Early stop")
                break


def get_max_preds(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width      = batch_heatmaps.shape[3]

    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = torch.argmax(heatmaps_reshaped, 2)

    preds = torch.zeros((batch_size, num_joints, 2))
    preds[:, :, 0] = idx % width
    preds[:, :, 1] = torch.floor(idx.float() / width)
    return preds


def test_and_visualize_all(model, test_loader, device, model_path="best_hrnet_model.pth"):
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_imgs   = []
    all_preds  = []
    all_labels = []
 
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            preds = get_max_preds(outputs.cpu())   # (B, 17, 2)
 
            for i in range(images.shape[0]):
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                all_imgs.append(img)
                all_preds.append(preds[i])
                all_labels.append(f"Batch {batch_idx} - Img {i}")
 
    total = len(all_imgs)

    for idx in range(total):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(all_imgs[idx])
 
        scale_factor = 4
        for k in range(all_preds[idx].shape[0]):
            x = all_preds[idx][k][0] * scale_factor
            y = all_preds[idx][k][1] * scale_factor
            ax.scatter(x, y, s=15, c='red', marker='o')
 
        ax.axis('off')
        ax.set_title(f"[{idx + 1}/{total}]  {all_labels[idx]}")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
 
        user_input = input(f"[{idx + 1}/{total}] Enter=다음  q=종료 > ").strip().lower()
        plt.close(fig)
 
        if user_input == 'q':
            break


COCO_SIGMAS = np.array([
    .26, .25, .25, .35, .35,
    .79, .79, .72, .72, .62,
    .62, 1.07, 1.07, .87, .87,
    .89, .89
], dtype=np.float64) / 10.0

COCO_VARS = (COCO_SIGMAS * 2) ** 2


def compute_oks_single(pred_kps, gt_kps, gt_vis, gt_area):
    visible = gt_vis > 0
    if visible.sum() == 0:
        return 0.0
    dx = pred_kps[:, 0] - gt_kps[:, 0]
    dy = pred_kps[:, 1] - gt_kps[:, 1]
    d2 = dx ** 2 + dy ** 2
    denominator = np.maximum(2.0 * gt_area * COCO_VARS, np.finfo(np.float64).eps)
    oks_per_kp = np.exp(-d2 / denominator)
    return float((oks_per_kp * visible).sum() / visible.sum())


def _compute_ap_single_threshold(all_predictions, gt_by_image, oks_threshold):
    sorted_preds = sorted(all_predictions, key=lambda x: -x['score'])
    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt == 0:
        return 0.0

    matched_gt = {img_id: [False] * len(gts) for img_id, gts in gt_by_image.items()}
    tp_list, fp_list = [], []

    for pred in sorted_preds:
        img_id  = pred['image_id']
        gt_list = gt_by_image.get(img_id, [])
        if not gt_list:
            tp_list.append(0); fp_list.append(1); continue

        pred_kps  = np.array(pred['keypoints']).reshape(17, 3)[:, :2]
        best_oks, best_gt_j = -1.0, -1

        for j, gt in enumerate(gt_list):
            if matched_gt[img_id][j]: continue
            oks = compute_oks_single(
                pred_kps,
                np.array(gt['keypoints']).reshape(17, 3)[:, :2],
                np.array(gt['keypoints']).reshape(17, 3)[:, 2],
                gt['area']
            )
            if oks > best_oks:
                best_oks, best_gt_j = oks, j

        if best_oks >= oks_threshold:
            matched_gt[img_id][best_gt_j] = True
            tp_list.append(1); fp_list.append(0)
        else:
            tp_list.append(0); fp_list.append(1)

    tp_cum = np.cumsum(tp_list, dtype=np.float64)
    fp_cum = np.cumsum(fp_list, dtype=np.float64)
    recalls    = tp_cum / total_gt
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

    ap = 0.0
    for r_thr in np.linspace(0, 1, 101):
        mask = recalls >= r_thr
        ap += precisions[mask].max() if mask.any() else 0.0
    return float(ap / 101.0)


def _get_pred_area(pred, gt_by_image):
    gts = gt_by_image.get(pred['image_id'], [])
    if not gts: return 0.0
    pred_kps = np.array(pred['keypoints']).reshape(17, 3)[:, :2]
    best_area, best_oks = 0.0, -1.0
    for gt in gts:
        oks = compute_oks_single(
            pred_kps,
            np.array(gt['keypoints']).reshape(17, 3)[:, :2],
            np.array(gt['keypoints']).reshape(17, 3)[:, 2],
            gt['area']
        )
        if oks > best_oks:
            best_oks, best_area = oks, gt['area']
    return best_area


def _compute_mean_recall(predictions, gt_by_image, thresholds):
    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt == 0: return 0.0
    recalls = []
    for thr in thresholds:
        matched_gt = {img_id: [False] * len(gts) for img_id, gts in gt_by_image.items()}
        tp = 0
        for pred in sorted(predictions, key=lambda x: -x['score']):
            img_id  = pred['image_id']
            gt_list = gt_by_image.get(img_id, [])
            if not gt_list: continue
            pred_kps = np.array(pred['keypoints']).reshape(17, 3)[:, :2]
            best_oks, best_gt_j = -1.0, -1
            for j, gt in enumerate(gt_list):
                if matched_gt[img_id][j]: continue
                oks = compute_oks_single(
                    pred_kps,
                    np.array(gt['keypoints']).reshape(17, 3)[:, :2],
                    np.array(gt['keypoints']).reshape(17, 3)[:, 2],
                    gt['area']
                )
                if oks > best_oks:
                    best_oks, best_gt_j = oks, j
            if best_oks >= thr:
                matched_gt[img_id][best_gt_j] = True
                tp += 1
        recalls.append(tp / total_gt)
    return float(np.mean(recalls))


def compute_oks_ap(predictions, gt_json_path):
    if not predictions:
        print("경고: 예측 결과가 비어있습니다.")
        return {}

    with open(gt_json_path, 'r') as f:
        coco_data = json.load(f)

    gt_by_image = {}
    for ann in coco_data['annotations']:
        if ann.get('num_keypoints', 0) == 0: continue
        if 'area' not in ann or ann['area'] == 0:
            _, _, bw, bh = ann['bbox']
            ann['area'] = bw * bh
        gt_by_image.setdefault(ann['image_id'], []).append(ann)

    thresholds = np.arange(0.50, 1.00, 0.05)
    ap_list = [_compute_ap_single_threshold(predictions, gt_by_image, t) for t in thresholds]

    medium_preds = [p for p in predictions if  32**2 < _get_pred_area(p, gt_by_image) <= 96**2]
    large_preds  = [p for p in predictions if _get_pred_area(p, gt_by_image) > 96**2]

    result = {
        'AP':   round(float(np.mean(ap_list)), 4),
        'AP50': round(ap_list[0], 4),
        'AP75': round(ap_list[5], 4),
        'APm':  round(float(np.mean([_compute_ap_single_threshold(medium_preds, gt_by_image, t) for t in thresholds])), 4),
        'APl':  round(float(np.mean([_compute_ap_single_threshold(large_preds,  gt_by_image, t) for t in thresholds])), 4),
        'AR':   round(_compute_mean_recall(predictions, gt_by_image, thresholds), 4),
    }

    print("\n[OKS AP 결과]")
    for k, v in result.items():
        print(f"  {k} = {v:.4f}")

    return result


def heatmap_preds_to_coco_format(preds_coords, metas, output_size, input_size, score=1.0):
    sx = input_size[1] / output_size[1]
    sy = input_size[0] / output_size[0]
    results = []
    for i, meta in enumerate(metas):
        bx, by, bw, bh = meta['bbox']
        kps_x = preds_coords[i, :, 0] * sx * (bw / input_size[1]) + bx
        kps_y = preds_coords[i, :, 1] * sy * (bh / input_size[0]) + by
        kps_flat = []
        for j in range(17):
            kps_flat.extend([float(kps_x[j]), float(kps_y[j]), 2])
        results.append({
            'image_id':    int(meta['image_id']),
            'category_id': 1,
            'keypoints':   kps_flat,
            'score':       float(score),
        })
    return results