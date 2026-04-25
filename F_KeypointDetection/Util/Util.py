import os
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
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
        self.transform = transform
        
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

    # str_path = str_path + ""
    # json_file = str_path + ""

    test_ds = COCOKeypointDataset(str_path_validation, json_file=json_file_validation)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

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
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = (train_correct / train_total) * 100

        model.eval()
        val_loss, val_correct, val_total_pixels = 0.0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = (val_correct / val_total_pixels) * 100

        if val_loss < Best_val_loss:
            Best_val_loss = val_loss
            patience_count = 0

        graph.update_graph(train_acc, avg_train_loss, val_acc, avg_val_loss, epoch, patience_count)

        pbar.set_postfix({
            'Epoch': epoch + 1,
            'Train Loss': f'{train_loss:.4f}',
            'Val Acc': f'{val_acc:.2f}%',
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
    width = batch_heatmaps.shape[3]
    
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = torch.argmax(heatmaps_reshaped, 2)
    
    preds = torch.zeros((batch_size, num_joints, 2))
    preds[:, :, 0] = idx % width
    preds[:, :, 1] = torch.floor(idx.float() / width)
    return preds

def test_and_visualize_all(model, test_loader, device, model_path="best_hrnet_model.pth"):
    print("\n--- 모든 테스트 데이터 시각화 시작 ---")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            preds = get_max_preds(outputs.cpu())
            
            batch_size = images.shape[0]
            cols = 4
            rows = (batch_size + cols - 1) // cols
            
            fig = plt.figure(figsize=(15, rows * 4))
            
            for i in range(batch_size):
                ax = fig.add_subplot(rows, cols, i + 1)
                
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                ax.imshow(img)
                
                scale_factor = 4
                for k in range(preds.shape[1]):
                    x, y = preds[i][k][0] * scale_factor, preds[i][k][1] * scale_factor
                    ax.scatter(x, y, s=5, c='red')
                
                ax.axis('off')
                ax.set_title(f"Batch {batch_idx} - Img {i}")

            plt.tight_layout()
            plt.show()
