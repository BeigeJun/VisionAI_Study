import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import matplotlib.pyplot as plt


LABEL_MAP = { 'good': 0, 'broken': 1, 'contamination': 2}
CLASS_NAMES = [ 'Good', 'Broken', 'Contamination']
CLASS_COLORS = [ (0, 0, 255), (0, 255, 0), (255, 0, 0)] # BGR

class MVTecMultiDataset(Dataset):
    def __init__(self, root, split='train', input_size=(1024, 1024)):
        self.root = Path(root)
        self.split = split
        self.input_size = input_size
        self.image_paths = sorted(glob.glob(str(self.root / split / '**' / '*.png'), recursive=True))

        self.img_transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=Image.NEAREST),
        ])

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = Path(self.image_paths[idx])
        img = Image.open(img_path).convert('RGB')
        img_t = self.img_transform(img)

        anomaly_type = img_path.parent.name
        label_idx = LABEL_MAP.get(anomaly_type, 0)

        if label_idx == 0:
            mask = np.zeros(self.input_size, dtype=np.int64)
        else:
            mask_path = self.root / 'ground_truth' / anomaly_type / f'{img_path.stem}_mask.png'
            if mask_path.exists():
                mask = Image.open(mask_path).convert('L')
                mask = self.mask_transform(mask)
                mask = (np.array(mask) > 128).astype(np.int64) * label_idx
            else:
                mask = np.zeros(self.input_size, dtype=np.int64)

        return img_t, torch.from_numpy(mask)

def segmentation_data_loader(str_path, batch_size, input_size_dl):
    val_split = 'validation' if os.path.exists(os.path.join(str_path, 'validation')) else 'test'
    
    train_ds = MVTecMultiDataset(str_path, split='train', input_size=(input_size_dl, input_size_dl))
    val_ds = MVTecMultiDataset(str_path, split=val_split, input_size=(input_size_dl, input_size_dl))
    test_ds = MVTecMultiDataset(str_path, split='test', input_size=(input_size_dl, input_size_dl))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

def train_model(device, model, train_loader, val_loader, graph, epochs=100, lr=1e-4, patience=10, save_path="checkpoints"):
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    patience_count = 0
    Best_val_loss = float('inf')
    pbar = tqdm(total=epochs, desc='Total Progress', position=0)

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device).long()
            optimizer.zero_grad()
            outputs = model(imgs)
            if hasattr(model, 'deep_supervision') and model.deep_supervision:
                loss = sum(criterion(out, masks) for out in outputs) / len(outputs)
                final_out = outputs[-1]
            else:
                loss = criterion(outputs, masks)
                final_out = outputs
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (final_out.argmax(1) == masks).sum().item()
            train_total += masks.numel()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = (train_correct / train_total) * 100

        model.eval()
        val_loss, val_correct, val_total_pixels = 0.0, 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device).long()
                outputs = model(imgs)
                if hasattr(model, 'deep_supervision') and model.deep_supervision:
                    v_loss = sum(criterion(out, masks) for out in outputs) / len(outputs)
                    final_v_out = outputs[-1]
                else:
                    v_loss = criterion(outputs, masks)
                    final_v_out = outputs
                val_loss += v_loss.item()
                val_correct += (final_v_out.argmax(1) == masks).sum().item()
                val_total_pixels += masks.numel()

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

def draw_results(img_np, pred_mask):
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = img_cv.copy()
    for cls_id in range(1, len(CLASS_NAMES)):
        cls_mask = (pred_mask == cls_id).astype(np.uint8)
        color = CLASS_COLORS[cls_id]
        overlay[cls_mask > 0] = color
        contours, _ = cv2.findContours(cls_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 10: continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, 1)
            cv2.putText(img_cv, CLASS_NAMES[cls_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return cv2.addWeighted(overlay, 0.4, img_cv, 0.6, 0)

def visualize_random_10(model, device, load_path, input_size=1024):
    model.eval()
    val_ds = MVTecMultiDataset(load_path, split='test', input_size=(input_size, input_size))
    indices = np.random.choice(len(val_ds), 10, replace=False)
    inv_norm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    
    for idx in indices:
        img_t, _ = val_ds[idx]
        with torch.no_grad():
            pred = model(img_t.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        
        img_vis = (inv_norm(img_t).permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        res_img = draw_results(img_vis, pred)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1); plt.imshow(img_vis); plt.title(f"Sample : Original"); plt.axis('off')
        plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)); plt.title(f"Sample : Pred"); plt.axis('off')
        
        plt.tight_layout()
        plt.show(block=True)


def visualize_all_test_set(model, device, load_path, input_size=1024):
    model.eval()
    val_ds = MVTecMultiDataset(load_path, split='test', input_size=(input_size, input_size))
    inv_norm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    
    plt.ion() 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    for idx in range(len(val_ds)):
        img_t, _ = val_ds[idx]
        with torch.no_grad():
            output = model(img_t.unsqueeze(0).to(device))
            pred = output.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        
        img_vis = (inv_norm(img_t).permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        res_img = draw_results(img_vis, pred)

        ax1.clear(); ax1.imshow(img_vis); ax1.set_title(f"[{idx+1}/{len(val_ds)}] Original"); ax1.axis('off')
        ax2.clear(); ax2.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)); ax2.set_title("Prediction"); ax2.axis('off')
        
        plt.draw()
        if not plt.waitforbuttonpress():
            break

    plt.ioff()
    plt.close()