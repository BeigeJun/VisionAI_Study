import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import ImageFolder
from G_Model_Zoo.Models.ObjectDetection.Util.Utils import iou_width_height

#-----------------------------------------------classification----------------------------------------------------------


def Classification_data_loader(str_path, batch_size, info):
    transform_info = info

    train_dataset = ImageFolder(root=str_path + "//train", transform=transform_info)
    validation_dataset = ImageFolder(root=str_path + "//validation", transform=transform_info)
    test_dataset = ImageFolder(root=str_path + "//test", transform=transform_info)

    #num_workers는 데이터를 불러올 때 사용할 프로세스 수. 기본값은 0이고 커질수록 데이터를 불러오는 속도가 빨라짐.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    return train_loader, validation_loader, test_loader

#-----------------------------------------------objectdetection---------------------------------------------------------


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for transform in self.transforms:
            img, bboxes = transform(img), bboxes
        return img, bboxes


class YoloV1DataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_dir, img_dir, label_dir, transform=None, grid_size=7, boxes=2, channel=20):
        #csv 파일(이미지 경로, 라벨 경로)
        self.annotations = pd.read_csv(csv_dir)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.grid_size = grid_size
        self.boxes = boxes
        self.channel = channel

    #__는 던더(더블 언더스코어라고 하며 파이썬 내장 함수와 연산자와 연결하여 객체의 기본 동작을 사용자가 정의할수 있게해줌
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        #한 이미지와 그 이미지에 있는 바운딩 박스 가져오기
        label_path = os.path.join(self.label_dir, self.annotations.iloc[item, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, center_x, center_y, width, height = [float(x) if float(x) != int(float(x)) else int(x)
                                                                 for x in label.replace("\n", "").split()]
                boxes.append([class_label, center_x, center_y, width, height])
        img_path = os.path.join(self.img_dir, self.annotations.iloc[item, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)
        #channel=각 라벨에 대한 확률을 넣기 위해서, 5 = class_label, x_cell, y_cell, width_cell, height_cell, boxes=바운딩박스수
        label_matrix = torch.zeros((self.grid_size, self.grid_size, self.channel+5*self.boxes))

        #박스의 센터, 가로, 세로 정제
        for box in boxes:
            class_label, center_x, center_y, width, height = box.tolist()
            class_label = int(class_label)

            #실제 센터가 존재하는 그리드 찾기
            grid_x, grid_y = int(self.grid_size * center_x), int(self.grid_size * center_y)
            #실제 존재하는 그리드에서의 위치 찾기
            x_cell, y_cell = self.grid_size * center_x - grid_x, self.grid_size * center_y - grid_y
            #바운딩 박스 크기
            width_cell, height_cell = width * self.grid_size, height * self.grid_size

            #욜로1은 다중 감지가 안된다. 단, 한 그리드 안에 여러 물체의 중심이 있을때만이다
            if label_matrix[grid_y, grid_x, 20] == 0:
                label_matrix[grid_y, grid_x, 20] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                label_matrix[grid_y, grid_x, 21:25] = box_coordinates
                label_matrix[grid_y, grid_x, class_label] = 1

        return image, label_matrix


#-------------------------------------------YoloV3----------------------------------------------------------------------

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]


class YoloV3DataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir,  label_dir, anchors=ANCHORS, image_size=416, S=[13, 26, 52], C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = (width * S, height * S)
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)

