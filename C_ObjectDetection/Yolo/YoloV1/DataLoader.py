import os
from PIL import Image
import torch
import pandas as pd

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

