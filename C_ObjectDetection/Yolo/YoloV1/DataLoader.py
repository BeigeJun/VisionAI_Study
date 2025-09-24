import torch
from torchvision.datasets import VOCDetection

class YoloV1DataLoader(torch.utils.data.Dataset):
    def __init__(self, root = "D:", year="2012", image_set='train', transform=None, grid_size=7, boxes=2, channel=20):
        self.voc = VOCDetection(root=root, year=year, image_set=image_set, download=True)
        self.transform = transform
        self.grid_size = grid_size
        self.boxes = boxes
        self.channel = channel

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, target = self.voc[idx]
        if self.transform:
            img = self.transform(img)

        # VOC object annotation parsing
        bboxes = []
        for obj in target['annotation']['object']:
            # 객체가 하나일 경우 object가 dict 형식일 수 있음
            if isinstance(target['annotation']['object'], dict):
                objs = [target['annotation']['object']]
            else:
                objs = target['annotation']['object']
            break  # 반복문 첫번 째에서 분리 후 실제 객체 리스트 사용
        else:
            objs = []

        if isinstance(objs, list):
            for obj in objs:
                bbox = obj['bndbox']
                class_name = obj['name']
                # 클래스 인덱스화: 예를 들어 VOC 클래스명이 20개인데, class_name을 인덱스로 변환하는 매핑 필요
                class_idx = self.class_to_idx(class_name)
                xmin = float(bbox['xmin'])
                ymin = float(bbox['ymin'])
                xmax = float(bbox['xmax'])
                ymax = float(bbox['ymax'])

                # 이미지 크기 정보 가져오기
                w = float(target['annotation']['size']['width'])
                h = float(target['annotation']['size']['height'])

                # YOLO 형식: center_x, center_y, width, height (all relative 0~1)
                center_x = ((xmin + xmax) / 2) / w
                center_y = ((ymin + ymax) / 2) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h

                bboxes.append([class_idx, center_x, center_y, bw, bh])
        else:
            # 객체가 한 개일 경우
            obj = objs
            bbox = obj['bndbox']
            class_name = obj['name']
            class_idx = self.class_to_idx(class_name)
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            w = float(target['annotation']['size']['width'])
            h = float(target['annotation']['size']['height'])
            center_x = ((xmin + xmax) / 2) / w
            center_y = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            bboxes.append([class_idx, center_x, center_y, bw, bh])

        boxes = torch.tensor(bboxes)

        label_matrix = torch.zeros((self.grid_size, self.grid_size, self.channel + 5 * self.boxes))

        for box in boxes:
            class_label, center_x, center_y, width, height = box.tolist()
            class_label = int(class_label)

            grid_x, grid_y = int(self.grid_size * center_x), int(self.grid_size * center_y)
            x_cell, y_cell = self.grid_size * center_x - grid_x, self.grid_size * center_y - grid_y
            width_cell, height_cell = width * self.grid_size, height * self.grid_size

            if label_matrix[grid_y, grid_x, 20] == 0:
                label_matrix[grid_y, grid_x, 20] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[grid_y, grid_x, 21:25] = box_coordinates
                label_matrix[grid_y, grid_x, class_label] = 1

        return img, label_matrix

    def class_to_idx(self, class_name):
        # VOC 클래스명 리스트 (20 클래스)
        voc_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        return voc_classes.index(class_name)
# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms
#
#     def __call__(self, img, bboxes):
#         for transform in self.transforms:
#             img, bboxes = transform(img), bboxes
#         return img, bboxes
#

# class YoloV1DataLoader(torch.utils.data.Dataset):
#     def __init__(self, csv_dir, img_dir, label_dir, transform=None, grid_size=7, boxes=2, channel=20):
#         #csv 파일(이미지 경로, 라벨 경로)
#         self.annotations = pd.read_csv(csv_dir)
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         self.transform = transform
#         self.grid_size = grid_size
#         self.boxes = boxes
#         self.channel = channel
#
#     #__는 던더(더블 언더스코어라고 하며 파이썬 내장 함수와 연산자와 연결하여 객체의 기본 동작을 사용자가 정의할수 있게해줌
#     def __len__(self):
#         return len(self.annotations)
#
#     def __getitem__(self, item):
#         #한 이미지와 그 이미지에 있는 바운딩 박스 가져오기
#         label_path = os.path.join(self.label_dir, self.annotations.iloc[item, 1])
#         boxes = []
#         with open(label_path) as f:
#             for label in f.readlines():
#                 class_label, center_x, center_y, width, height = [float(x) if float(x) != int(float(x)) else int(x)
#                                                                  for x in label.replace("\n", "").split()]
#                 boxes.append([class_label, center_x, center_y, width, height])
#         img_path = os.path.join(self.img_dir, self.annotations.iloc[item, 0])
#         image = Image.open(img_path)
#         boxes = torch.tensor(boxes)
#
#         if self.transform:
#             image, boxes = self.transform(image, boxes)
#         #channel=각 라벨에 대한 확률을 넣기 위해서, 5 = class_label, x_cell, y_cell, width_cell, height_cell, boxes=바운딩박스수
#         label_matrix = torch.zeros((self.grid_size, self.grid_size, self.channel+5*self.boxes))
#
#         #박스의 센터, 가로, 세로 정제
#         for box in boxes:
#             class_label, center_x, center_y, width, height = box.tolist()
#             class_label = int(class_label)
#
#             #실제 센터가 존재하는 그리드 찾기
#             grid_x, grid_y = int(self.grid_size * center_x), int(self.grid_size * center_y)
#             #실제 존재하는 그리드에서의 위치 찾기
#             x_cell, y_cell = self.grid_size * center_x - grid_x, self.grid_size * center_y - grid_y
#             #바운딩 박스 크기
#             width_cell, height_cell = width * self.grid_size, height * self.grid_size
#
#             #욜로1은 다중 감지가 안된다. 단, 한 그리드 안에 여러 물체의 중심이 있을때만이다
#             if label_matrix[grid_y, grid_x, 20] == 0:
#                 label_matrix[grid_y, grid_x, 20] = 1
#                 box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
#
#                 label_matrix[grid_y, grid_x, 21:25] = box_coordinates
#                 label_matrix[grid_y, grid_x, class_label] = 1
#
#         return image, label_matrix

