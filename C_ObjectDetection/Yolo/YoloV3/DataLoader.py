import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from G_Model_Zoo.Models.ObjectDetection.Util.Utils import iou_width_height


class YoloV3DataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir,  label_dir, anchors, image_size, S, C, scale, transform='train'):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

        self.train_transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=int(image_size * scale)),
                A.PadIfNeeded(
                    min_height=int(image_size * scale),
                    min_width=int(image_size * scale),
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.RandomCrop(width=image_size, height=image_size),
                A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
                A.OneOf(
                    [
                        A.ShiftScaleRotate(
                            rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                        ),
                        A.Affine(shear=15, p=0.5, mode="constant"),
                    ],
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Blur(p=0.1),
                A.CLAHE(p=0.1),
                A.Posterize(p=0.1),
                A.ToGray(p=0.1),
                A.ChannelShuffle(p=0.05),
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[], ),
        )

        self.test_transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(
                    min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
                ),
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
        )

        if transform == "train":
            self.transform = self.train_transforms
        elif transform == "test":
            self.transform = self.test_transforms



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


