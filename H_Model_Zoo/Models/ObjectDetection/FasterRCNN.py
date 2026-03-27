#출처 : https://kubig-2022-2.tistory.com/79
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import time
import cv2
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET

def xml_parser(xml_path):
    xml = open(xml_path, "r")
    tree = ET.parse(xml)
    root = tree.getroot()
    size = root.find('size')
    file_name = root.find('filename').text
    object_name = []
    bbox = []
    objects = root.findall('object')
    for _object in objects:
        name = _object.find('name').text
        object_name.append(name)
        bndbox = _object.find('bndbox')
        one_bbox = []
        xmin = bndbox.find("xmin").text
        one_bbox.append(int(float(xmin)))
        ymin = bndbox.find("ymin").text
        one_bbox.append(int(float(ymin)))
        xmax = bndbox.find("xmax").text
        one_bbox.append(int(float(xmax)))
        ymax = bndbox.find("ymax").text
        one_bbox.append(int(float(ymax)))
        bbox.append(one_bbox)
    return file_name, object_name, bbox

def makeBox(voc_im, bbox, objects):
    image = voc_im.copy()
    for i in range(len(objects)):
        cv2.rectangle(image, (int(bbox[i][0]), int(bbox[i][1])), (int(bbox[i][2]), int(bbox[i][3])), color=(0, 255, 0), thickness=1)
        cv2.putText(image, objects[i], (int(bbox[i][0]), int(bbox[i][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image

xml_list = os.listdir("D:/Image_Data/MaskorNonMask/annotations/")
xml_list.sort()

label_set = set()

for i in range(len(xml_list)):
    xml_path = 'D:/Image_Data/MaskorNonMask/annotations/' + str(xml_list[i])
    file_name, object_name, bbox = xml_parser(xml_path)
    for name in object_name:
        label_set.add(name)

label_set = sorted(list(label_set))

label_dic = {}
for i, key in enumerate(label_set):
    label_dic[key] = (i + 1)

class Pascal_Vo(Dataset):
    def __init__(self, xml_list, len_data):
        self.xml_list = xml_list
        self.len_data = len_data
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((600, 600))

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        xml_path = 'D:/Image_Data/MaskorNonMask/annotations/' + str(xml_list[idx])
        file_name, object_name, bbox = xml_parser(xml_path)
        image_path = 'D:/Image_Data/MaskorNonMask/images/' + str(file_name)
        image = Image.open(image_path).convert('RGB')
        image = self.resize(image)
        image = self.to_tensor(image)

        targets = []
        d = {}
        d['boxes'] = torch.tensor(bbox)
        d['labels'] = torch.tensor([label_dic[x] for x in object_name], dtype=torch.int64)
        targets.append(d)

        return image, targets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = torchvision.models.vgg16(pretrained=True).features[:-1]
backbone_out = 512
backbone.out_channels = backbone_out

anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

resolution = 7
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=resolution, sampling_ratio=2)

box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(in_channels=backbone_out * (resolution ** 2), representation_size=4096)
box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(4096, len(label_set) + 1)

model = torchvision.models.detection.FasterRCNN(backbone, num_classes=None,
                                               min_size=600, max_size=1000,
                                               rpn_anchor_generator=anchor_generator,
                                               rpn_pre_nms_top_n_train=6000, rpn_pre_nms_top_n_test=6000,
                                               rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
                                               rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                                               rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                                               box_roi_pool=roi_pooler, box_head=box_head, box_predictor=box_predictor,
                                               box_score_thresh=0.05, box_nms_thresh=0.7, box_detections_per_img=300,
                                               box_fg_iou_thresh=0.5, box_be_iou_thresh=0.5,
                                               box_batch_size_per_image=128, box_positive_fraction=0.25)

model.to(device)

for param in model.rpn.parameters():
    torch.nn.init.normal_(param, mean=0.0, std=0.01)

for name, param in model.roi_heads.named_parameters():
    if "bbox_pred" in name:
        torch.nn.init.normal_(param, mean=0.0, std=0.001)
    elif "weight" in name:
        torch.nn.init.normal_(param, mean=0.0, std=0.01)
    if "bias" in name:
        torch.nn.init.zeros_(param)

def Total_loss(loss):
    loss_objectness = loss['loss_objectness']
    loss_rpn_box_reg = loss['loss_rpn_box_reg']
    loss_classifier = loss['loss_classifier']
    loss_box_reg = loss['loss_box_reg']

    rpn_total = loss_objectness + 10 * loss_rpn_box_reg
    fast_rcnn_total = loss_classifier + 1 * loss_box_reg

    total_loss = rpn_total + fast_rcnn_total

    return total_loss

total_epoch = 15
len_data = 25
loss_sum = 0

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch, eta_min=0.00001)

start_epoch = 0
start_idx = 0

print("start_epoch = {} , start_idx = {}".format(start_epoch, start_idx))

print("Training Start")
model.train()
start = time.time()

for epoch in range(start_epoch, total_epoch):
    dataset = Pascal_Vo(xml_list[:len_data], len_data - start_idx)
    dataloader = DataLoader(dataset, shuffle=True)

    for i, (image, targets) in enumerate(dataloader, start_idx):
        optimizer.zero_grad()

        image = image.to(device)
        targets[0]['boxes'] = targets[0]['boxes'].to(device)
        targets[0]['labels'] = targets[0]['labels'].to(device)

        targets[0]['boxes'].squeeze_(0)
        targets[0]['labels'].squeeze_(0)

        loss = model(image, targets)
        total_loss = Total_loss(loss)
        loss_sum += total_loss

        total_loss.backward()
        optimizer.step()

    start_idx = 0
    scheduler.step()
    state = {
        'epoch': epoch,
        'iter': i + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    print('epoch:' + str(epoch))
    print('loss:' + str(total_loss))

model.eval()
image, targets = next(iter(DataLoader(Pascal_Vo(xml_list[:len_data], len_data), batch_size=1)))
image = image.to(device)
targets[0]['boxes'] = targets[0]['boxes'].to(device)
targets[0]['labels'] = targets[0]['labels'].to(device)

preds = model(image)

boxes = preds[0]['boxes']
labels = preds[0]['labels']
objects = []
for lb in labels:
    objects.append([k for k, v in label_dic.items() if v == lb][0])

plot_image = image.squeeze().permute(1, 2, 0).cpu().numpy()
answer = makeBox(plot_image, boxes.cpu().numpy(), objects)

plt.imshow(answer)
plt.show()
