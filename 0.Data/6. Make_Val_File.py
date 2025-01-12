import os
import csv
import shutil


val_dir = 'D:/ImageNet/ImageNet/ILSVRC/Data/CLS-LOC/val'
csv_file = 'D:/ImageNet/ImageNet/LOC_val_solution.csv'
splited_val_dir = 'D:/ImageNet/ImageNet/ILSVRC/Data/CLS-LOC/Splited_val'

os.makedirs(splited_val_dir, exist_ok=True)

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    data = list(reader)

for row in data:
    image_id = row[0]
    class_id = row[1].split()[0]

    class_dir = os.path.join(splited_val_dir, class_id)
    os.makedirs(class_dir, exist_ok=True)

    src_path = os.path.join(val_dir, f"{image_id}.JPEG")
    dest_path = os.path.join(class_dir, f"{image_id}.JPEG")

    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)
