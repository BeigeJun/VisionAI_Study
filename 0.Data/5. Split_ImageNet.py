import os
import csv
import shutil
import random


def copy_subset_imagenet(src_root, dst_root, csv_file, images_per_class=10):
    train_src = os.path.join(src_root, 'train')
    train_dst = os.path.join(dst_root, 'train')

    if not os.path.exists(train_dst):
        os.makedirs(train_dst)

    for class_folder in os.listdir(train_src):
        src_class_path = os.path.join(train_src, class_folder)
        dst_class_path = os.path.join(train_dst, class_folder)

        if not os.path.exists(dst_class_path):
            os.makedirs(dst_class_path)

        all_images = [f for f in os.listdir(src_class_path) if f.endswith(('.JPEG', '.jpg', '.png'))]
        selected_images = random.sample(all_images, min(images_per_class, len(all_images)))

        for image in selected_images:
            src_image_path = os.path.join(src_class_path, image)
            dst_image_path = os.path.join(dst_class_path, image)
            shutil.copy2(src_image_path, dst_image_path)

    print("Finished processing train dataset")

    val_src = os.path.join(src_root, 'val')
    val_dst = os.path.join(dst_root, 'val')

    if not os.path.exists(val_dst):
        os.makedirs(val_dst)

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        data = list(reader)

    class_images = {}
    for row in data:
        image_id = row[0]
        class_id = row[1].split()[0]

        if class_id not in class_images:
            class_images[class_id] = []
        class_images[class_id].append(image_id)

    for class_id, images in class_images.items():
        dst_class_path = os.path.join(val_dst, class_id)
        if not os.path.exists(dst_class_path):
            os.makedirs(dst_class_path)

        selected_images = random.sample(images, min(images_per_class, len(images)))

        for image_id in selected_images:
            src_image_path = os.path.join(val_src, f"{image_id}.JPEG")
            dst_image_path = os.path.join(dst_class_path, f"{image_id}.JPEG")

            if os.path.exists(src_image_path):
                shutil.copy2(src_image_path, dst_image_path)
            else:
                print(f"경고: 이미지 {image_id}.JPEG를 찾을 수 없습니다")

    print("Finished processing validation dataset")


src_root = 'D:/ImageNet/ImageNet/ILSVRC/Data/CLS-LOC'
dst_root = 'D:/Splited_ImageNet'
csv_file = 'D:/ImageNet/ImageNet/LOC_val_solution.csv'

copy_subset_imagenet(src_root, dst_root, csv_file)
