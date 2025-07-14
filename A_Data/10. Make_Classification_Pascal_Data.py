import os
import shutil

voc_dir = 'D:/Pascal/VOC2012'
images_dir = os.path.join(voc_dir, 'JPEGImages')
image_sets_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
output_dir = 'D:/Pascal/Classification_Data'

for txt_file in os.listdir(image_sets_dir):
    if txt_file.endswith('.txt'):
        class_name = txt_file.replace('_trainval', '').replace('_train', '').replace('_val', '').replace('.txt', '')
        class_folder = os.path.join(output_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)
        txt_path = os.path.join(image_sets_dir, txt_file)
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_id, label = parts
                    if label == '1':
                        image_file = image_id + '.jpg'
                        src = os.path.join(images_dir, image_file)
                        dst = os.path.join(class_folder, image_file)
                        if os.path.exists(src):
                            shutil.copy(src, dst)
