import pandas as pd
import os
import shutil

csv_path = 'D:/Anomaly/Original/train_df.csv'
df = pd.read_csv(csv_path)

class_name = input("복사할 이미지의 클래스 이름을 입력하세요: ")

ng_images = df[(df['state'] != 'good') & (df['class'] == class_name)]

source_dir = 'D:/Anomaly/Original/train/train'
target_dir = f'D:/Anomaly/NG/{class_name}_NG'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for _, row in ng_images.iterrows():
    file_name = row['file_name']
    source_path = os.path.join(source_dir, file_name)
    target_path = os.path.join(target_dir, file_name)

    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)
        print(f"Copied: {file_name}")
    else:
        print(f"File not found: {file_name}")

print(f"Image copying process completed for class: {class_name} with state not good.")
print(f"Total images copied: {len(ng_images)}")
