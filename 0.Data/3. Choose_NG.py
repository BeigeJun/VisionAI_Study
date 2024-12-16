import pandas as pd
import os
import shutil

# CSV 파일 읽기
csv_path = 'D:/open/train_df.csv'
df = pd.read_csv(csv_path)

# 사용자로부터 클래스 이름 입력 받기
class_name = input("복사할 이미지의 클래스 이름을 입력하세요: ")

# 'state'가 'good'이 아닌 행만 필터링하고 입력받은 클래스와 일치하는 행 선택
ng_images = df[(df['state'] != 'good') & (df['class'] == class_name)]

# 소스 디렉토리와 대상 디렉토리 설정
source_dir = 'D:/open/train/train'
target_dir = f'D:/open/{class_name} NG'

# 대상 디렉토리가 없으면 생성
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 필터링된 이미지 복사
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
