import os
import shutil
from pathlib import Path
from tqdm import tqdm

def merge_to_new_dataset(src_root, dst_root, target_label="broken", source_labels=["broken_large", "broken_small"]):
    src_base = Path(src_root)
    dst_base = Path(dst_root)

    splits = ['train', 'test', 'validation', 'ground_truth']
    
    for split in splits:
        src_split_path = src_base / split
        if not src_split_path.exists():
            continue
            
        dst_split_path = dst_base / split
        os.makedirs(dst_split_path, exist_ok=True)
        
        print(f"\n[Processing {split}]")
        
        sub_dirs = [d for d in os.listdir(src_split_path) if os.path.isdir(src_split_path / d)]
        
        for sub_dir in sub_dirs:
            src_sub_path = src_split_path / sub_dir
            
            if sub_dir in source_labels:
                target_path = dst_split_path / target_label # 새로운 'broken' 폴더로
                os.makedirs(target_path, exist_ok=True)
                
                for file_name in tqdm(os.listdir(src_sub_path), desc=f"Merging {sub_dir}"):
                    new_name = f"{sub_dir}_{file_name}"
                    shutil.copy2(src_sub_path / file_name, target_path / new_name)
            
            else:
                target_path = dst_split_path / sub_dir
                if not target_path.exists():
                    shutil.copytree(src_sub_path, target_path)
                    print(f"Copied {sub_dir} as is.")

if __name__ == "__main__":
    original_path = r"D:\1. DataSet\bottle"
    new_path = r"D:\1. DataSet\bottle_merged"
    
    merge_to_new_dataset(original_path, new_path)
    print(f"\n✅ 작업 완료! 새로운 데이터셋 경로: {new_path}")