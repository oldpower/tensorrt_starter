import os
import random
import shutil
from pathlib import Path

def create_symlink_split(src_root, dst_root, split_ratio=0.8, seed=42):
    random.seed(seed)
    
    src_root = Path(src_root)
    dst_train = Path(dst_root) / "train"
    dst_val = Path(dst_root) / "val"
    
    # 创建目标目录
    for split in [dst_train, dst_val]:
        for cls_dir in src_root.iterdir():
            if cls_dir.is_dir():
                (split / cls_dir.name).mkdir(parents=True, exist_ok=True)

    # 遍历每个类别
    for cls_dir in src_root.iterdir():
        if not cls_dir.is_dir():
            continue
        all_files = [f for f in cls_dir.iterdir() if f.is_file()]
        random.shuffle(all_files)
        
        split_idx = int(len(all_files) * split_ratio)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        # 创建软链接
        for f in train_files:
            (dst_train / cls_dir.name / f.name).symlink_to(f.resolve())
        for f in val_files:
            (dst_val / cls_dir.name / f.name).symlink_to(f.resolve())
    
    print(f"✅ 软链接划分完成！训练集: {dst_train}, 验证集: {dst_val}")

if __name__ == "__main__":
    create_symlink_split(
        src_root="../assets/class_image_v2",
        dst_root="../assets/dataset/cls3_data_v2",
        split_ratio=0.85,
        seed = 66
    )
