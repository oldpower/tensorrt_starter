import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms
import numpy as np
import random

class SiameseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.pairs = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        img1_path = os.path.join(self.root_dir, row['img1'])
        img2_path = os.path.join(self.root_dir, row['img2'])
        # img1 = Image.open(img1_path).convert('RGB')
        # img2 = Image.open(img2_path).convert('RGB')

        img1 = Image.open(img1_path).convert('L').convert('RGB')
        img2 = Image.open(img2_path).convert('L').convert('RGB')

        # 转为 numpy array
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)

        # 合并所有像素
        all_pixels = np.concatenate([arr1.flatten(), arr2.flatten()])
        min_val = all_pixels.min()
        max_val = all_pixels.max()

        # 计算偏移区间
        Da = 255.0 - max_val  # 上界空间
        Db = 0.0 - min_val    # 下界空间 (负数)

        # 随机选择区间 [Db, 0] 或 [0, Da]，并从中采样 delta
        if random.random() < 0.5:
            delta = random.uniform(Db, 0.0)
        else:
            delta = random.uniform(0.0, Da)

        # 应用偏移
        arr1 += delta
        arr2 += delta

        # 由于 delta 的选取保证了范围，以下 clip 可省略
        # arr1 = np.clip(arr1, 0, 255).astype(np.uint8)  # 不再需要
        # arr2 = np.clip(arr2, 0, 255).astype(np.uint8)

        # 直接转回 uint8（因为已保证在 [0,255] 内）
        img1 = Image.fromarray(arr1.astype(np.uint8))
        img2 = Image.fromarray(arr2.astype(np.uint8))

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = row['label']
        return img1, img2, torch.tensor(label, dtype=torch.float)



mytransform = transforms.Compose([
    transforms.Resize((64, 64)),           # 确保是 64x64
    transforms.RandomHorizontalFlip(0.5),
    transforms.Grayscale(num_output_channels=3),   # 转为三通道灰度图（R=G=B）
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 标准化
])
