import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 64x64 -> 64x64
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 64x64 -> 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32x32 -> 32x32
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32x32 -> 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 16x16 -> 16x16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))                  # 强制输出 4x4
        )
        self.fc = nn.Linear(128 * 4 * 4, 128)  # 嵌入向量维度

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, img1, img2):
        feat1 = self.forward_once(img1)
        feat2 = self.forward_once(img2)
        # return F.cosine_similarity(feat1, feat2)  # 输出相似度 [-1, 1]
        # 或返回 (feat1, feat2) 用于计算 L2 距离
        return feat1,feat2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, dist, label):
        # dist: L2 距离, label: 0=相似, 1=不相似
        loss = (1 - label) * torch.pow(dist, 2) + \
               label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        return loss.mean()
