import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import os
from tqdm import tqdm  # 美化进度条
import matplotlib.pyplot as plt

from MyDataset import SiameseDataset,mytransform
from MyModel import SiameseNetwork,ContrastiveLoss
current_dir = os.path.dirname(os.path.abspath(__file__))

def train_epoch(model, dataloader, criterion, optimizer, device,margin):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_pairs = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for img1, img2, labels in progress_bar:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        # 前向传播
        feat1, feat2 = model(img1, img2)  # 返回两个特征向量
        dist = F.pairwise_distance(feat1, feat2)  # L2 距离

        # 计算损失
        loss = criterion(dist, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img1.size(0)

        # 计算准确率（简单阈值判断）
        pred = (dist > margin).float()  # 距离 < margin 判为相似
        correct_predictions += (pred == labels).sum().item()
        total_pairs += labels.size(0)

        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / total_pairs
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device, margin):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_pairs = 0

    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader, desc="Validating", leave=False):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            feat1, feat2 = model(img1, img2)
            dist = F.pairwise_distance(feat1, feat2)
            loss = criterion(dist, labels)

            running_loss += loss.item() * img1.size(0)

            pred = (dist > margin).float()
            correct_predictions += (pred == labels).sum().item()
            total_pairs += labels.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / total_pairs
    return epoch_loss, epoch_acc


def main():
    # 超参数
    BATCH_SIZE      = 32
    EPOCHS          = 20
    LEARNING_RATE   = 1e-4
    DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS     = 4 if DEVICE == 'cuda' else 0
    MODEL_SAVE_PATH = os.path.join(current_dir,'../models/siamese_stirring_model.pth')
    IMAGES_DIR      = os.path.join(current_dir,"../assets/dataset/siamesedata/images")
    MARGIN          = 1.0

    # 实例化数据集
    train_csv_dir   = os.path.join(current_dir,"../assets/dataset/siamesedata/train_pairs.csv")               # 输出目录
    val_csv_dir   = os.path.join(current_dir,"../assets/dataset/siamesedata/val_pairs.csv")               # 输出目录
    train_dataset = SiameseDataset(csv_file=train_csv_dir, root_dir=IMAGES_DIR, transform=mytransform)
    val_dataset = SiameseDataset(csv_file=val_csv_dir, root_dir=IMAGES_DIR, transform=mytransform)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    # 模型
    model = SiameseNetwork().to(DEVICE)
    # 损失函数
    criterion = ContrastiveLoss(margin=MARGIN)  # margin 可调
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # 学习率调度器（可选）
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # 每 20 轮学习率 ×0.5

    # 用于记录
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, MARGIN)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE, MARGIN)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 学习率调度
        scheduler.step()

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ 模型已保存至 {MODEL_SAVE_PATH}")

        # 打印日志
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(current_dir,'../assets/siamesenet_train_info.jpg'))
    plt.show()


def demo():
    import time
    from PIL import Image
    from torchvision import transforms
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = os.path.join(current_dir,'../models/siamese_stirring_model.pth')

    model = SiameseNetwork()
    model.load_state_dict(torch.load(MODEL_PATH)) 
    model.to(DEVICE)
    model.eval()

    transform_inference = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Grayscale(num_output_channels=3),   # 转为三通道灰度图（R=G=B）
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def predict_similarity(model, img_path1, img_path2, device):
        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')

        img1 = transform_inference(img1).unsqueeze(0).to(device)
        img2 = transform_inference(img2).unsqueeze(0).to(device)

        with torch.no_grad():
            feat1, feat2 = model(img1, img2)
            dist = F.pairwise_distance(feat1, feat2).item()

        similarity = 1 - dist  # 简单映射到 [0,1]
        return dist, "相似" if dist < 1.0 else "不相似"

    # 使用示例
    for _ in range(5):
        current_time = time.time()
        dist, result = predict_similarity(model, 
                                          '../assets/dataset/siamesedata/images/abnormal_0001.jpg',
                                          '../assets/dataset/siamesedata/images/abnormal_0100.jpg',
                                          DEVICE)


        print(f"距离: {dist:.4f}, 判断: {result},耗时: {time.time() - current_time:.2f}")


if __name__ == "__main__":
    main()
    # demo()
