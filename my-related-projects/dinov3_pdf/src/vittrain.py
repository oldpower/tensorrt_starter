import numpy as np
import cv2
import os
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision.models import vit_b_32, ViT_B_32_Weights
from PIL import Image
import time

def initVit():
    weights = ViT_B_32_Weights.DEFAULT
    vit_model = vit_b_32(weights=weights)
    vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, 3)
    return vit_model

def get_transforms(input_size=(224, 224)):
    """获取数据预处理流程"""
    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def trainVit():
    from torchvision import datasets
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR

    vit_model = initVit()
    train_transform,val_transform = get_transforms()

    # 使用与预训练相同的 transform（非常重要！）
    train_dataset = datasets.ImageFolder(root='../assets/dataset/cls3_data_v2/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root='../assets/dataset/cls3_data_v2/val', transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("Trainable parameters:")
    for name, param in vit_model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {tuple(param.shape)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model.to(device)

    num_epochs = 30
    criterion  = nn.CrossEntropyLoss()
    # optimizer  = optim.Adam(vit_model.heads.parameters(), lr=1e-3)  # 只训练分类头（快速）
    # 或者训练整个模型（更慢但可能更好）：
    optimizer = optim.Adam(vit_model.parameters(), lr=1e-5)
    # 学习率调度器
    scheduler  = StepLR(optimizer, step_size=7, gamma=0.1)
    
    print("🚀 start train.")
    vit_model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = vit_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {running_loss/len(train_loader):.4f}, "
              f"Acc: {100.*correct/total:.2f}%")

        scheduler.step()

        # 验证
        vit_model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = vit_model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        print(f"Val Acc: {100.*val_correct/val_total:.2f}%")
        vit_model.train()

        torch.save(vit_model.state_dict(), '../models/vit_b_32_3class.pth')
        print('savemodel ../models/vit_b_32_3class.pth')
    print("Training finished!")


if __name__ == "__main__":
    trainVit()
