# def dinov3_modelscope_weight():
#     import torch
#     from modelscope import AutoImageProcessor, AutoModel  # 确保是从modelscope导入
#     from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
#     from PIL import Image
#     import requests

#     # 假设这是ModelScope上的一个有效的DINOv3模型ID
#     pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"

#     # 创建图像处理器，这里我们手动定义transform以符合DINOv3的需求
#     transform = Compose([
#         Resize(224),
#         CenterCrop(224),
#         ToTensor(),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#     input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

#     # 加载预训练模型
#     processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
#     model = AutoModel.from_pretrained(pretrained_model_name)
#     state_dict = model.state_dict()
#     torch.save(state_dict, "dinov3_vitb16_backbone.pth")

#     class DINOv3Classifier(torch.nn.Module):
#         def __init__(self, backbone, num_classes=1000):  # 根据实际情况调整类别数
#             super(DINOv3Classifier, self).__init__()
#             self.backbone = backbone
#             self.classifier = torch.nn.Linear(768, num_classes)  # 假设输出维度是768

#         def forward(self, x):
#             with torch.no_grad():
#                 features = self.backbone(x)
#             cls_token = features['x_norm_clstoken'] if 'x_norm_clstoken' in features else features.pooler_output
#             return self.classifier(cls_token)

#     classifier = DINOv3Classifier(model, num_classes=10)  # 调整类别数

#     # 测试推理
#     with torch.no_grad():
#         logits = classifier(input_tensor)
#         pred = logits.argmax(-1)
#         print(logits)
#         print("Predicted class:", pred.item())


# def load_dinov3_torch():
#     import torch
#     import torch.nn as nn
#     # 已有本地 dinov3 代码（含 hubconf.py）
#     backbone = torch.hub.load(
#         "../assets/dinov3",
#         "dinov3_vitb16",
#         source="local",
#         pretrained=True
#     )

#     # 加载保存的权重
#     state_dict = torch.load("dinov3_vitb16_backbone.pth", map_location="cpu")
#     backbone.load_state_dict(state_dict, strict=True)

#     # 构建纯 torch 分类器
#     model = nn.Sequential(
#         backbone,
#         nn.Linear(768, 10)
#     )
#     
#     from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
#     from PIL import Image
#     import requests
#     transform = Compose([
#         Resize(224),
#         CenterCrop(224),
#         ToTensor(),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#     input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
#     with torch.no_grad():
#         logits = model(input_tensor)
#         pred = logits.argmax(-1)
#         print(logits)
#         print("Predicted class:", pred.item())

#     print("==================================")




# def load_dinov3_backbone(checkpoint_path: str):
#     import torch
#     import timm
#     # 1. 创建 timm ViT-B/16（与 DINOv3 结构一致）
#     model = timm.create_model(
#         "vit_base_patch16_224",
#         pretrained=False,
#         num_classes=0,          # 不要分类头
#         global_pool="token",    # 使用 cls token
#         dynamic_img_size=True,
#     )

#     # 2. 安全加载权重（weights_only=True）
#     state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

#     # 3. 清理 key（移除 backbone. 前缀）
#     clean_state_dict = {}
#     for k, v in state_dict.items():
#         if k.startswith("backbone."):
#             k = k[9:]
#         clean_state_dict[k] = v

#     # 4. 加载（现在应该 strict=True 也能成功）
#     model.load_state_dict(clean_state_dict, strict=True)  # ✅ 无警告

#     return model.eval()

# # load_dinov3_backbone("dinov3_vitb16_backbone.pth")

# def hf_mirror():
#     import os
#     os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#     import torch
#     from transformers import AutoImageProcessor, AutoModel
#     from transformers.image_utils import load_image
#     url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     image = load_image(url)

#     pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
#     processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
#     model = AutoModel.from_pretrained(
#         pretrained_model_name, 
#         device_map="auto", 
#     )

#     inputs = processor(images=image, return_tensors="pt").to(model.device)
#     with torch.inference_mode():
#         outputs = model(**inputs)

#     pooled_output = outputs.pooler_output
#     print("Pooled output shape:", pooled_output.shape)

import torch.nn as nn
import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from safetensors.torch import load_file
from torchvision.datasets import ImageFolder
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS  = 20
BATCH_SIZE  = 16
NUM_CLASSES = 3
DATASET_DIR = '../assets/dataset/cls3_data_v2'

def found_dinov3_model_name():
    """
    Found DINOv3 models:
        timm/convnext_base.dinov3_lvd1689m
        timm/convnext_large.dinov3_lvd1689m
        timm/convnext_small.dinov3_lvd1689m
        timm/convnext_tiny.dinov3_lvd1689m
        timm/vit_base_patch16_dinov3.lvd1689m
        timm/vit_base_patch16_dinov3_qkvb.lvd1689m
        timm/vit_huge_plus_patch16_dinov3.lvd1689m
        timm/vit_huge_plus_patch16_dinov3_qkvb.lvd1689m
        timm/vit_large_patch16_dinov3.lvd1689m
        timm/vit_large_patch16_dinov3.sat493m
        timm/vit_large_patch16_dinov3_qkvb.lvd1689m
        timm/vit_large_patch16_dinov3_qkvb.sat493m
        timm/vit_small_patch16_dinov3.lvd1689m
        timm/vit_small_patch16_dinov3_qkvb.lvd1689m
        timm/vit_small_plus_patch16_dinov3.lvd1689m
        timm/vit_small_plus_patch16_dinov3_qkvb.lvd1689m
        timm/vit_7b_patch16_dinov3.lvd1689m
        timm/vit_7b_patch16_dinov3.sat493m
    """
    from huggingface_hub import list_models
    # 获取所有由 'timm' 上传的、包含 'dinov3' 的模型
    dinov3_models = [
        model.id 
        for model in list_models(author="timm") 
        if "dinov3" in model.id.lower()
    ]

    print("Found DINOv3 models:")
    for name in dinov3_models:
        print(name)

def load_model():
    # 加载模型，并设置 num_classes=0 以获取特征维度
    model = timm.create_model(
        model_name  = 'timm/vit_base_patch16_dinov3.lvd1689m',
        pretrained  = False,
        num_classes = 0)
    # weight_path = "../models/model.safetensors" 
    weight_path = "../models/model_b.safetensors" 
    state_dict = load_file(weight_path)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"权重加载状态: {msg}")
    model.eval()  # 设置为评估模式
    # 冻结模型的所有参数
    for param in model.parameters():
        param.requires_grad = False
    print("模型装载完成")
    return model


def def_model(model):
    # 获取DINOv3输出的特征维度
    feature_dim = model.num_features
    # 假设你的数据集有10个类别
    num_classes = NUM_CLASSES
    # 定义一个简单的分类头
    classifier = nn.Sequential(
        nn.Linear(feature_dim, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(256, num_classes)
    )
    # 将骨干网络和分类头组合成完整模型
    class CustomDINOv3(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            features = self.backbone(x)  # 形状为 [batch_size, feature_dim]
            output = self.head(features)
            return output
    custom_model = CustomDINOv3(model, classifier)
    print("定义训练模型完成")
    return custom_model, num_classes, feature_dim


def load_dataset(model):
    # 获取模型对应的数据预处理配置（注意：DINOv3 默认 input_size 是 (3, 518, 518)）
    data_config = timm.data.resolve_model_data_config(model)
    
    # 注意：is_training=False 用于验证集，True 用于训练集（含增强）
    train_transform = timm.data.create_transform(**data_config, is_training=True)
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    # 使用 ImageFolder 加载自定义数据集
    train_dataset = ImageFolder(root=f'{DATASET_DIR}/train', transform=train_transform)
    test_dataset = ImageFolder(root=f'{DATASET_DIR}/val', transform=val_transform)

    # 检查类别数是否为3
    assert len(train_dataset.classes) == 3, f"Expected 3 classes, got {len(train_dataset.classes)}"
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"装载自定义数据集完成，类别: {train_dataset.classes}")
    return train_loader, test_loader, data_config


def train_model(custom_model, train_loader):
    custom_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(custom_model.head.parameters(), lr=0.001)
    num_epochs = NUM_EPOCHS

    for epoch in range(num_epochs):
        custom_model.train()
        running_loss = 0.0
        
        # 使用 tqdm 包装 train_loader，显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = custom_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 实时更新进度条后缀（显示当前 batch 的 loss）
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}')
    
    print('✅ 模型训练完成')

# def train_model(custom_model, train_loader):
#     custom_model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     # 只对分类头的参数进行优化
#     optimizer = optim.Adam(custom_model.head.parameters(), lr=0.001)
#     num_epochs = 10
#     for epoch in range(num_epochs):
#         custom_model.train()  # 设置模型为训练模式
#         running_loss = 0.0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = custom_model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print(
#             f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
#     print('模型训练完成')


def save_model(custom_model, num_classes, feature_dim, data_config):
    # 只保存分类头的权重（文件更小，更灵活）
    torch.save({
        'classifier_state_dict': custom_model.head.state_dict(),
        'num_classes': num_classes,
        'feature_dim': feature_dim,
        'training_config': {
            'model_name': 'facebook/dinov3-base-pretrain-lvd1689m',
            'input_size': data_config['input_size']
        }
    }, '../models/dino_classifier_head_v2_b.pth')
    print("模型权重保存完成")


def eval_mode(custom_model, test_loader):
    correct = 0
    total = 0
    custom_model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = custom_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'测试集准确率: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    model = load_model()
    custom_model, num_classes, feature_dim = def_model(model)
    train_loader, test_loader, data_config = load_dataset(model)
    train_model(custom_model, train_loader)
    save_model(custom_model, num_classes, feature_dim, data_config)
    eval_mode(custom_model, test_loader)

    # model = timm.create_model(
    #     model_name  = 'timm/vit_base_patch16_dinov3.lvd1689m',
    #     pretrained  = True,
    #     num_classes = 0)
    
