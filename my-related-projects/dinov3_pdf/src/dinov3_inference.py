import torch
import torch.nn as nn
import timm
from PIL import Image
import time
from safetensors.torch import load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model() :
    # 加载模型，并设置 num_classes=0 以获取特征维度
    backbone = timm.create_model(
        model_name  = 'timm/vit_small_patch16_dinov3.lvd1689m',
        pretrained  = False,
        num_classes = 0)
    weight_path = "../models/model.safetensors" 
    state_dict = load_file(weight_path)
    msg = backbone.load_state_dict(state_dict, strict=False)
    print(f"权重加载状态: {msg}")

    data_config = timm.data.resolve_model_data_config(backbone)
    backbone.eval()
    # 加载分类头
    checkpoint = torch.load('../models/dino_classifier_head.pth', map_location=device)
    classifier = nn.Sequential(
        nn.Linear(checkpoint['feature_dim'], 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(256, checkpoint['num_classes'])
    )
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    # 组合模型
    model = nn.Sequential(backbone, classifier)
    model.to(device)
    model.eval()
    return model,data_config


def load_model_custom_device(custom_device) :
    # 加载模型，并设置 num_classes=0 以获取特征维度
    backbone = timm.create_model(
        model_name  = 'timm/vit_small_patch16_dinov3.lvd1689m',
        pretrained  = False,
        num_classes = 0)
    weight_path = "../models/model.safetensors" 
    state_dict = load_file(weight_path)
    msg = backbone.load_state_dict(state_dict, strict=False)
    print(f"权重加载状态: {msg}")

    data_config = timm.data.resolve_model_data_config(backbone)
    backbone.eval()
    # 加载分类头
    checkpoint = torch.load('../models/dino_classifier_head.pth', map_location=custom_device)
    classifier = nn.Sequential(
        nn.Linear(checkpoint['feature_dim'], 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(256, checkpoint['num_classes'])
    )
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    # 组合模型
    model = nn.Sequential(backbone, classifier)
    model.to(custom_device)
    model.eval()
    return model,data_config

def load_model_custom_device_v2(custom_device) :
    # 加载模型，并设置 num_classes=0 以获取特征维度
    backbone = timm.create_model(
        model_name  = 'timm/vit_base_patch16_dinov3.lvd1689m',
        pretrained  = False,
        num_classes = 0)
    weight_path = "../models/model_b.safetensors" 
    state_dict = load_file(weight_path)
    msg = backbone.load_state_dict(state_dict, strict=False)
    print(f"权重加载状态: {msg}")

    data_config = timm.data.resolve_model_data_config(backbone)
    backbone.eval()
    # 加载分类头
    checkpoint = torch.load('../models/dino_classifier_head_v2_b.pth', map_location=custom_device)
    classifier = nn.Sequential(
        nn.Linear(checkpoint['feature_dim'], 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(256, checkpoint['num_classes'])
    )
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    # 组合模型
    model = nn.Sequential(backbone, classifier)
    model.to(custom_device)
    model.eval()
    print(f"DINO Model Load To Device : {device}")
    return model,data_config


def process_image(data_config,file_path) :
    # 预处理和分类
    transform = timm.data.create_transform(**data_config, is_training=False)
    image = Image.open(file_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor

def classifier(model,input_tensor,file_path) :
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        print([round(x, 4) for x in prob[0].tolist()])
        confidence, predicted = torch.max(prob, 1)
    class_names = ['molecular','reaction','others']
    end_time = time.time()
    print(f"图片：{file_path} 预测: {class_names[predicted.item()]} 置信度: {confidence.item():.4f} 执行时间: {end_time - start_time:.4f} 秒")

if __name__ == '__main__':
    # model,data_config = load_model()
    model,data_config = load_model_custom_device_v2(device)
    input_tensor = process_image(data_config,'../assets/class_image/0/10.1073_pnas.95.9.4810.pdf_00.jpg')
    classifier(model,input_tensor,'test0.png')
    input_tensor = process_image(data_config,'../assets/class_image/1/SI-001-00001.png')
    classifier(model,input_tensor,'test1.png')
    input_tensor = process_image(data_config,'../assets/class_image_v2/2/10.31299_hrri.52.1.7.pdf_01.jpg')
    classifier(model,input_tensor,'test2.png')
