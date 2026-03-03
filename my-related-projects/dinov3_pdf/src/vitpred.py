import cv2
import torch
from PIL import Image,ImageOps
from torchvision import transforms
from torchvision.models import vit_b_32
import numpy as np
from tqdm import main

def vit_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model = vit_b_32()
    vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, 3)
    vit_model.load_state_dict(torch.load('../models/vit_b_32_3class.pth'))
    # vit_model.load_state_dict(torch.load('../models/vit_b_32_3class.pth',map_location='cpu'))
    vit_model.to(device)
    vit_model.eval()
    vit_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=3),   # 转为三通道灰度图（R=G=B）
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    input_size = 224
    image = np.random.randint(0, 256, size=(input_size, input_size, 3), dtype=np.uint8)
    image_pil = Image.fromarray(image)
    vit_input = vit_transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        # mvit_output = mvit(mvit_input)
        mvit_output = vit_model(vit_input)
        probabilities = torch.nn.functional.softmax(mvit_output, dim=1)
        prob_values, predicted_classes = torch.max(probabilities, 1)
    class_names = ['molecular','reaction','others']
    # 准备要显示的文本
    label = f"{class_names[predicted_classes.item()]}: {prob_values.item():.2f}"
    print(label)


def load_vit_model(device):
    vit_model = vit_b_32()
    vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, 3)
    vit_model.load_state_dict(torch.load('../models/vit_b_32_3class.pth'))
    # vit_model.load_state_dict(torch.load('../models/vit_b_32_3class.pth',map_location='cpu'))
    vit_model.to(device)
    vit_model.eval()
    vit_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    print(f"ViT Model Load To Device : {device}")
    return vit_model,vit_transform

if __name__ == "__main__":
   vit_prediction() 
