import torch
import torch.nn.functional as F
from torchvision import transforms
import time
from PIL import Image
import os
from MyModel import SiameseNetwork
current_dir = os.path.dirname(os.path.abspath(__file__))

def demo():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = os.path.join(current_dir,'../models/siamese_stirring_model.pth')

    model = SiameseNetwork()
    model.load_state_dict(torch.load(MODEL_PATH)) 
    model.to(DEVICE)
    model.eval()

    transform_inference = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
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
    image1 = os.path.join(current_dir,'../assets/dataset/siamesedata/images/normal_0006.jpg')
    image2 = os.path.join(current_dir,'../assets/dataset/siamesedata/images/normal_0016.jpg')
    for _ in range(5):
        current_time = time.time()
        dist, result = predict_similarity(model,image1,image2, DEVICE)
        print(f"距离: {dist:.4f}, 判断: {result},耗时: {time.time() - current_time:.2f}")


if __name__ == "__main__":
    demo()
