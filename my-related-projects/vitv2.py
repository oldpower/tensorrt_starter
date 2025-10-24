import torch
import torch.nn as nn
from torchvision.models import vit_b_32, ViT_B_32_Weights
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import time
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# 自定义 2D 位置编码模块
class PositionEmbedding2D(nn.Module):
    def __init__(self, grid_size=(7, 7), embed_dim=768):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.embed_dim = embed_dim
        
        # 分别为行和列创建可学习嵌入
        self.row_embed = nn.Embedding(grid_size[0], embed_dim // 2)
        self.col_embed = nn.Embedding(grid_size[1], embed_dim // 2)
        self.cls_token_pos = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # 初始化
        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)

    def forward(self):
        device = self.row_embed.weight.device
        row_idx = torch.arange(self.grid_h, device=device)
        col_idx = torch.arange(self.grid_w, device=device)
        
        row_emb = self.row_embed(row_idx)  # [7, D//2]
        col_emb = self.col_embed(col_idx)  # [7, D//2]

        # 构造 7x7 网格的所有位置编码
        pos_emb_2d = torch.cat([
            row_emb.unsqueeze(1).expand(-1, self.grid_w, -1),  # [7,7,D//2]
            col_emb.unsqueeze(0).expand(self.grid_h, -1, -1),  # [7,7,D//2]
        ], dim=-1).view(1, self.grid_h * self.grid_w, self.embed_dim)  # [1, 49, D]

        cls_pos = self.cls_token_pos  # [1, 1, D]
        return torch.cat([cls_pos, pos_emb_2d], dim=1)  # [1, 50, D]

class CustomViT(nn.Module):
    def __init__(self, pretrained_vit, grid_size=(7,7)):
        super().__init__()
        self.vit = pretrained_vit
        
        # 保存原始 patch size 和 hidden dim
        self.embed_dim = pretrained_vit.hidden_dim
        
        # 替换位置编码为 2D 版本
        self.pos_embedding = PositionEmbedding2D(grid_size, self.embed_dim)
        
        # 可以冻结主干（可选）冻结 ViT 主干网络（backbone）的所有参数，使其在训练过程中不被更新,包括PositionEmbedding2D
        # for param in self.vit.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        # 使用 Vit 的 patch embedding
        x = self.vit._process_input(x)  # [B, 49, D]
        
        n = x.shape[0]  # batch size
        # 添加 CLS token
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # ❌ 不再使用 self.vit.encoder.pos_embedding
        # ✅ 改用我们自己的 2D 位置编码
        pos_embed = self.pos_embedding()  # [1, 50, D]
        x = x + pos_embed
        
        x = self.vit.encoder.dropout(x)
        x = self.vit.encoder.layers(x)
        x = self.vit.encoder.ln(x)
        return self.vit.heads(x[:, 0])  # 取 CLS token 输出


def initVitv2():
    weights = ViT_B_32_Weights.DEFAULT
    base_vit = vit_b_32(weights=weights)
    # base_vit.eval()  # 冻结 BN 等

    # 包装成自定义模型
    model = CustomViT(base_vit, grid_size=(7, 7))

    vit_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    return model, vit_transform

def trainVit():
    from torchvision import datasets
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR

    vit_model,vit_transform = initVitv2()

    # 使用与预训练相同的 transform（非常重要！）
    train_dataset = datasets.ImageFolder(root='./assets/dataset/vitdataset/train', transform=vit_transform)
    val_dataset = datasets.ImageFolder(root='./assets/dataset/vitdataset/val', transform=vit_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_class  = 2
    num_epochs = 2

    vit_model.vit.heads.head = torch.nn.Linear(vit_model.vit.heads.head.in_features, num_class)
    # vit_model.heads = torch.nn.Linear(vit_model.heads.head.in_features, num_class)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model.to(device)

    criterion  = nn.CrossEntropyLoss()
    # 将位置编码参数和分类头参数一起加入优化器
    optimizer = optim.Adam(
        list(vit_model.pos_embedding.parameters()) + 
        list(vit_model.vit.heads.parameters()),
        lr=1e-3
    )
    # 或者训练整个模型（更慢但可能更好）：
    # optimizer = optim.Adam(vit_model.parameters(), lr=1e-5)
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

        torch.save(vit_model.state_dict(), './models/vitv2_b_32_2class.pth')
        print('savemodel ./models/vitv2_b_32_2class.pth')
    print("Training finished!")

def predictVit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    num_class = 2

    vit_model,vit_transform = initVitv2()
    # vit_model.heads.head = nn.Linear(768, 2)  # 重新定义
    vit_model.vit.heads.head = torch.nn.Linear(vit_model.vit.heads.head.in_features, num_class)
    vit_model.load_state_dict(torch.load('./models/vitv2_b_32_2class.pth'))
    vit_model.to(device)
    vit_model.eval()
    
    pil_image = Image.open(os.path.join(current_dir,'./assets/StirringLL/20250814-StirringLiquidLiquid_frame_0000.png'))
    pil_image = Image.open(os.path.join(current_dir,'./assets/StirringSL/20250814-StirringSolidLiquid_frame_0000.png'))
    # 3. 应用ViT预处理
    input_tensor = vit_transform(pil_image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    # input_batch = torch.cat([input_batch,input_batch],dim=0)
    with torch.no_grad():
        for _ in range(5):
            starttime = time.time()
            output = vit_model(input_batch)
            print(f"⏰torch推理耗时: {time.time() - starttime:.4f}")
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            print("原始输出：", output)
            # print("概率分布：", probabilities)
            # 获取类别及其对应的概率
            prob_values, predicted_classes = torch.max(probabilities, 1)
            # print(f"预测类别: {predicted_classes.item()}, 概率: {prob_values.item()}")

def export_norm_onnx():
    import onnx
    import onnxsim

    file    = os.path.join(current_dir,'models/vitv2_b_32_2class.onnx')
    input   = torch.rand(1, 3, 224, 224, device='cuda')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_class = 2
    vit_model,_ = initVitv2()
    vit_model.vit.heads.head = torch.nn.Linear(vit_model.vit.heads.head.in_features, num_class)
    vit_model.load_state_dict(torch.load('models/vitv2_b_32_2class.pth'))
    vit_model.to(device)

    vit_model.eval()
    torch.onnx.export(
        model         = vit_model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        dynamic_axes  = {"input0" :{0:"batch"},
                         "output0":{0:"batch"}},
        opset_version = 15)
    print("Finished normal onnx export")

    model_onnx = onnx.load(file)

    # 检查导入的onnx model
    onnx.checker.check_model(model_onnx)

    # 使用onnx-simplifier来进行onnx的简化。
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, file)

def inference_onnx():
    import onnxruntime as ort
    import onnx

    # 加载 ONNX 模型（可选，用于查看信息）
    model_onnx = onnx.load('./models/vitv2_b_32_2class.onnx')
    print("模型输入名:", model_onnx.graph.input[0].name)
    print("模型输出名:", model_onnx.graph.output[0].name)

    # 创建 ONNX Runtime 推理会话
    session = ort.InferenceSession('./models/vit_b_32_2class.onnx', providers=['CPUExecutionProvider'])
    # 如果有 GPU，可以使用:
    # session = ort.InferenceSession('./models/vit_b_32_2class.onnx', providers=['CUDAExecutionProvider'])

    # 获取输入信息
    input_name = session.get_inputs()[0].name
    print("输入节点名:", input_name)

    # 加载并预处理图片
    # image_path = './assets/StirringLL/20250814-StirringLiquidLiquid_frame_0000.png'
    # pil_image = Image.open(image_path).convert('RGB')
    pil_image = Image.open(os.path.join(current_dir,'./assets/StirringLL/20250814-StirringLiquidLiquid_frame_0000.png'))
    pil_image = Image.open(os.path.join(current_dir,'./assets/StirringSL/20250814-StirringSolidLiquid_frame_0000.png'))

    # 注意：ONNX 模型的输入是经过预处理的张量，必须和训练时一致！
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),  # ViT-B/32 使用 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(pil_image).unsqueeze(0)  # 添加 batch 维度
    # input_tensor = torch.cat([input_tensor, input_tensor], dim=0)
    input_numpy = input_tensor.numpy()  # 转为 numpy array
    print(input_numpy.shape)
   
    for _ in range(5):
        starttime = time.time()
        # 推理
        outputs = session.run(None, {input_name: input_numpy})  # None 表示返回所有输出
        print(f"⏰onnx推理耗时: {time.time() - starttime:.4f}")

        # 获取输出
        logits = outputs[0]  # shape: [1, num_classes]
        print("原始输出:", logits)
        # 转为概率（Softmax）
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
        # print("概率分布:", probabilities)
        # 获取预测类别
        predicted_class = probabilities.argmax()
        # print(f"预测类别: {predicted_class}, 置信度: {probabilities[predicted_class]:.4f}")

def inference_trt():
    from z_trt import inference_TRTInfer
    inference_TRTInfer()


def vitDemo():
    vit_model, vit_transform = initVitv2()
    print(f"vit 位置编码形状: {vit_model.pos_embedding().shape}")  # 应输出 [1, 50, 768]
    print(vit_model.vit.heads.head)

if __name__ == "__main__":
    # trainVit() 
    # export_norm_onnx()
    predictVit()
    # inference_onnx()
    # inference_trt()

