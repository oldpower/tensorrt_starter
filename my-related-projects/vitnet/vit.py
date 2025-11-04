import numpy as np
import cv2
import os
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision.models import vit_b_32, ViT_B_32_Weights
from PIL import Image
import time

current_dir = os.path.dirname(os.path.abspath(__file__))

def initVit():
    # 加载预训练模型和权重
    weights = ViT_B_32_Weights.DEFAULT
    vit_model = vit_b_32(weights=weights)
    vit_model.eval()
    # 定义ViT预处理变换
    vit_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 确保尺寸正确
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return vit_model,vit_transform

def vitDemo():
    # LL_dir = os.path.join(current_dir,'assets/StirringLL_cut')
    # LL_Image_List = sorted(os.listdir(LL_dir))

    LS_dir = os.path.join(current_dir,'assets/StirringSL_cut')
    LS_Image_List = sorted(os.listdir(LS_dir))

    vit_model,vit_transform = initVit()
    
    # ROI 参数
    # x, y, w, h = 700, 750, 100, 550
    x, y, w, h = 740, 1200, 70, 100
    rectangle_current = None
    rectangle_hisory = None
    process = True
    patches = []  # 存放每个 32x32 的 patch
    for image_name in LS_Image_List:
        image_path = os.path.join(LS_dir,image_name)

        frame = cv2.imread(image_path)

        # cv2.rectangle(img       = img,
        #               rec       = (x, y, w, h),
        #               color     = (0,255,0),
        #               thickness = 2)
        if rectangle_hisory is None:
            rectangle_hisory = frame[y:y+h,x:x+w]
            roi = frame[y:y+32, x:x+32]  # 提取 32x32 区域
            continue

        rectangle_current = frame[y:y+h,x:x+w]
        roi = frame[y:y+32, x:x+32]  # 提取 32x32 区域
        patches.append(roi)
        if len(patches) == 7*7:
            patches = np.array(patches)  # 形状变为 (49, 32, 32, 3)
            grid = patches.reshape(7, 7, 32, 32, 3)
            # 交换轴，准备拼接: (7, 32, 7, 32, 3)
            grid = np.transpose(grid, (0, 2, 1, 3, 4))
            # 合并空间维度
            # 先在行方向拼接（每行7个 patch 横向拼）
            # rows = [np.concatenate(grid[i], axis=1) for i in range(7)]  # 每行: (32, 224, 3)
            rows = [np.concatenate([grid[i, :, j] for j in range(7)], axis=1) for i in range(7)]
            print(len(rows),rows[0].shape)
            print(grid.shape)
            print(grid[0, :, 0].shape)
            # 再把7行纵向拼接
            concatenated = np.concatenate(rows, axis=0)  # 最终: (224, 224, 3)
            print(concatenated.shape)

            
            cv2.imwrite('./assets/vitconcatenated.jpg',concatenated) 
            # 准备输入ViT模型
            # 1. 将BGR转换为RGB
            concatenated_rgb = cv2.cvtColor(concatenated, cv2.COLOR_BGR2RGB)
            # 2. 转换为PIL图像
            pil_image = Image.fromarray(concatenated_rgb)
            # 3. 应用ViT预处理
            input_tensor = vit_transform(pil_image)
            # 4. 添加batch维度 [1, 3, 224, 224]
            input_batch = input_tensor.unsqueeze(0)
            print(f"ViT输入张量形状: {input_batch.shape}")
            # 现在可以将 input_batch 输入到ViT模型中
            with torch.no_grad():
                output = vit_model(input_batch)
                print(output.shape)
                print(output[0][:10])

            
            patches = []
            break


        if process:
            pass

def createDataset():
    import random
    from pathlib import Path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    random.seed(10)
    
    # 定义源目录和目标目录
    source_A = os.path.join(current_dir, 'assets', 'StirringLL')
    source_B = os.path.join(current_dir, 'assets', 'StirringSL')
    target_dataset = os.path.join(current_dir, 'assets', 'dataset', 'vitdataset')
    
    # 类别名称（用于目标文件夹命名）
    class_A_name = 'StirringLL'
    class_B_name = 'StirringSL'
    
    # 创建目标目录结构
    train_A = os.path.join(target_dataset, 'train', class_A_name)
    train_B = os.path.join(target_dataset, 'train', class_B_name)
    val_A = os.path.join(target_dataset, 'val', class_A_name)
    val_B = os.path.join(target_dataset, 'val', class_B_name)
    
    for dir_path in [train_A, train_B, val_A, val_B]:
        os.makedirs(dir_path, exist_ok=True)
    
    def process_class(source_dir, class_name):
        if not os.path.exists(source_dir):
            print(f"源目录不存在: {source_dir}")
            return
        
        # 获取所有图片文件（可根据需要扩展后缀）
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        all_files = [
            f for f in os.listdir(source_dir)
            if os.path.isfile(os.path.join(source_dir, f)) and
            Path(f).suffix.lower() in extensions
        ]
        
        if not all_files:
            print(f"在 {source_dir} 中未找到图片文件")
            return
        
        # 随机打乱
        random.shuffle(all_files)
        
        # 按 8:2 划分
        split_idx = int(0.8 * len(all_files))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        # 为训练集创建软链接
        for fname in train_files:
            src = os.path.join(source_dir, fname)
            dst = os.path.join(train_A if class_name == 'StirringLL' else train_B, fname)
            create_symlink(src, dst)
        
        # 为验证集创建软链接
        for fname in val_files:
            src = os.path.join(source_dir, fname)
            dst = os.path.join(val_A if class_name == 'StirringLL' else val_B, fname)
            create_symlink(src, dst)
        
        print(f"{class_name}: {len(train_files)} 训练, {len(val_files)} 验证")
    
    def create_symlink(src, dst):
        """创建软链接，若目标已存在则删除"""
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        try:
            os.symlink(src, dst)
            # print(f"链接: {dst} -> {src}")
        except Exception as e:
            print(f"创建软链接失败 {dst} -> {src}: {e}")
            # 备用：如果软链接失败（如Windows权限问题），可改为复制
            # shutil.copy2(src, dst)
    
    # 处理两个类别
    process_class(source_A, class_A_name)
    process_class(source_B, class_B_name)

    print("数据集划分与软链接创建完成！")
    print(f"数据集路径: {target_dataset}")

def trainVit():
    from torchvision import datasets
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR

    vit_model,vit_transform = initVit()

    # 使用与预训练相同的 transform（非常重要！）
    train_dataset = datasets.ImageFolder(root='./assets/dataset/vitdataset/train', transform=vit_transform)
    val_dataset = datasets.ImageFolder(root='./assets/dataset/vitdataset/val', transform=vit_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_class  = 2
    num_epochs = 2

    vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, num_class)
    # vit_model.heads = torch.nn.Linear(vit_model.heads.head.in_features, num_class)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model.to(device)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(vit_model.heads.parameters(), lr=1e-3)  # 只训练分类头（快速）
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

        torch.save(vit_model.state_dict(), './models/vit_b_32_2class.pth')
        print('savemodel ./models/vit_b_32_2class.pth')
    print("Training finished!")

def predictVit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_class = 2

    vit_model,vit_transform = initVit()
    # vit_model.heads.head = nn.Linear(768, 2)  # 重新定义
    vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, num_class)
    vit_model.load_state_dict(torch.load('./models/vit_b_32_2class.pth'))
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
            print("概率分布：", probabilities)
            # 获取类别及其对应的概率
            prob_values, predicted_classes = torch.max(probabilities, 1)
            print(f"预测类别: {predicted_classes.item()}, 概率: {prob_values.item()}")

def export_norm_onnx():
    import onnx
    import onnxsim

    file    = os.path.join(current_dir,'models/vit_b_32_2class.onnx')
    input   = torch.rand(1, 3, 224, 224, device='cuda')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    num_class = 2
    vit_model,_ = initVit()
    # vit_model.heads.head = nn.Linear(768, 2)  # 重新定义
    vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, num_class)
    vit_model.load_state_dict(torch.load('models/vit_b_32_2class.pth'))
    vit_model.to(device)
    # vit_model = torch.jit.script(vit_model,input) 
    # vit_model = torch.jit.trace(vit_model,input) 
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
    model_onnx = onnx.load('./models/vit_b_32_2class.onnx')
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
        # 推理
        starttime = time.time()
        outputs = session.run(None, {input_name: input_numpy})  # None 表示返回所有输出
        print(f"⏰onnx推理耗时: {time.time() - starttime:.4f}")

        # 获取输出
        logits = outputs[0]  # shape: [1, num_classes]
        # print("原始输出:", logits)
        # 转为概率（Softmax）
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
        # print("概率分布:", probabilities)
        # 获取预测类别
        predicted_class = probabilities.argmax()
        # print(f"预测类别: {predicted_class}, 置信度: {probabilities[predicted_class]:.4f}")

def inference_trt():
    from z_trt import inference_TRTInfer
    inference_TRTInfer()
if __name__ == "__main__":
    # vitDemo()
    # createDataset()
    # trainVit() 
    # predictVit()
    # export_norm_onnx()
    inference_onnx()
    inference_trt()




    
