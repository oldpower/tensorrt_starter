import cv2
from ultralytics import YOLO
import os
from typing import Callable
import numpy as np
from skimage.metrics import structural_similarity as ssim
from utils import StateBuffer
import time
import torch
from torchvision import transforms
from siamesenet.MyModel import SiameseNetwork
import torch.nn.functional as F
from PIL import Image
import onnxruntime as ort
import onnx
from collections import deque

current_dir = os.path.dirname(os.path.abspath(__file__))
buffer = StateBuffer(size = 10)


def cut_image(img):
    top_ratio = 0.00   # 上边距比例
    bottom_ratio = 0.00  # 下边距比例
    left_ratio = 0.4  # 左边距比例
    right_ratio = 0.3  # 右边距比例

    h, w = img.shape[:2]
    # 计算中间保留区域
    top_margin = int(h * top_ratio)
    bottom_margin = int(h * bottom_ratio)
    left_margin = int(w * left_ratio)
    right_margin = int(w * right_ratio)
    
    center_h = h - top_margin - bottom_margin
    center_w = w - left_margin - right_margin
        
    x1 = left_margin
    y1 = top_margin
    x2 = x1 + center_w
    y2 = y1 + center_h

    return img[y1:y2, x1:x2],x1,y1,x2,y2


def getAreaListA(x1, y1, w, h, ssimboxnum=10):
    half_w = w // 2
    # half_h = h // 2
    ssim_box_x1 = int(x1 + half_w + half_w * 0.4)
    ssim_box_y1 = int(y1 + h * 0.05)
    ssim_box_x2 = int(x1 + half_w + half_w * 0.6)
    ssim_box_y2 = int(y1 + h * 0.95)
    ssim_box_w  = ssim_box_x2 - ssim_box_x1
    ssim_box_h  = ssim_box_y2 - ssim_box_y1
    # print(f"ssim box x1 y1 x2 y2: {ssim_box_x1} {ssim_box_y1} {ssim_box_x2} {ssim_box_y2}")
    # print(f"ssim box w h: {ssim_box_w} {ssim_box_h}")
    area_list = []
    area_h = ssim_box_h // ssimboxnum
    for i in range(ssimboxnum):
        area_y1 = ssim_box_y1 + area_h * i
        area_y2 = ssim_box_y1 + area_h * (i + 1)
        area_list.append([ssim_box_x1, area_y1, ssim_box_x2, area_y2])
    return area_list


def getAreaListB(x1, y1, w, h, ssimboxnum=2):
    half_w = w // 2
    # half_h = h // 2
    ssim_box_x1 = int(x1 + half_w + half_w * 0.4)
    ssim_box_y1 = int(y1 + h * 0.15)
    ssim_box_x2 = int(x1 + half_w + half_w * 0.6)
    ssim_box_y2 = int(y1 + h * 0.85)
    ssim_box_w  = ssim_box_x2 - ssim_box_x1
    ssim_box_h  = ssim_box_y2 - ssim_box_y1
    # print(f"ssim box x1 y1 x2 y2: {ssim_box_x1} {ssim_box_y1} {ssim_box_x2} {ssim_box_y2}")
    # print(f"ssim box w h: {ssim_box_w} {ssim_box_h}")
    area_list = []
    area_w = ssim_box_w // ssimboxnum
    area_h = ssim_box_h // ssimboxnum
    for i in range(ssimboxnum):
        for j in range(ssimboxnum):
            area_x1 = ssim_box_x1 + area_w * j 
            area_y1 = ssim_box_y1 + area_h * i
            area_x2 = ssim_box_x1 + area_w * (j + 1)
            area_y2 = ssim_box_y1 + area_h * (i + 1)
            area_list.append([area_x1, area_y1, area_x2, area_y2])
    return area_list


def differenceOfSpatialv2(input):

    colors = {}
    np.random.seed(20)
    for i in range(5):
        colors[i] = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    font_color = (255,255,255)

    # 0: Liquid&Liquid
    # 1: Solid&Liquid
    results = model(input,verbose = False, iou = 0.45, conf = 0.25)
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model.names

    max_conf = 0.0
    class_idx = None
    boxes_idx = None
    i = 0
    for idx,conf in zip(class_indices,confidences):
        idx = int(idx)
        if conf > max_conf:
            max_conf = conf
            class_idx = idx
            boxes_idx = i
        i += 1

    if class_idx is None:
        buffer.clear()
        patch_0.clear()
        patch_1.clear()
        patch_2.clear()
        patch_3.clear()
    elif class_idx == 0:
        class_name = class_names[class_idx]
        # print(f"检测到: {class_name}, 置信度: {max_conf:.2f}, 坐标: {boxes.xyxy[boxes_idx]}")
        x1, y1, x2, y2 = boxes.xyxy[boxes_idx].floor().int().tolist()
        _ , _ ,  w,  h = boxes.xywh[boxes_idx].floor().int().tolist()

        ############
        #   SSIM   #
        ############
        # area_list = getAreaListA(x1, y1, w, h, ssimboxnum=6)
        area_list = getAreaListB(x1, y1, w, h, ssimboxnum=2)
        # print(area_list)
      
        # 方式1: SSIM 计算
        # ssim_rst = False
        # area_benchmark = input[area_list[0][1]:area_list[0][3],area_list[0][0]:area_list[0][2]].copy()
        # gray_benchmark = cv2.cvtColor(area_benchmark, cv2.COLOR_BGR2GRAY)
        # for area_x1,area_y1,area_x2,area_y2 in area_list[1:]:
        #     area_current = input[area_y1:area_y2,area_x1:area_x2].copy()
        #     gray_current = cv2.cvtColor(area_current, cv2.COLOR_BGR2GRAY)
        #     score, _ = ssim(gray_benchmark, gray_current, 
        #                        full=True,
        #                        # data_range=gray_current.max() - gray_current.min()
        #                        )
        #     if score < 0.8:
        #         buffer.add(False)
        #         break
        #     else:
        #         buffer.add(True)

        # 方式2: SiameseNet
        input_rgb = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        area_benchmark = input_rgb[area_list[0][1]:area_list[0][3],area_list[0][0]:area_list[0][2]].copy()
        area_benchmark = Image.fromarray(area_benchmark)
        img1 = transform_inference(area_benchmark).unsqueeze(0).to(DEVICE)
        for area_x1,area_y1,area_x2,area_y2 in area_list[1:]:
            area_current = input_rgb[area_y1:area_y2,area_x1:area_x2].copy()
            area_current = Image.fromarray(area_current)
            img2 = transform_inference(area_current).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat1, feat2 = siamesen_model(img1, img2)  # 假设 forward 返回两个特征
                dist = F.pairwise_distance(feat1, feat2).item()
                score = dist
            if score > MARGIN:
                buffer.add(False)
                break
            else:
                buffer.add(True)

        # SSIM area 画框
        for area_x1,area_y1,area_x2,area_y2 in area_list:
            cv2.rectangle(img       = input, 
                          pt1       = (area_x1,area_y1),
                          pt2       = (area_x2,area_y2),
                          color     = (0,255,0),
                          thickness = 2)
        if buffer.all_true():
            label = f"Normal: {score:.2f}"
            label_color = font_color
        elif buffer.all_false():
            label = f"Abnormal: {score:.2f}"
            label_color = (255,0,255)
        else:
            label = f"Detecting: {score:.2f}"
            label_color = (255,255,0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 2
        text_x,text_y = 20, 50
        cv2.putText(img       = input,
                    text      = label,
                    org       = (text_x,text_y),
                    fontFace  = font,
                    fontScale = font_scale,
                    color     = label_color,
                    thickness = thickness,
                    lineType  = cv2.LINE_AA)
    elif class_idx == 1:
        class_name = class_names[class_idx]
        # print(f"检测到: {class_name}, 置信度: {max_conf:.2f}, 坐标: {boxes.xyxy[boxes_idx]}")
        x1, y1, x2, y2 = boxes.xyxy[boxes_idx].floor().int().tolist()
        _ , _ ,  w,  h = boxes.xywh[boxes_idx].floor().int().tolist()

        half_w = w // 2
        half_y = h // 2

        vit_box_center_x0 = int(x1 + half_w * 0.7)
        vit_box_center_x1 = int(x1 + half_w * 1.3)
        vit_box_center_x2 = int(x1 + half_w * 1.85)
        vit_box_center_x3 = int(x1 + half_w * 1.8)

        vit_box_y1 = int(y1 + h // 2 - 64)
        vit_box_y2 = int(y1 + h // 2 - 32)

        vit_box_y3 = int(y1 + h // 2 + 16)
        vit_box_y4 = int(y1 + h // 2 + 16 + 32)

        area_list = []
        area_list.append([vit_box_center_x0 - 16, vit_box_y1, vit_box_center_x0 + 16, vit_box_y2])
        area_list.append([vit_box_center_x1 - 16, vit_box_y1, vit_box_center_x1 + 16, vit_box_y2])
        area_list.append([vit_box_center_x2 - 16, vit_box_y1, vit_box_center_x2 + 16, vit_box_y2])
        area_list.append([vit_box_center_x3 - 16, vit_box_y3, vit_box_center_x3 + 16, vit_box_y4])


        patch_0.append(input[area_list[0][1]:area_list[0][3],area_list[0][0]:area_list[0][2]].copy())
        patch_1.append(input[area_list[1][1]:area_list[1][3],area_list[1][0]:area_list[1][2]].copy())
        patch_2.append(input[area_list[2][1]:area_list[2][3],area_list[2][0]:area_list[2][2]].copy())
        patch_3.append(input[area_list[3][1]:area_list[3][3],area_list[3][0]:area_list[3][2]].copy())
        patch_list = [patch_0,patch_1,patch_2,patch_3]
        if len(patch_0) == 49:
            input_tensor_list = []
            for i, patch in enumerate(patch_list):
                patches = np.array(patch)
                grid = patches.reshape(7, 7, 32, 32, 3)
                grid = np.transpose(grid, (0, 2, 1, 3, 4))
                rows = [np.concatenate([grid[i, :, j] for j in range(7)], axis=1) for i in range(7)]
                concatenated = np.concatenate(rows, axis=0)  # 最终: (224, 224, 3)
                concatenated_rgb = cv2.cvtColor(concatenated, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(concatenated_rgb)
                input_tensor = transformOfSL(pil_image).unsqueeze(0)
                input_tensor_list.append(input_tensor)
            # 拼接batch输入数据
            input_batch = torch.cat(input_tensor_list, dim=0)
            input_numpy = input_batch.numpy() 

            # 推理
            starttime = time.time()
            # 方式1: onnx
            # input_name = session.get_inputs()[0].name
            # outputs = session.run(None, {input_name: input_numpy})  # None 表示返回所有输出
            # print(f"⏰onnx推理耗时: {time.time() - starttime:.4f}")
            # logits = outputs[0] 
            # probabilities_batch = torch.softmax(torch.from_numpy(logits), dim=1)# (B, C)
            
            # 方式2: torch
            outputs = vit_model(input_batch.to(DEVICE))
            print(f"⏰torch推理耗时: {time.time() - starttime:.4f}")
            logits = outputs
            probabilities_batch = torch.softmax(logits, dim=1)# (B, C)

            predicted_batch = probabilities_batch.argmax(dim=1).numpy()# (B,)
            for i in range(probabilities_batch.shape[0]):  # B 次循环
                probabilities = probabilities_batch[i]
                predicted_class = predicted_batch[i]
                if predicted_class == 1:
                    label_color = font_color
                elif predicted_class == 0:
                    label_color = (255,0,255)
                else:
                    label_color = (255,255,0)
                label = f"{SLClsssNameDict[predicted_class]}, {probabilities[predicted_class]:.4f}"
                cv2.putText(img       = input,
                            text      = label,
                            org       = (20,50 + 50*i),
                            fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 1,
                            color     = label_color,
                            thickness = 2,
                            lineType  = cv2.LINE_AA)
                
        for area_x1,area_y1,area_x2,area_y2 in area_list:
            cv2.rectangle(img       = input, 
                          pt1       = (area_x1,area_y1),
                          pt2       = (area_x2,area_y2),
                          color     = (0,255,0),
                          thickness = 2)


    # draw
    if class_idx is not None:
        x1, y1, x2, y2 = boxes.xyxy[boxes_idx].floor().int().tolist()
        class_name = class_names[class_idx]
        cv2.rectangle(img       = input, 
                      pt1       = (x1,y1),
                      pt2       = (x2,y2),
                      color     = colors[boxes_idx],
                      thickness = 3)

        # 准备要显示的文本
        label = f"{class_name}: {conf:.2f}"
        (text_width,text_height),baseline = cv2.getTextSize(text      = label,
                                                            fontFace  = font,
                                                            fontScale = font_scale,
                                                            thickness = thickness)
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

        # 绘制文本背景框
        cv2.rectangle(img       = input,
                      pt1       = (text_x, text_y - text_height - 4),
                      pt2       = (text_x + text_width , text_y + baseline),
                      color     = colors[boxes_idx],
                      thickness = -1)
        # 绘制文字
        cv2.putText(img       = input,
                    text      = label,
                    org       = (text_x,text_y),
                    fontFace  = font,
                    fontScale = font_scale,
                    color     = font_color,
                    thickness = thickness,
                    lineType  = cv2.LINE_AA)
    return input


def differenceOfTimeSequence(input):
    pass


def stirring(video_path:str,output_folder:str,detect_function:Callable[[cv2.Mat], any])->None:
    """
    video_path: input video path
    output_folder: output path
    detect_function: function
    """
    # video_path = "D:/data_temp/20250814/Sampling/20250814-Purple-090.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频源")
        exit()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    video_name = os.path.basename(video_path)
    out = cv2.VideoWriter(os.path.join(output_folder,video_name), fourcc, 25, (1400, 800))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = frame.copy()
        if False:
            input,x1,y1,x2,y2 = cut_image(image)
            output = detect_function(input)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 3)
            image[y1:y2, x1:x2] = output
        else:
            image = detect_function(image)

        image = cv2.resize(image,(1400,800))
        cv2.imshow('result', image)
        out.write(image)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    out.release()


model_path = os.path.join(current_dir,'./runs/detect/train8/weights/best.pt')
model = YOLO(model_path)

# 液液预加载
MARGIN = 1.0
MODEL_PATH = os.path.join(current_dir,'models/siamese_stirring_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
siamesen_model = SiameseNetwork()
siamesen_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')) 
siamesen_model.to(DEVICE)
siamesen_model.eval()
transform_inference = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=3),   # 转为三通道灰度图（R=G=B）
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 固液预加载onnx
session = ort.InferenceSession('./models/vitv3_b_32_3class.onnx', providers=['CPUExecutionProvider'])
transformOfSL = transforms.Compose([
        transforms.Resize((224,224)),  # ViT-B/32 使用 224x224
        transforms.Grayscale(num_output_channels=3),   # 转为三通道灰度图（R=G=B）
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
SLClsssNameDict={0:"Abnormal", 1:"Normal", 2:"StandStill"}
patch_0 = deque(maxlen=49)
patch_1 = deque(maxlen=49)
patch_2 = deque(maxlen=49)
patch_3 = deque(maxlen=49)
#--torch--
from torchvision.models import vit_b_32
from vitnet.vitmodel import CustomViT
base_vit = vit_b_32()
vit_model = CustomViT(base_vit, grid_size=(7, 7))
vit_model.vit.heads.head = torch.nn.Linear(vit_model.vit.heads.head.in_features, 3)
vit_model.load_state_dict(torch.load('./models/vitv3_b_32_3class.pth',map_location='cpu'))
vit_model.to(DEVICE)


# video_folder = os.path.join(current_dir,'assets/original_video_cut')
video_folder = os.path.join(current_dir,'assets/SolidLiquidCut')
output_folder = os.path.join(current_dir,'assets/result_1104/vitnet')
def main():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file_name in os.listdir(video_folder):
        if file_name.endswith(".mp4"):
            video_path = os.path.join(video_folder, file_name)
            print(video_path)
            stirring(video_path,output_folder,differenceOfSpatialv2)


if __name__ == "__main__":
    main()
