import cv2
from ultralytics import YOLO
import os
from typing import Callable
import numpy as np
from skimage.metrics import structural_similarity as ssim
from utils import StateBuffer

import torch
from torchvision import transforms
from siamesenet.MyModel import SiameseNetwork
import torch.nn.functional as F
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
buffer = StateBuffer(size = 25)

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
    ssim_box_y1 = int(y1 + h * 0.1)
    ssim_box_x2 = int(x1 + half_w + half_w * 0.6)
    ssim_box_y2 = int(y1 + h * 0.9)
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

    if class_idx is None or class_idx == 1:
        buffer.clear()
    elif class_idx == 0:
        class_name = class_names[class_idx]
        # print(f"检测到: {class_name}, 置信度: {max_conf:.2f}, 坐标: {boxes.xyxy[boxes_idx]}")
        x1, y1, x2, y2 = boxes.xyxy[boxes_idx].floor().int().tolist()
        _ , _ ,  w,  h = boxes.xywh[boxes_idx].floor().int().tolist()

        ############
        #   SSIM   #
        ############
        area_list = getAreaListA(x1, y1, w, h, ssimboxnum=6)
        # area_list = getAreaListB(x1, y1, w, h, ssimboxnum=2)
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


# yolo 搅拌状态检测
model_path = os.path.join(current_dir,'./runs/detect/train8/weights/best.pt')
model = YOLO(model_path)
# SiameseNet
MARGIN = 0.8
MODEL_PATH = os.path.join(current_dir,'models/siamese_stirring_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
siamesen_model = SiameseNetwork()
siamesen_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')) 
siamesen_model.to(DEVICE)
siamesen_model.eval()
transform_inference = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 视频输入路径和输出路径
video_folder = os.path.join(current_dir,'assets/original_video')
output_folder = os.path.join(current_dir,'assets/resultSpatialv2')

def main():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(video_folder):
        if file_name.endswith(".mp4"):
            video_path = os.path.join(video_folder, file_name)
            print(video_path)
            # stirring(video_path,output_folder,detectOfYolo)
            stirring(video_path,output_folder,differenceOfSpatialv2)
            # break


if __name__ == "__main__":
    main()
