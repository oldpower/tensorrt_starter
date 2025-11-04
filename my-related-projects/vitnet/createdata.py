import cv2
from ultralytics import YOLO
from typing import Callable
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import StateBuffer
from collections import deque
current_dir = os.path.dirname(os.path.abspath(__file__))
buffer = StateBuffer(size = 49)
pairs = []
model_path = os.path.join(current_dir,'../runs/detect/train8/weights/best.pt')
output_path = os.path.join(current_dir,'../assets/datasets/SLForVit/StandStill')
model = YOLO(model_path)
number = 1
patch_0 = deque(maxlen=49)
patch_1 = deque(maxlen=49)
patch_2 = deque(maxlen=49)
patch_3 = deque(maxlen=49)
def create_data(input):
    global number
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

    if class_idx is None or class_idx == 0:
        buffer.clear()
        patch_0.clear()
        patch_1.clear()
        patch_2.clear()
        patch_3.clear()
        pass
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
            for patch in patch_list:
                patches = np.array(patch)  # 形状变为 (49, 32, 32, 3)
                grid = patches.reshape(7, 7, 32, 32, 3)
                # 交换轴，准备拼接: (7, 32, 7, 32, 3)
                grid = np.transpose(grid, (0, 2, 1, 3, 4))
                # 合并空间维度
                rows = [np.concatenate([grid[i, :, j] for j in range(7)], axis=1) for i in range(7)]
                # 再把7行纵向拼接
                concatenated = np.concatenate(rows, axis=0)  # 最终: (224, 224, 3)
                filename = os.path.join(output_path,f"StandStill03_{number:04d}.jpg")
                cv2.imwrite(filename,concatenated)
                number += 1
                print(f"number: {number},shape: {concatenated.shape}")

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
   
def main(video_path:str,output_path:str,detect_function:Callable[[cv2.Mat], any])->None:
    """
    video_path: input video path
    output_folder: output path
    detect_function: function
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频源")
        exit()

    Interval = 1
    count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = frame.copy()
        # if False:
        #     input,x1,y1,x2,y2 = cut_image(image)
        #     output = detect_function(input)

        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 3)
        #     image[y1:y2, x1:x2] = output
        # else:
        if count%Interval == 0:
            image = detect_function(image)
            count = 1
        else:
            count += 1

        image = cv2.resize(image,(1400,800))
        cv2.imshow('result', image)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # video_path = os.path.join(current_dir,'../assets/SolidLiquidCut/Abnormal-03.mp4')
    # video_path = os.path.join(current_dir,'../assets/SolidLiquidCut/Normal-01.mp4')
    video_path = os.path.join(current_dir,'../assets/SolidLiquidCut/StandStill-03.mp4')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    main(video_path,output_path,create_data)

    # df = pd.DataFrame(pairs, columns=['img1', 'img2', 'label'])
    # df.to_csv(os.path.join(current_dir,'../assets/siamesedata/abnormal_pairs.csv'), index=False)


