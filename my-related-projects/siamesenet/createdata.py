import cv2
from ultralytics import YOLO
from typing import Callable
import numpy as np
import pandas as pd
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

pairs = []
model_path = os.path.join(current_dir,'../runs/detect/train8/weights/best.pt')
output_path = os.path.join(current_dir,'../assets/siamesedata/images')
model = YOLO(model_path)
number = 1
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

    if class_idx is None:
        pass
    elif class_idx == 0:
        class_name = class_names[class_idx]
        # print(f"检测到: {class_name}, 置信度: {max_conf:.2f}, 坐标: {boxes.xyxy[boxes_idx]}")
        x1, y1, x2, y2 = boxes.xyxy[boxes_idx].floor().int().tolist()
        _ , _ ,  w,  h = boxes.xywh[boxes_idx].floor().int().tolist()

        half_w = w // 2

        ssim_box_x1 = int(x1 + half_w + half_w * 0.3)
        ssim_box_y1 = int(y1 + h * 0.05)
        ssim_box_x2 = int(x1 + half_w + half_w * 0.6)
        ssim_box_y2 = int(y1 + h * 0.95)
        ssim_box_w  = ssim_box_x2 - ssim_box_x1
        ssim_box_h  = ssim_box_y2 - ssim_box_y1
        # print(f"ssim box x1 y1 x2 y2: {ssim_box_x1} {ssim_box_y1} {ssim_box_x2} {ssim_box_y2}")
        # print(f"ssim box w h: {ssim_box_w} {ssim_box_h}")

        area_list = []
        ssimboxnum = 6
        area_h = ssim_box_h // ssimboxnum
        for i in range(ssimboxnum):
            y1 = ssim_box_y1 + area_h * i
            y2 = ssim_box_y1 + area_h * (i + 1)
            area_list.append([ssim_box_x1, y1, ssim_box_x2, y2])

        area_benchmark = input[area_list[0][1]:area_list[0][3],area_list[0][0]:area_list[0][2]].copy()
        filename = os.path.join(output_path,f"abnormal_{number:04d}.jpg")
        name_benchmark = f"abnormal_{number:04d}.jpg"
        print(filename)
        cv2.imwrite(filename,area_benchmark)
        number += 1
        for area_x1,area_y1,area_x2,area_y2 in area_list[1:][:]: 
            area_current = input[area_y1:area_y2,area_x1:area_x2].copy()
            filename = os.path.join(output_path,f"abnormal_{number:04d}.jpg")
            # area_current = cv2.resize(area_current,(64,64))
            cv2.imwrite(filename,area_current)
            # pairs.append([name_benchmark, f"normal_{number:04d}.jpg", 1])
            pairs.append([name_benchmark, f"abnormal_{number:04d}.jpg", 0])
            number += 1

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

    Interval = 25
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
    video_path = os.path.join(current_dir,'../assets/original_video/Absiamese.mp4')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    main(video_path,output_path,create_data)

    df = pd.DataFrame(pairs, columns=['img1', 'img2', 'label'])
    df.to_csv(os.path.join(current_dir,'../assets/siamesedata/abnormal_pairs.csv'), index=False)

