import cv2
from ultralytics import YOLO
import os
from typing import Callable
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import deque
from utils import StateBuffer

current_dir = os.path.dirname(os.path.abspath(__file__))
queue = deque(maxlen=25)
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


def detect_ALL(input):
    results = model(input, verbose=False,    iou=0.45,  conf=0.25)

    colors = {}
    np.random.seed(20)
    for i in range(5):
        colors[i] = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    font_color = (255,255,255)


    # 处理检测结果
    for result in results:
        boxes         = result.boxes  # 检测框对象
        class_indices = boxes.cls.tolist()  # 类别索引（Tensor）-> list
        confidences   = boxes.conf.tolist()  # 置信度（Tensor）-> list
        class_names   = model.names

        i = 0
        for idx, conf in zip(class_indices, confidences):
            class_name = class_names[int(idx)]  # 根据索引获取类别名
            # print(f"检测到: {class_name}, 置信度: {conf:.2f}, 坐标: {boxes.xyxy[i]}")
            x1, y1, x2, y2 = boxes.xyxy[i].floor().int().tolist()
            cv2.rectangle(img       = input, 
                          pt1       = (x1,y1),
                          pt2       = (x2,y2),
                          color     = colors[idx],
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
                          color     = colors[idx],
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
            i+=1

    return input

def detectOfYolo(input):
    results = model(input, verbose=False,    iou=0.45,  conf=0.25)

    colors = {}
    np.random.seed(20)
    for i in range(5):
        colors[i] = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    font_color = (255,255,255)

    result = results[0] # 单图片推理，只取第一张

    boxes         = result.boxes  # 检测框对象
    class_indices = boxes.cls.tolist()  # 类别索引（Tensor）-> list
    confidences   = boxes.conf.tolist()  # 置信度（Tensor）-> list
    class_names   = model.names
    
    if len(confidences)>0:

        max_confidence = max(confidences)
        result_index   = confidences.index(max_confidence)
        class_id       = class_indices[result_index]

        i    = result_index
        conf = max_confidence
        idx  = int(class_id)

        class_name = class_names[idx]
        # print(f"检测到: {class_name}, 置信度: {conf:.2f}, 坐标: {boxes.xyxy[i]}")
        x1, y1, x2, y2 = boxes.xyxy[i].floor().int().tolist()
        cv2.rectangle(img       = input, 
                      pt1       = (x1,y1),
                      pt2       = (x2,y2),
                      color     = colors[idx],
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
                      color     = colors[idx],
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
        
        if idx == 0 or idx == 2:
            buffer.add(False)
        else:
            buffer.add(True)

    else:
        buffer.add(False)

    if buffer.all_true():
        class_name = 'Normal'
        font_color = (0,255,0)
    elif buffer.all_false():
        class_name = 'Abnormal'
        font_color = (255,0,255)
    else:
        class_name = 'Detecting'
        font_color = (255,0,0)

    label = f"{class_name}"
    offset_x,offset_y = 864,0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 2
    text_x,text_y = 20 + offset_x, 50 + y2 + offset_y
    cv2.putText(img       = input,
                text      = label,
                org       = (text_x,text_y),
                fontFace  = font,
                fontScale = font_scale,
                color     = font_color,
                thickness = thickness,
                lineType  = cv2.LINE_AA)

    return input



# queue = deque(maxlen=25)
# buffer = StateBuffer(size = 25)
# buffer.init_state(state = False)
def differenceOfSpatial(input):
    offset_x,offset_y = 864,0
    x, y, w, h = 700 + offset_x, 750 + offset_y, 100, 550

    rectangle_current = input[y:y+h,x:x+w].copy()
    h,w,_ = rectangle_current.shape
    A = rectangle_current[:h//2][:]
    B = rectangle_current[h//2:][:]
    # print(A.shape,B.shape)
    gray_A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    gray_B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(gray_A, gray_B, 
                       full=True,
                       # data_range=gray_current.max() - gray_current.min()
                       )

    # rectangle_hisory = rectangle_current.copy()
    # === 处理差异图 diff ===
    # diff 是 [0, 1] 浮点范围的单通道图像
    diff_normalized = (diff * 255).astype("uint8")                    # 转为 0~255
    diff_colored = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)  # 伪彩色映射
    # === 水平拼接：左=原图块，右=差异热力图 ===
    combined_display = np.hstack([A, B,diff_colored])
    y1 = 0 + offset_y 
    y2 = combined_display.shape[0] + offset_y
    x1 = 0 + offset_x 
    x2 = combined_display.shape[1] + offset_x
    input[y1:y2, x1:x2] = combined_display
    
    # 绘制框
    cv2.rectangle(img       = input, 
                  pt1       = (x,y),
                  pt2       = (x+w,y+h//2),
                  color     = (0,255,0),
                  thickness = 3)
    cv2.rectangle(img       = input, 
                  pt1       = (x, y+h//2),
                  pt2       = (x+w,y+h),
                  color     = (255,255,0),
                  thickness = 3)

    # 绘制文字
    # if score > 0.85:
    #     class_name = 'Normal'
    #     font_color = (0,255,0)
    #     queue.append(True)
    # else:
    #     class_name = 'Abnormal'
    #     font_color = (255,0,255)
    #     queue.append(False)

    # if all(queue):
    #     class_name = 'Normal'
    #     font_color = (0,255,0)
    # elif not any(queue):
    #     class_name = 'Abnormal'
    #     font_color = (255,0,255)
    # else:
    #     class_name = 'Detecting'
    #     font_color = (255,0,0)

    if score > 0.85:
        class_name = 'Normal'
        font_color = (0,255,0)
        buffer.add(True)
    else:
        class_name = 'Abnormal'
        font_color = (255,0,255)
        buffer.add(False)

    if buffer.all_true():
        class_name = 'Normal'
        font_color = (0,255,0)
    elif buffer.all_false():
        class_name = 'Abnormal'
        font_color = (255,0,255)
    else:
        class_name = 'Detecting'
        font_color = (255,0,0)

    label = f"{class_name}: {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 2
    text_x,text_y = 20 + offset_x, 50 + y2
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



model_path = os.path.join(current_dir,'./runs/detect/train7/weights/best.pt')
model = YOLO(model_path)
video_folder = os.path.join(current_dir,'assets/original_video')
# output_folder = os.path.join(current_dir,'assets/result')
output_folder = os.path.join(current_dir,'assets/resultSpatial')
def main():
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file_name in os.listdir(video_folder):
        if file_name.endswith(".mp4"):
            video_path = os.path.join(video_folder, file_name)
            print(video_path)
            # stirring(video_path,output_folder,detect_ALL)
            stirring(video_path,output_folder,differenceOfSpatial)
            # break

if __name__ == "__main__":
    main()
