from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("../runs/train/weights/best.pt")  # pretrained YOLO11n model
def inference():
    # Load a model
    # 预测单张图片
    image_path = "../assets/pdfimg-image/00029.jpg"
    image = cv2.imread(image_path)
    image = np.random.randint(0, 256, (1000, 1500, 3), dtype=np.uint8)
    results = model(image, verbose=False,    iou=0.45,  conf=0.25)

    colors = {}
    np.random.seed(20)
    for i in range(5):
        colors[i] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) 

    # 处理检测结果
    for result in results:
        boxes = result.boxes  # 检测框对象
        # 获取所有检测框的类别索引和置信度
        class_indices = boxes.cls  # 类别索引（Tensor）
        confidences = boxes.conf  # 置信度（Tensor）
        # 将Tensor转为Python列表
        class_indices = class_indices.tolist()  # 例如 [0, 2, 1]
        confidences = confidences.tolist()      # 例如 [0.95, 0.88, 0.76]
        # 获取类别名称映射（如 {0: 'person', 1: 'car'}）
        class_names = model.names
        # 打印每个检测框的类别和置信度
        # 打印每个检测框的类别和置信度
        i = 0
        for idx, conf in zip(class_indices, confidences):
            class_name = class_names[int(idx)]  # 根据索引获取类别名
            print(f"检测到: {class_name}, 置信度: {conf:.2f}, 坐标: {boxes.xyxy[i]}")
            x1, y1, x2, y2 = boxes.xyxy[i].floor().int().tolist()
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[int(idx)], 2)

            # 准备要显示的文本
            label = f"{class_name}: {conf:.2f}"

            # 设置字体、大小、颜色等
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            color = (255, 255, 255)  # 白色文字
            bg_color = colors[int(idx)] 
            # 获取文本框大小
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            # 将标签放在框的上方，带背景
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10  # 防止标签超出图像顶部
            # 绘制背景矩形
            cv2.rectangle(image, (text_x, text_y - text_height - 4), 
                        (text_x + text_width, text_y + baseline + 2), bg_color, -1)
            # 绘制文字
            cv2.putText(image, label, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

            i+=1
        imagename = f"../assets/detection_{class_name}.jpg"
    cv2.imwrite(imagename,image)

if __name__ == "__main__":
    for i in range(100):
        inference()
