import fitz
import numpy as np
import cv2
import io
import time
from ultralytics import YOLO
ckpt_path_yolo = r'./yolo11_ocr_smiles/models/train/weights/best.pt'

class PDFStreamReader:
    def __init__(self, pdf_path_or_bytes):
        """
        初始化PDF阅读器
        :param pdf_path_or_bytes: PDF文件路径或字节数据
        """
        if isinstance(pdf_path_or_bytes, bytes):
            self.doc = fitz.open(stream=pdf_path_or_bytes, filetype="pdf")
        else:
            self.doc = fitz.open(pdf_path_or_bytes)
        
        self.total_pages = len(self.doc)
    
    def get_page_as_image(self, page_num, dpi=150, return_format="numpy"):
        """
        获取指定页面为图片
        :param page_num: 页码 (从0开始)
        :param dpi: 分辨率
        :param return_format: 返回格式 "numpy", "pil", "bytes"
        """
        if page_num >= self.total_pages:
            raise ValueError(f"页码 {page_num} 超出范围，总页数: {self.total_pages}")
        
        page = self.doc[page_num]
        # 提取文本内容
        text_content = page.get_text()

        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        
        # 转换为不同格式
        if return_format == "numpy":
            # 转换为numpy数组 (OpenCV格式)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:  # RGBA转RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            return img_array,text_content
        
        elif return_format == "pil":
            from PIL import Image
            img_data = pix.tobytes("ppm")
            return Image.open(io.BytesIO(img_data)),text_content
        
        elif return_format == "bytes":
            return pix.tobytes("png"),text_content
    
    # def get_all_pages_as_images(self, dpi=150, return_format="numpy"):
    #     """获取所有页面为图片列表"""
    #     images = []
    #     for i in range(self.total_pages):
    #         images.append(self.get_page_as_image(i, dpi, return_format))
    #     return images
    def get_all_pages_as_images(self, dpi=150, return_format="numpy"):
        """
        获取所有页面为图片和文本内容列表
        :return: [(image1, text1), (image2, text2), ...]
        """
        images_and_texts = []
        for i in range(self.total_pages):
            image, text = self.get_page_as_image(i, dpi, return_format)
            images_and_texts.append((image, text))
        return images_and_texts
    
    def close(self):
        """关闭文档"""
        self.doc.close()

def test_01():
    # 使用示例
    pdf_reader = PDFStreamReader("./assets/jrm.2019.04703.pdf")

    # # 实时获取单页
    # page_image,text_content = pdf_reader.get_page_as_image(2, dpi=100, return_format="numpy")
    # page_image = test_02(page_image)
    # print(f"图片尺寸: {page_image.shape}")
    # print(text_content)

    # # 直接在OpenCV中显示
    # cv2.imshow('PDF Page', page_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 获取所有页面
    all_images = pdf_reader.get_all_pages_as_images(dpi=150)
    print(f"共获取 {len(all_images)} 页")
    for image,text_content in all_images:
        starttime = time.time()
        test_02(image)
        print(f"cost time:{time.time() - starttime:.2f}")
        print(text_content)

    pdf_reader.close()

def test_02(input_):
    input = input_.copy()
    colors = {}
    np.random.seed(20)
    for i in range(5):
        colors[i] = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    font_color = (255,255,255)

    model = YOLO(model= ckpt_path_yolo)  
    results = model(input,verbose = False, iou = 0.45, conf = 0.1)
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model.names
    
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

if __name__ == "__main__":
    start_time = time.time()
    test_01()
    print(f"cost time:{time.time() - start_time:.2f}")
