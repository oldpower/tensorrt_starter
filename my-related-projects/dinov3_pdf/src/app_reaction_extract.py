import base64
import os
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
from pydantic import BaseModel
import cv2
import numpy as np
import io
import json
from ultralytics import YOLO

# load dinov3 class model
from dinov3_inference import load_model_custom_device_v2

import timm
import torch

# ==============================
# 获取当前 worker 的序号（Uvicorn 会设置 UVICORN_WORKER_ID）
# worker_id = int(os.environ.get("UVICORN_WORKER_ID", 0))  # 默认为 0
# device = torch.device(f"cuda:{worker_id % 4}")  # 假设你有 4 张卡：0,1,2,3
# print(f"[Worker {worker_id}] Using device: {device}")


gpu_id = 2
device = torch.device(f"cuda:{gpu_id}")
print(f"Using device: {device}")
# ==============================

# Load models (each worker loads its own copy on its GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov3_model,data_config = load_model_custom_device_v2(device)
dinov3_transform = timm.data.create_transform(**data_config, is_training=False)

# load yolo detect model
model = YOLO("../runs/train5/weights/best.pt")
model.model.to(device)
app = FastAPI()

class ImageBase64(BaseModel):
    image: str

def base64_to_cv2_image(base64_str: str) -> np.ndarray:
    """Convert a base64 string to an OpenCV image."""
    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_numpy = np.array(image)
    image = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
    return image

@app.post("/reaction_class/")
async def reaction_class(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass

    # Convert BGR (OpenCV) → RGB
    input_rgb = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    # Convert numpy array (HWC, uint8, [0,255]) to PIL Image
    image = Image.fromarray(input_rgb)
    input_tensor = dinov3_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = dinov3_model(input_tensor)
        prob = torch.softmax(output, dim=1)
        # print([round(x, 4) for x in prob[0].tolist()])
        confidence, predicted = torch.max(prob, 1)

    class_names = ['molecular','reaction','others']
    result = {"result"    :predicted.item(),
              "class_name":class_names[predicted.item()],
              "confidence":f'{confidence.item():.4f}'}
    # result = json.dumps(result)
    return result


@app.post("/reaction_detect_and_dinoclass/")
async def reaction_detect_and_dinoclass(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        return {"result": False, "error": "Invalid image"}
    results = model(input,verbose = False, iou = 0.45, conf = 0.45)
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model.names

    rst_flag = False
    i = 0
    for idx, conf in zip(class_indices, confidences):
        if int(idx) in [0,1]:
            x1, y1, x2, y2 = boxes.xyxy[i].floor().int().tolist()
            input_rgb = input[y1:y2,x1:x2]
            image = Image.fromarray(input_rgb)
            input_tensor = dinov3_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = dinov3_model(input_tensor)
                prob = torch.softmax(output, dim=1)
                # print([round(x, 4) for x in prob[0].tolist()])
                confidence, predicted = torch.max(prob, 1)

            class_names = ['molecular','reaction','others']
            if predicted.item() in [1] and confidence.item()>0.85:
                rst_flag = True
                break
        i+=1

    return {"result":rst_flag}

if __name__ == "__main__":
    uvicorn.run("app_reaction_extract:app", host="0.0.0.0", port=8283, workers=4)

