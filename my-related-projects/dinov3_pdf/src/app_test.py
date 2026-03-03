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
import time

weightpath = "../runs/train3/weights/best.pt"
model_01 = YOLO(weightpath)
model_02 = YOLO(weightpath)
model_03 = YOLO(weightpath)
model_04 = YOLO(weightpath)
model_05 = YOLO(weightpath)
model_06 = YOLO(weightpath)
model_07 = YOLO(weightpath)
model_08 = YOLO(weightpath)
model_09 = YOLO(weightpath)
model_10 = YOLO(weightpath)

weightpath = "../runs/train/weights/best.pt"
model_11 = YOLO(weightpath)
model_12 = YOLO(weightpath)
model_13 = YOLO(weightpath)
model_14 = YOLO(weightpath)
model_15 = YOLO(weightpath)
model_16 = YOLO(weightpath)
model_17 = YOLO(weightpath)
model_18 = YOLO(weightpath)
model_19 = YOLO(weightpath)
model_20 = YOLO(weightpath)


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


@app.post("/chem_01/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_01(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_01.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}
    print(f"{__name__}:inference time {Interval:.3f}")
    return result

@app.post("/chem_02/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_02(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_02.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}
    print(f"{__name__}:inference time {Interval:.3f}")
    return result

@app.post("/chem_03/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_03(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_03.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}

    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result

@app.post("/chem_04/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_04(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_04.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}

    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result

@app.post("/chem_05/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_05(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_05.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}

    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result

@app.post("/chem_06/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_06(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_06.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}

    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result

@app.post("/chem_07/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_07(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_07.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}

    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result

@app.post("/chem_08/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_08(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_08.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}

    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result

@app.post("/chem_09/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_09(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_09.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}

    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result

@app.post("/chem_10/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_10(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_10.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}
    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result

@app.post("/chem_11/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_11(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_11.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}
    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result


@app.post("/chem_12/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_12(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_12.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}
    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result


@app.post("/chem_13/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_10(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_13.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}
    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result


@app.post("/chem_14/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_14(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_14.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}
    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result



@app.post("/chem_15/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_15(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_15.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}
    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result


@app.post("/chem_16/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_16(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_16.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}
    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result


@app.post("/chem_17/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_17(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_17.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}
    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result


@app.post("/chem_18/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_18(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_18.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}
    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result


@app.post("/chem_19/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_19(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_19.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}
    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result

@app.post("/chem_20/")
async def dissolve_detect(image_data: ImageBase64):
    try:
        input = base64_to_cv2_image(image_data.image)
    except Exception as e:
        pass
    starttime = time.time()
    results = model_20(input,verbose = False, iou = 0.45, conf = 0.45)
    Interval = time.time() - starttime
    result = results[0]
    boxes           = result.boxes
    class_indices   = boxes.cls.tolist()
    confidences     = boxes.conf.tolist()
    class_names     = model_20.names
    if len(confidences):
        result = {"result":True,"inference time":f"{Interval:.3f}"}
    else:
        result = {"result":False,"inference time":f"{Interval:.3f}"}
    # result = json.dumps(result)
    print(f"{__name__}:inference time {Interval:.3f}")
    return result

if __name__ == "__main__":
    uvicorn.run("app_test:app", host="0.0.0.0", port=8181, workers=5)

