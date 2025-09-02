from ultralytics import YOLO
import torch
import time
import os
model = YOLO("./yolo11m.pt")  # pretrained YOLO11n model

# print(model.model)

# pytorch model
torch_model = model.model

x = torch.randn(1,3,640,640)

y = torch_model(x)


imagelist = os.listdir('./data/source')
# 多次调用推理
for img in imagelist:
    image_path = os.path.join("data/source",img)
    print(image_path)
    t0 = time.time()
    results = model(image_path,verbose=False)
    results[0].save(filename="result.jpg")  # save to disk
    print('time:',round(time.time() - t0,3))

# 递归打印所有 tensor 的 shape
def print_shapes(x, indent=0):
    if isinstance(x, (list, tuple)):
        for i, item in enumerate(x):
            print(' ' * indent + f'[{i}]')
            print_shapes(item, indent + 2)
    elif isinstance(x, torch.Tensor):
        print(' ' * indent + f'Shape: {x.shape}')
    else:
        print(' ' * indent + f'Type: {type(x)}')

print("模型输出结构：")
print_shapes(y)


# results = model("./bus.jpg")

# # Process results generator
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     # result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk

