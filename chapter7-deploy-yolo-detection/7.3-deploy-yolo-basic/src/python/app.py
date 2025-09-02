import ctypes
import os
import time

# 指定新版 libstdc++ 所在路径，例如：
system_libstdcpp_path = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

# 提前加载新版 libstdc++
ctypes.CDLL(system_libstdcpp_path, mode=ctypes.RTLD_GLOBAL)
# 加载共享库
lib = ctypes.CDLL(os.path.join(os.getcwd(), "libinfer.so"))

# 设置函数签名
lib.init_model.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.run_inference.argtypes = [ctypes.c_char_p]

# 初始化模型（只调用一次）
onnx_path = b"models/onnx/yolo11m.onnx"
lib.init_model(onnx_path, 0, 0, 1)  # VERB=0, GPU=0, FP16=1

imagelist = os.listdir('./data/source')

# 多次调用推理
for img in imagelist:
    image_path = os.path.join("data/source",img).encode('utf-8')
    print(image_path)
    t0 = time.time()
    lib.run_inference(image_path)
    print('time:',round(time.time() - t0,3))

