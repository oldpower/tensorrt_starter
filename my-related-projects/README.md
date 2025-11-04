my-related-projects
===

## vit.py

 - vit模型默认架构:[vit.py](./vitnet/vit.py)

## vitv2.py

 - vit模型自定义架构:[vitv2.py](./vitnet/vitv2.py)

🎯 自定义架构说明
视频的时序帧中提取数据。
每帧取固定位置的 32×32 区域。
按照时间顺序，每 49 帧（7×7）组成一个“图像块”。
将这 49 个 patch 按行优先顺序排列，拼成一个 224×224 的伪图像（7×7 网格）。
然后输入给 ViT 模型处理。

👉 这本质上是：将时间序列建模为空间结构（2D 网格），利用 ViT 的全局注意力来捕捉时序依赖。

| 推荐等级 | 方案 | 说明 |
| - | - | - |
|⭐⭐⭐⭐⭐ | 可学习的 2D 位置编码| 最适合你 7×7 网格结构，显式建模行列|
|⭐⭐⭐⭐☆ | 普通可学习 1D 编码|（如 ViT 原版） 也可以，但不如 2D 精细|
|⭐⭐☆☆☆ | Sinusoidal 编码| 不推荐，不适合结构化网格|
|⭐⭐⭐⭐☆ + | 时间嵌入| 可考虑加入时间信息，增强时序建模|

使用`可学习的 2D 位置编码 PositionEmbedding2D `

## vitv3.py

 - vit模型自定义架构:[vitv3.py](./vitnet/vitv3.py)

🎯 自定义架构说明
`vitv3`是`vitv2`的升级版本，在`PositionEmbedding2D`上添加了时间编码`Time_Embedding`, 变为了**2D空间+1D时间**的方式`PositionEmbedding2Dv2`.


## trt.py

 - 包含了onnx2trt enginer , enginer inf:[trt.py](./vitne/trt.py)

```bash

# 1. 确保你有 pip 和 Python（3.8~3.11） 测试使用了3.11
python --version

# 2. 安装 tensorrt-cu12（自动包含所有依赖）
pip install tensorrt-cu12
# 或从 tar 文件安装（下载自 NVIDIA Developer）,指定版本
# pip install tensorrt-8.x.x.x-cp3x-none-linux_x86_64.whl

# 3. （可选）安装 pycuda，用于 GPU 内存管理
pip install pycuda

# 4. 其他
pip install Pillow
pip install torch torchvision

Installing collected packages: nvidia-cusparselt-cu12, mpmath, triton, sympy, nvidia-nvtx-cu12, nvidia-nvshmem-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufil
e-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, networkx, jinja2, fsspec, filelock, nvidia-cusparse-cu12, nvidia-cufft-cu12, nvidia-cudn
n-cu12, nvidia-cusolver-cu12, torch, torchvision
  Attempting uninstall: nvidia-cuda-runtime-cu12
    Found existing installation: nvidia-cuda-runtime-cu12 12.9.79
    Uninstalling nvidia-cuda-runtime-cu12-12.9.79:
      Successfully uninstalled nvidia-cuda-runtime-cu12-12.9.79
Successfully installed filelock-3.20.0 fsspec-2025.9.0 jinja2-3.1.6 mpmath-1.3.0 networkx-3.5 nvidia-cublas-cu12-12.8.4.1 nvidia-cuda-cupti-cu12-12.8.90 nvidia-cuda-nvrtc-cu12-12.8.93 nvidia
-cuda-runtime-cu12-12.8.90 nvidia-cudnn-cu12-9.10.2.21 nvidia-cufft-cu12-11.3.3.83 nvidia-cufile-cu12-1.13.1.3 nvidia-curand-cu12-10.3.9.90 nvidia-cusolver-cu12-11.7.3.90 nvidia-cusparse-cu1
2-12.5.8.93 nvidia-cusparselt-cu12-0.7.1 nvidia-nccl-cu12-2.27.5 nvidia-nvjitlink-cu12-12.8.93 nvidia-nvshmem-cu12-3.3.20 nvidia-nvtx-cu12-12.8.90 sympy-1.14.0 torch-2.9.0 torchvision-0.24.0
 triton-3.5.0

pip install opencv-python
pip install onnxruntime
pip install onnx

```
####  可优化项
 - 输入数据归一化不使用torchvision
 - softmax 不使用 torch
 - infer 连续推理，相同输入尺寸下，内存预分配和复用

## siamesenet

 - SiameseNet 全流程:[siamesenet](./siamesenet)
 - [README](./siamesenet/README.md)
 - 重点是**MySataset**: `SiameseDataset`, `mytransform`

        .
        ├── createdata.py   # 创建数据集
        ├── splitdata.py    # 分割数据集
        ├── MyModel.py      # SiameseNet
        ├── MyDataset.py    # Dataset Load
        ├── train.py        # 训练
        ├── inference.py    # 推理
        └── README.md       # 说明文档
