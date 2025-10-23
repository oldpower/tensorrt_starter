"""
说明：

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

---
# 可优化项
 - 输入数据归一化不使用torchvision
 - softmax 不使用 torch
 - infer 连续推理，相同输入尺寸下，内存预分配和复用

"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

import torch
from torchvision import transforms
from PIL import Image
import time
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

def onnx2engine():

    # 1 初始化资源
    # 初始化 Logger 和 Builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # 创建网络定义（启用显式批处理）
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    # 设置 workspace 大小（例如 2GB）
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)

    # 启用 FP16（如果硬件支持）
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 加载 ONNX 模型
    parser = trt.OnnxParser(network, logger)
    onnx_file_path = "./models/vit_b_32_2class.onnx"

    with open(onnx_file_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ Failed to parse ONNX file:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            exit(1)

    print("✅ ONNX parsed successfully.")

    # 2 配置profile
    # 获取输入张量
    input_tensor = network.get_input(0)  # 或通过名字: network.get_input_by_name("input")
    input_name = input_tensor.name
    print(f"Input name: {input_name}, shape: {input_tensor.shape}")

    # 创建优化 profile
    profile = builder.create_optimization_profile()

    # 设置动态维度范围
    # set_shape(name, min, opt, max)
    #   - min: 最小 shape（保证支持）
    #   - opt: 最优 shape（TensorRT 为此优化 kernel）
    #   - max: 最大 shape（不能超过）
    # profile.set_shape(
    #     input_name, 
    #     min=(1, 3, 256, 256),    # 最小输入
    #     opt=(4, 3, 512, 512),    # 常见输入（性能为此优化）
    #     max=(8, 3, 1024, 1024)   # 最大输入
    # )
    profile.set_shape(
        input_name,
        min=(1, 3, 224, 224),   # batch=1, 其他固定
        opt=(4, 3, 224, 224),   # batch=4（最常见）
        max=(8, 3, 224, 224)    # batch=8
    )

    # 将 profile 添加到 config
    config.add_optimization_profile(profile)

    print("✅ Optimization profile added for dynamic shape.")



    # 3 构建engine
    print("🔧 Building serialized engine...")
    engine_bytes = builder.build_serialized_network(network, config)

    if engine_bytes is None:
        print("❌ Engine build failed.")
        exit(1)

    # 保存 engine 文件
    engine_file_path = "./build/engines/pythontrt.engine"
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
    print(f"✅ Serialized engine saved to {engine_file_path}")

def inference_trt():
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open("./build/engines/pythontrt.engine", "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    # ✅ 用 with 确保 context 被正确销毁
    with engine.create_execution_context() as context:
        # ✅ 设置优化 profile（必须在设置 shape 前）
        context.set_optimization_profile_async(0, 0)

        input_shape = (3, 3, 224, 224)
        # context.set_binding_shape(0, input_shape)
        context.set_input_shape("input0", input_shape)
        actual_shape = context.get_tensor_shape("input0")
        print(f"input0 shape: {actual_shape}") 

        host_input = np.random.rand(*input_shape).astype(np.float32)
        device_input = cuda.mem_alloc(host_input.nbytes)

        output_shape = tuple(context.get_tensor_shape("output0"))
        print(f"output0 shape : {output_shape}")
        host_output = cuda.pagelocked_empty(output_shape, dtype=np.float32)
        device_output = cuda.mem_alloc(host_output.nbytes)

        # ✅ 设置 tensor 地址（关键！）
        context.set_tensor_address("input0", device_input)
        context.set_tensor_address("output0", device_output)

        stream = cuda.Stream()
        cuda.memcpy_htod_async(device_input, host_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(host_output, device_output, stream)
        stream.synchronize()
        
        print("✅ Inference with dynamic shape completed.")
        print(host_output)

        # ✅ 显式释放 GPU 内存
        device_input.free()
        device_output.free()


class TRTInfer:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # 🔍 自动提取所有输入和输出的名字
        self.input_names = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)

        print(f"✅ Engine loaded with {len(self.input_names)} inputs: {self.input_names}")
        print(f"✅ And {len(self.output_names)} outputs: {self.output_names}")

        # 用于管理内存
        self.device_inputs = {}
        self.device_outputs = {}
        self.host_outputs = {}

        # 👇 记录是否已分配内存,记录输入/输出分配的字节数（用于判断是否需要 realloc）
        self._allocated = False
        self.input_sizes = {}
        self.output_sizes = {}

    def infer(self, input_data):
        """
        推理函数
        :param input_data: 可以是：
            - dict: {input_name: np.ndarray}
            - np.ndarray: 自动绑定到第一个输入
        """
        # === 处理输入：统一为 dict 格式 ===
        if isinstance(input_data, dict):
            input_data_dict = input_data
        elif isinstance(input_data, np.ndarray):
            if len(self.input_names) != 1:
                raise ValueError(f"Model has {len(self.input_names)} inputs, but only one array provided. Use dict format.")
            input_data_dict = {self.input_names[0]: input_data}
        else:
            raise TypeError("input_data must be a dict or a numpy array")

        # === 检查所有输入是否都提供了 ===
        for name in self.input_names:
            if name not in input_data_dict:
                raise KeyError(f"Missing input tensor: '{name}'. Provided keys: {list(input_data_dict.keys())}")

        # 设置 profile
        self.context.set_optimization_profile_async(0, 0)

        # === 第一步：设置每个输入的 shape，并分配或复用显存 ===
        # 初始化 input_sizes 字典（第一次运行时）
        # if not hasattr(self, 'input_sizes'):
        #     self.input_sizes = {}
        for name in self.input_names:
            data = input_data_dict[name]  # 获取对应的 numpy 输入
            shape = data.shape

            # 设置输入 shape（动态 shape 支持）
            self.context.set_input_shape(name, shape)

            # 检查是否成功设置
            actual_shape = tuple(self.context.get_tensor_shape(name))
            if -1 in actual_shape:
                raise ValueError(f"Shape for input '{name}' not fully specified: {actual_shape}")

            # 分配 GPU 内存（如果之前没分配过，或 shape 变了）
            nbytes = data.nbytes
            if name not in self.device_inputs or self.input_sizes.get(name, 0) < nbytes:
                if name in self.device_inputs:
                    self.device_inputs[name].free()
                self.device_inputs[name] = cuda.mem_alloc(nbytes)
                self.input_sizes[name] = nbytes     # ✅ 记录大小

            # ✅ 绑定 tensor 地址
            self.context.set_tensor_address(name, self.device_inputs[name])

        # === 第二步：为每个输出分配内存并绑定地址 ===
        if not self._allocated:
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = np.float32  # 假设输出是 float32，可根据需要调整
                host_buf = cuda.pagelocked_empty(shape, dtype=dtype)
                device_buf = cuda.mem_alloc(host_buf.nbytes)

                self.host_outputs[name] = host_buf
                self.device_outputs[name] = device_buf
                self.output_sizes[name] = host_buf.nbytes  # ✅ 记录大小

                # ✅ 绑定输出 tensor 地址
                self.context.set_tensor_address(name, device_buf)
            self._allocated = True
        else:
            # ✅ 后续推理：检查 shape 是否变化，若变大则重新分配
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = np.float32
                nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

                if self.output_sizes[name] < nbytes:
                    # 释放旧内存
                    self.device_outputs[name].free()
                    # 重新分配
                    self.host_outputs[name] = cuda.pagelocked_empty(shape, dtype=dtype)
                    self.device_outputs[name] = cuda.mem_alloc(nbytes)
                    self.output_sizes[name] = nbytes

                # ✅ 每次必须重新绑定地址（因为 context 可能变了）
                self.context.set_tensor_address(name, self.device_outputs[name])

        # === 第三步：数据拷贝（Host -> Device）===
        for name in self.input_names:
            cuda.memcpy_htod_async(self.device_inputs[name], input_data_dict[name], self.stream)

        # === 第四步：执行推理 ===
        self.context.execute_async_v3(self.stream.handle)

        # === 第五步：拷贝输出结果（Device -> Host）===
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.host_outputs[name], self.device_outputs[name], self.stream)

        # 同步流
        self.stream.synchronize()

        # 返回输出字典
        return {name: self.host_outputs[name].copy() for name in self.output_names}  # .copy() 确保脱离 pagelocked memory

    def cleanup(self):
        """手动清理 GPU 内存"""
        for buf in self.device_inputs.values():
            buf.free()
        for buf in self.device_outputs.values():
            buf.free()
        self.device_inputs.clear()
        self.device_outputs.clear()

    def __del__(self):
        self.cleanup()
        try:
            del self.context
            del self.engine
            del self.runtime
        except Exception as _:
            pass

infer = TRTInfer('./build/engines/pythontrt.engine')
def inference_TRTInfer():
    pil_image = Image.open(os.path.join(current_dir,'./assets/StirringSL/20250814-StirringSolidLiquid_frame_0000.png'))
    transform = transforms.Compose([
        transforms.Resize((224,224)),  # ViT-B/32 使用 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(pil_image).unsqueeze(0)  # 添加 batch 维度
    # input_tensor = torch.cat([input_tensor, input_tensor], dim=0)
    inputs = input_tensor.numpy()

    starttime = time.time()
    for _ in range(5):
        # 推理
        outputs = infer.infer(inputs)['output0']
        print(f"⏰trtengine推理耗时: {time.time() - starttime:.4f} s")

        # 获取输出
        logits = outputs  # shape: [1, num_classes]
        # print("原始输出:", logits)
        # 转为概率（Softmax）
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
        # print("概率分布:", probabilities)
        # 获取预测类别
        predicted_class = probabilities.argmax()
        # print(f"预测类别: {predicted_class}, 置信度: {probabilities[predicted_class]:.4f}")

if __name__ == "__main__":
    # onnx2engine()
    # inference_trt()
    inference_TRTInfer()
    
