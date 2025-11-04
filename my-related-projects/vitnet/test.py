import onnxruntime as ort
import numpy as np
import torch

# 加载模型
session = ort.InferenceSession("../models/vitv3_b_32_3class.onnx")

# 测试不同 batch size
for bs in [1, 4, 32]:
    dummy_input = np.random.randn(bs, 3, 224, 224).astype(np.float32)
    try:
        outputs = session.run(None, {"input0": dummy_input})
        print(f"✅ Batch {bs}: output shape = {outputs[0].shape}")
        # print(outputs)
        # print(outputs[0])
        # step 3: 转为 torch.Tensor（可选）
        probabilities = torch.from_numpy(outputs[0]).softmax(1)  # shape (1, 3)
        print(probabilities)
        # step 4: 获取预测类别
    except Exception as e:
        print(f"❌ Batch {bs} failed: {str(e)}")
