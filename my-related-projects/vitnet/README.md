Vision Transforms
===

### 1 创建数据集

```bash
python createdata.py
python splitdata.py

```

### 2 训练推理

```bash
python vitv3.py
```

### 3 处理流程

 - [piplinev3.py](./piplinev3.py) : onnx 和 torch 单batch推理

 - [piplinev3plus.py](./piplinev3plus.py) : onnx 和 torch 多batch推理

 - 多batch推理，需要注意[vitv3.py](./vitv3.py),export onnx时, input的batch设置大于1.

    ```bash
    python piplinev3.py
    ```



