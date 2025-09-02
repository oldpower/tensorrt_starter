from ultralytics import YOLO
import onnx
import onnxsim
import torch 

# 确保使用 GPU（如果可用）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

modelname = "yolov8n"
model = YOLO(f"./{modelname}.pt").to(device)  # pretrained YOLO11n model
# model.export(format='onnx',half=False)

model.export(
    format="onnx",
    opset=12,          # 使用 ONNX opset 12（兼容性更好）
    simplify=True,     # 启用 ONNX 简化（会优化 INT64）
    dynamic=False,     # 禁用动态输入（除非你需要可变尺寸）
    device = device,
)


# 加载 ONNX 模型
onnx_model = onnx.load(f"{modelname}.onnx")
# 检查模型是否有效
onnx.checker.check_model(onnx_model)
# 使用onnx-simplifier来进行onnx的简化。
print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
model_onnx, check = onnxsim.simplify(onnx_model)
assert check, "assert check failed"
onnx.save(model_onnx, f"{modelname}.onnx")


# # 打印模型基本信息
# print("=== 模型元数据 ===")
# print(f"IR version: {onnx_model.ir_version}")
# print(f"Opset import: {onnx_model.opset_import}")
# print(f"Producer name: {onnx_model.producer_name}")
# print(f"Producer version: {onnx_model.producer_version}")

# # 打印输入输出信息
# print("\n=== 输入 ===")
# for i, input in enumerate(onnx_model.graph.input):
#     print(f"输入 {i}:")
#     print(f"  名称: {input.name}")
#     print("  维度:")
#     for dim in input.type.tensor_type.shape.dim:
#         print(f"    {dim.dim_value} (dim_param: {dim.dim_param})")

# print("\n=== 输出 ===")
# for i, output in enumerate(onnx_model.graph.output):
#     print(f"输出 {i}:")
#     print(f"  名称: {output.name}")
#     print("  维度:")
#     for dim in output.type.tensor_type.shape.dim:
#         print(f"    {dim.dim_value} (dim_param: {dim.dim_param})")

# # 打印节点信息（可选）
# print("\n=== 前5个节点（操作） ===")
# for i, node in enumerate(onnx_model.graph.node[:5]):
#     print(f"节点 {i}:")
#     print(f"  op_type: {node.op_type}")
#     print(f"  输入: {node.input}")
#     print(f"  输出: {node.output}")
