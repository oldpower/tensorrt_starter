import torch
print(torch.__version__)          # 应显示 2.x.x
print(torch.cuda.is_available())  # 应返回 True
print(torch.version.cuda)         # 应显示 12.1

# 确保使用 GPU（如果可用）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
