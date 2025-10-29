import pandas as pd
import random
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# ----------------------------
# 1. 配置路径
# ----------------------------
normal_csv   = os.path.join(current_dir,'../assets/dataset/siamesedata/normal_pairs.csv')      # label=0，相似对
abnormal_csv = os.path.join(current_dir,'../assets/dataset/siamesedata/abnormal_pairs.csv')  # label=1，不相似对
output_dir   = os.path.join(current_dir,"../assets/dataset/siamesedata")               # 输出目录

# ----------------------------
# 2. 读取原始数据
# ----------------------------
df_normal = pd.read_csv(normal_csv)   # 270 条，label=0
df_abnormal = pd.read_csv(abnormal_csv) # 800 条，label=1

# ----------------------------
# 3. 从 abnormal 中随机抽取 270 条（与 normal 平衡）
# ----------------------------
df_abnormal_balanced = df_abnormal.sample(n=270, random_state=42)  # 固定随机种子便于复现

# ----------------------------
# 4. 合并两类样本
# ----------------------------
# 添加 label 列（虽然可能已有，但确保一致）
df_normal['label'] = 0
df_abnormal_balanced['label'] = 1

# 合并并打乱顺序
df_combined = pd.concat([df_normal, df_abnormal_balanced], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)  # 全局打乱

# ----------------------------
# 5. 划分训练集和验证集（80% : 20%）
# ----------------------------
split_ratio = 0.8
split_idx = int(len(df_combined) * split_ratio)

df_train = df_combined[:split_idx]
df_val = df_combined[split_idx:]

# ----------------------------
# 6. 保存为 CSV
# ----------------------------
df_train.to_csv(os.path.join(output_dir, 'train_pairs.csv'), index=False)
df_val.to_csv(os.path.join(output_dir, 'val_pairs.csv'), index=False)

# ----------------------------
# 7. 打印统计信息
# ----------------------------
print(f"原始 normal 数量: {len(df_normal)}")
print(f"原始 abnormal 数量: {len(df_abnormal)}")
print(f"抽取 abnormal 数量: {len(df_abnormal_balanced)}")
print(f"合并后总数量: {len(df_combined)}")
print(f"训练集大小: {len(df_train)} (正常: {len(df_train[df_train['label']==0])}, 异常: {len(df_train[df_train['label']==1])})")
print(f"验证集大小: {len(df_val)} (正常: {len(df_val[df_val['label']==0])}, 异常: {len(df_val[df_val['label']==1])})")
print("✅ train_pairs.csv 和 val_pairs.csv 已生成！")
