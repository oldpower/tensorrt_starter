import os

def splitDataset():
    import random
    from pathlib import Path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    random.seed(10)
    
    # 定义源目录和目标目录
    source_A = os.path.join(current_dir, '../assets/SLForVit/Normal')
    source_B = os.path.join(current_dir, '../assets/SLForVit/Abnormal')
    source_C = os.path.join(current_dir, '../assets/SLForVit/StandStill')
    # target_dataset = os.path.join(current_dir, '../assets/dataset/vitdataset')
    target_dataset = '/home/quan/workspace/cvstirring/assets/dataset/vitdataset'
    
    # 类别名称（用于目标文件夹命名）
    class_A_name = 'Normal'
    class_B_name = 'Abnormal'
    class_C_name = 'StandStill'
    
    # 创建目标目录结构
    train_A = os.path.join(target_dataset, 'train', class_A_name)
    train_B = os.path.join(target_dataset, 'train', class_B_name)
    train_C = os.path.join(target_dataset, 'train', class_C_name)
    val_A = os.path.join(target_dataset, 'val', class_A_name)
    val_B = os.path.join(target_dataset, 'val', class_B_name)
    val_C = os.path.join(target_dataset, 'val', class_C_name)
    
    for dir_path in [train_A, train_B, train_C, val_A, val_B, val_C]:
        os.makedirs(dir_path, exist_ok=True)
    
    def process_class(source_dir, class_name,dirlist):
        if not os.path.exists(source_dir):
            print(f"源目录不存在: {source_dir}")
            return
        
        # 获取所有图片文件（可根据需要扩展后缀）
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        all_files = [
            f for f in os.listdir(source_dir)
            if os.path.isfile(os.path.join(source_dir, f)) and
            Path(f).suffix.lower() in extensions
        ]
        
        if not all_files:
            print(f"在 {source_dir} 中未找到图片文件")
            return
        
        # 随机打乱
        random.shuffle(all_files)
        
        # 按 8:2 划分
        split_idx = int(0.8 * len(all_files))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        # 为训练集创建软链接
        for fname in train_files:
            src = os.path.join(source_dir, fname)
            dst = os.path.join(dirlist[0], fname)
            create_symlink(src, dst)
        
        # 为验证集创建软链接
        for fname in val_files:
            src = os.path.join(source_dir, fname)
            dst = os.path.join(dirlist[1], fname)
            create_symlink(src, dst)
        
        print(f"{class_name}: {len(train_files)} 训练, {len(val_files)} 验证")
    
    def create_symlink(src, dst):
        """创建软链接，若目标已存在则删除"""
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        try:
            os.symlink(src, dst)
            # print(f"链接: {dst} -> {src}")
        except Exception as e:
            print(f"创建软链接失败 {dst} -> {src}: {e}")
            # 备用：如果软链接失败（如Windows权限问题），可改为复制
            # shutil.copy2(src, dst)
    
    # 处理两个类别
    process_class(source_A, class_A_name,[train_A,val_A])
    process_class(source_B, class_B_name,[train_B,val_B])
    process_class(source_C, class_C_name,[train_C,val_C])

    print("数据集划分与软链接创建完成！")
    print(f"数据集路径: {target_dataset}")

if __name__ == "__main__":
    splitDataset()
