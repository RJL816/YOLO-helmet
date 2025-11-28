import os
import shutil
import random

def split_dataset(image_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # 确保比例和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "训练、验证、测试比例之和必须为1"

    # 检查输出文件夹是否存在
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    if not all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
        raise FileNotFoundError("请确保 train、val、test 文件夹已存在！")

    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        raise FileNotFoundError("total 文件夹中没有找到图片文件！")

    # 随机打乱图片顺序
    random.shuffle(image_files)

    # 计算各数据集的大小
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count  # 确保总数一致

    # 分配图片
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    # 复制图片到对应文件夹
    for file in train_files:
        shutil.copy(os.path.join(image_dir, file), os.path.join(train_dir, file))
    
    for file in val_files:
        shutil.copy(os.path.join(image_dir, file), os.path.join(val_dir, file))
    
    for file in test_files:
        shutil.copy(os.path.join(image_dir, file), os.path.join(test_dir, file))

    print(f"分配完成：训练集 {len(train_files)} 张，验证集 {len(val_files)} 张，测试集 {len(test_files)} 张")


image_dir = 'HelmetDetect\\datasets\\images\\total'  # 图片所在文件夹
output_dir = 'HelmetDetect\\datasets\\images'       # 输出文件夹（train/val/test 需已存在）
split_dataset(image_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)