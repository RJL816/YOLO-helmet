# rename_images.py
import os
import shutil

def rename_images_sequentially(folder_path, extensions=None):
    """
    将文件夹中的图片文件按顺序重命名为 00001.jpg, 00002.jpg, ...
    
    :param folder_path: 图片所在的文件夹路径
    :param extensions: 要处理的图片扩展名列表
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

    # 转换为小写以便比较
    extensions = [ext.lower() for ext in extensions]

    # 获取文件夹中所有符合条件的图片文件
    files = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                files.append(filename)

    # 按文件名排序（可选，确保顺序一致）
    files.sort()

    print(f"找到 {len(files)} 个图片文件。")

    # 重命名
    for index, filename in enumerate(files, start=1):
        old_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1]  # 保留原始扩展名
        new_filename = f"{index:05d}{ext}"   # 格式如 00001.jpg
        new_path = os.path.join(folder_path, new_filename)

        # 防止目标文件已存在
        if os.path.exists(new_path):
            print(f"跳过: {new_path} 已存在")
            continue

        shutil.move(old_path, new_path)
        print(f"重命名: {filename} -> {new_filename}")

    print("重命名完成！")

if __name__ == "__main__":

    folder = "E:/Desktop/yolo/HelmetDetect/datasets/images/train"  

    if not os.path.exists(folder):
        print(f"错误：文件夹 '{folder}' 不存在！")
    else:
        rename_images_sequentially(folder)