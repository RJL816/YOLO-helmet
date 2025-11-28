import os
import torch
from ultralytics import YOLO
from datetime import datetime
import multiprocessing
#V2best
def main():
    # ===================== 基本配置 =====================
    model_cfg = r"E:/Desktop/yolo/HelmetDetect/ultralytics/cfg/models/11/yolo11.yaml"
    data_yaml = r"E:/Desktop/yolo/HelmetDetect/datasets/data.yaml"
    epochs = 100
    batch_size = 16
    imgsz = 640
    device = 0

    # ===================== 自定义训练名称和项目路径 =====================

    # 1. 自定义项目根目录，所有训练都会保存在这里
    #    如果注释掉下面这行，YOLO会使用默认的 'runs/detect'
    custom_project_dir = r"E:/Desktop/yolo/HelmetDetect/my_baseTraining_runs"

    # 2. 自定义本次实验的名称
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    custom_experiment_name = "helmet_baselineV4"

    model = YOLO(model_cfg)

    print(f"开始训练，本次实验名称：{custom_experiment_name}")

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,

            project=custom_project_dir,  # 告诉YOLO总的保存位置
            name=custom_experiment_name,  # 告诉YOLO本次训练的文件夹名


            plots=True,  # 开启图片生成（您已经做对了）
            verbose=True,
            exist_ok=True,  # 如果 'name' 目录已存在，则覆盖
            resume=False,

            # # --- 强效增强：解决“框不准”和“过拟合” ---
            # degrees = 15,  # 随机旋转 +/- 15 度
            # translate = 0.2,  # 随机平移 +/- 20%
            # scale = 0.5,  # 随机缩放 +/- 50% (这是关键！)
            # shear = 10,  # 随机剪切 +/- 10 度
            # perspective = 0.001,
            #
            #     # --- 强效增强：丰富背景，抗过拟合 ---
            # copy_paste = 0.1  # 10% 概率从其他图片复制物体粘贴过来

        )

    except Exception as e:
        print(f"训练过程异常: {e}")
        return

    save_dir = results.save_dir  # 这就是 YOLO 自动生成的目录，例如: E:/Desktop/yolo/HelmetDetect/my_training_runs/train_20251112_133000

    print("\n" + "=" * 60)
    print("所有任务完成！")
    print(f"1. 权重、图片和日志均保存在一个地方：\n   {save_dir}")

    print(f"\n2. 权重文件位于：\n   {os.path.join(save_dir, 'weights', 'best.pt')}")

    print(f"\n3. 所有图片 (PR_curve等) 位于：\n   {save_dir}")

    print(f"\n4. 完整的 TensorBoard 日志位于（它记录了所有epoch）：\n   {save_dir}")
    print(f"   请使用此命令查看：")
    print(f"   tensorboard --logdir=\"{save_dir}\"")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()