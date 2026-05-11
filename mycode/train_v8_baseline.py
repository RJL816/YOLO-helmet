"""
YOLOv8n 基线训练脚本（消融/对比实验专用）
-------------------------------------------
与 train2.py 的区别：
  - 使用原生 yolov8.yaml（无 DySample，使用 nn.Upsample）
  - loss 回退 CIoU（通过环境变量 USE_WIOU=0 控制，不需要改任何源码）
  - 实验结果保存到 my_baseTraining_runs/v8_baseline

恢复使用 WIoU + DySample（即 train2.py）时，直接运行 train2.py 即可，无需任何改动。
"""
import os
# 必须在 import ultralytics 之前设置，告知 loss.py 回退 CIoU
os.environ['USE_WIOU'] = '0'

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO
import multiprocessing


def main():
    # ===================== 配置 =====================
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]
    data_yaml = ROOT / "datasets/data.yaml"

    epochs     = 100
    batch_size = 16
    imgsz      = 640
    device     = 0
    custom_project_dir = ROOT / "my_baseTraining_runs"

    # ---- 选择要训练的模型 ----
    # 选项: 'n' = yolov8n, 's' = yolov8s, 'both' = 两者都训练
    MODEL_SIZE = 's'  # 修改这里: 'n', 's', 或 'both'

    # 模型配置映射
    model_configs = {
        'n': ("yolov8n.pt", "yolov8n_compare"),
        's': ("yolov8s.pt", "yolov8s_compare"),
    }

    # 根据选择确定要训练的实验
    if MODEL_SIZE == 'both':
        experiments = [model_configs['n'], model_configs['s']]
    else:
        experiments = [model_configs[MODEL_SIZE]]

    for model_pt, exp_name in experiments:
        print("=" * 50)
        print(f"  训练: {model_pt}  →  {exp_name}")
        print(f"  USE_WIOU = {os.environ.get('USE_WIOU')}")
        print("=" * 50)

        model = YOLO(model_pt)   # 从预训练权重出发 finetune
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=custom_project_dir,
            name=exp_name,
            exist_ok=False,
            patience=50,
            save=True,
            plots=True,
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
