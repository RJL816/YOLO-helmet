"""
VisDrone 数据集对比实验脚本
用于验证改进方法在公开数据集上的泛化能力
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # 添加项目目录到 sys.path 优先级

import os
from ultralytics import YOLO

# ==================== 配置区 ====================
# 修改这里来选择要训练的模型
MODE = "improved"  # 选项: "baseline" 或 "improved"

# 训练配置
EPOCHS = 50  # VisDrone用50轮足够看出效果
BATCH_SIZE = 16
IMG_SIZE = 640

# ==================== 路径配置 ====================
ROOT = Path(__file__).parent.parent
data_yaml = "ultralytics/cfg/datasets/VisDrone.yaml"
pretrain_pt = ROOT / "mycode/yolo11n.pt"
custom_yaml = ROOT / "ultralytics/cfg/models/11/yolo11.yaml"


def get_common_cfg():
    """获取通用配置"""
    return {
        "ROOT": ROOT,
        "data": data_yaml,
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
        "imgsz": IMG_SIZE,
        "project": str(ROOT / "my_baseTraining_runs"),
        "exist_ok": True,
        "plots": True,
        "verbose": True,
        "device": 0,  # 使用GPU，CPU改为"cpu"
    }


def train_baseline():
    """
    基线模型：YOLO11n + CIoU + nn.Upsample
    """
    cfg = get_common_cfg()

    print("=" * 60)
    print("  VisDrone 基线模型：YOLO11n (CIoU + nn.Upsample)")
    print("=" * 60)

    # 使用官方预训练权重
    model = YOLO(str(pretrain_pt))

    # 关闭 WIoU，使用标准 CIoU
    os.environ['USE_WIOU'] = '0'

    try:
        results = model.train(
            data=cfg["data"],
            epochs=cfg["epochs"],
            batch=cfg["batch"],
            imgsz=cfg["imgsz"],
            project=cfg["project"],
            name="VisDrone_baseline",
            exist_ok=cfg["exist_ok"],
            plots=cfg["plots"],
            verbose=cfg["verbose"],
            device=cfg["device"],
            patience=15,  # 早停
        )
        print(f"\n结果保存至：{results.save_dir}")
    except Exception as e:
        print(f"训练异常: {e}")
    finally:
        os.environ['USE_WIOU'] = '1'


def train_improved():
    """
    改进模型：YOLO11n + WIoU + DySample
    """
    cfg = get_common_cfg()

    print("=" * 60)
    print("  VisDrone 改进模型：YOLO11n + WIoU + DySample")
    print("=" * 60)

    # 使用自定义yaml（含DySample）
    model = YOLO(str(custom_yaml))
    model.load(str(pretrain_pt))  # 加载预训练权重

    # 开启 WIoU
    os.environ['USE_WIOU'] = '1'

    try:
        results = model.train(
            data=cfg["data"],
            epochs=cfg["epochs"],
            batch=cfg["batch"],
            imgsz=cfg["imgsz"],
            project=cfg["project"],
            name="VisDrone_improved",
            exist_ok=cfg["exist_ok"],
            plots=cfg["plots"],
            verbose=cfg["verbose"],
            device=cfg["device"],
            patience=15,
        )
        print(f"\n结果保存至：{results.save_dir}")
    except Exception as e:
        print(f"训练异常: {e}")
    finally:
        os.environ['USE_WIOU'] = '1'


def main():
    # 数据集会自动下载到 datasets/VisDrone
    print("\n📊 VisDrone 数据集对比实验")
    print(f"模式: {MODE}")
    print(f"轮数: {EPOCHS}")
    print(f"数据集: {data_yaml}")
    print("-" * 60)

    if MODE == "baseline":
        train_baseline()
    elif MODE == "improved":
        train_improved()
    else:
        print(f"❌ 未知模式: {MODE}")
        print("请设置 MODE='baseline' 或 MODE='improved'")


if __name__ == "__main__":
    main()
