import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Using project root: {PROJECT_ROOT}")

import os
import multiprocessing
from ultralytics import YOLO

# ============================================================
# MODE 切换（改这一行，然后运行脚本）：
#
#   "baseline"      → 原版 YOLO11n（CIoU + nn.Upsample）      ← 消融第1行 (100轮)
#   "wiou_only"     → YOLO11n + WIoU，无 DySample              ← 消融第2行 (100轮)
#   "dysample_only" → YOLO11n + DySample，无 WIoU（CIoU）       ← 消融第3行 (150轮)
#   "improved"      → YOLO11n + WIoU + DySample（完整改进）    ← 消融第4行 (150轮)
#
# 消融实验运行顺序建议：
#   baseline → wiou_only → dysample_only → improved
#
# 注意：含 DySample 的版本训练 150 轮（随机初始化需更多时间收敛）
#       原版结构的版本训练 100 轮即可
# ============================================================
MODE = "wiou_only"


def get_common_cfg():
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]
    return dict(
        ROOT=ROOT,
        data_yaml=ROOT / "datasets/data.yaml",
        project_dir=ROOT / "my_baseTraining_runs",
        batch_size=16,
        imgsz=640,
        device=0,
    )


def train_improved():
    """改进版：YOLO11n + WIoU + DySample，从 yolo11n.pt 迁移预训练权重"""
    cfg = get_common_cfg()
    ROOT = cfg["ROOT"]

    custom_yaml = ROOT / "ultralytics/cfg/models/11/yolo11.yaml"
    pretrain_pt = ROOT / "mycode/yolo11n.pt"

    model = YOLO(str(custom_yaml))
    model.load(str(pretrain_pt))

    print("=" * 60)
    print("  模式：改进版（WIoU + DySample）- 150轮")
    print("=" * 60)

    try:
        results = model.train(
            data=cfg["data_yaml"],
            epochs=150,
            imgsz=cfg["imgsz"],
            batch=cfg["batch_size"],
            device=cfg["device"],
            project=cfg["project_dir"],
            name="WiseIou_DySample150_Pretrained",
            exist_ok=True,
            resume=False,
            plots=True,
            verbose=True,
            amp=True,
            workers=2,
        )
        print(f"\n结果保存至：{results.save_dir}")
    except Exception as e:
        print(f"训练过程异常: {e}")


def train_baseline():
    """原版基线：标准 YOLO11n（CIoU + nn.Upsample），直接 finetune yolo11n.pt
    与改进版参数量相同，起点相同，公平对比。

    重要说明：虽然使用官方 yolo11n.pt（其预训练时用的是 CIoU），
    但本代码已修改 loss.py 添加了 WIoU，且默认值是 USE_WIOU='1'。
    因此必须显式设置 USE_WIOU='0' 才能使用标准 CIoU 作为基线。
    """
    cfg = get_common_cfg()
    ROOT = cfg["ROOT"]
    pretrain_pt = ROOT / "mycode/yolo11n.pt"

    model = YOLO(str(pretrain_pt))   # 原版结构 + 完整预训练权重

    print("=" * 60)
    print("  模式：原版 YOLO11n 基线（CIoU + nn.Upsample）")
    print("=" * 60)

    try:
        os.environ['USE_WIOU'] = '0'   # 关闭 WIoU，回退标准 CIoU
        results = model.train(
            data=cfg["data_yaml"],
            epochs=100,
            imgsz=cfg["imgsz"],
            batch=cfg["batch_size"],
            device=cfg["device"],
            project=cfg["project_dir"],
            name="yolo11n_baseline",
            exist_ok=True,
            resume=False,
            plots=True,
            verbose=True,
            amp=True,
            workers=2,
        )
        print(f"\n结果保存至：{results.save_dir}")
    except Exception as e:
        print(f"训练过程异常: {e}")
    finally:
        os.environ['USE_WIOU'] = '1'   # 恢复默认


def train_wiou_only():
    """消融：YOLO11n + WIoU，保留原版 nn.Upsample，不用 DySample。
    使用原版 yolo11n.pt（无 DySample 层），直接加载全部预训练权重。
    USE_WIOU=1（默认），只改 loss，不改上采样结构。
    """
    cfg = get_common_cfg()
    ROOT = cfg["ROOT"]
    pretrain_pt = ROOT / "mycode/yolo11n.pt"

    # 原版结构（无 DySample），USE_WIOU 保持默认 '1'
    os.environ['USE_WIOU'] = '1'
    model = YOLO(str(pretrain_pt))

    print("=" * 60)
    print("  模式：消融 - 仅 WIoU（原版 Upsample，无 DySample）- 100轮")
    print("=" * 60)

    try:
        results = model.train(
            data=cfg["data_yaml"],
            epochs=100,
            imgsz=cfg["imgsz"],
            batch=cfg["batch_size"],
            device=cfg["device"],
            project=cfg["project_dir"],
            name="ablation_wiou_only",
            exist_ok=True,
            resume=False,
            plots=True,
            verbose=True,
            amp=True,
            workers=2,
        )
        print(f"\n结果保存至：{results.save_dir}")
    except Exception as e:
        print(f"训练过程异常: {e}")


def train_dysample_only():
    """消融：YOLO11n + DySample，使用 CIoU（不用 WIoU）。
    使用自定义 yaml（含 DySample），但 loss 回退 CIoU。
    """
    cfg = get_common_cfg()
    ROOT = cfg["ROOT"]
    custom_yaml = ROOT / "ultralytics/cfg/models/11/yolo11.yaml"
    pretrain_pt = ROOT / "mycode/yolo11n.pt"

    os.environ['USE_WIOU'] = '0'   # 关闭 WIoU，只让 DySample 起作用
    model = YOLO(str(custom_yaml))
    model.load(str(pretrain_pt))

    print("=" * 60)
    print("  模式：消融 - 仅 DySample（CIoU，无 WIoU）- 150轮")
    print("=" * 60)

    try:
        results = model.train(
            data=cfg["data_yaml"],
            epochs=150,
            imgsz=cfg["imgsz"],
            batch=cfg["batch_size"],
            device=cfg["device"],
            project=cfg["project_dir"],
            name="ablation_dysample_only",
            exist_ok=True,
            resume=False,
            plots=True,
            verbose=True,
            amp=True,
            workers=2,
        )
        print(f"\n结果保存至：{results.save_dir}")
    except Exception as e:
        print(f"训练过程异常: {e}")
    finally:
        os.environ['USE_WIOU'] = '1'


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if MODE == "improved":
        train_improved()
    elif MODE == "baseline":
        train_baseline()
    elif MODE == "wiou_only":
        train_wiou_only()
    elif MODE == "dysample_only":
        train_dysample_only()
    else:
        raise ValueError(f"未知 MODE: {MODE}")