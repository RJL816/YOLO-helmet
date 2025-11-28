import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path


# 1. 模型路径
MODEL_PATH = r"E:/Desktop/yolo/HelmetDetect/my_baseTraining_runs/helmet_baseline/weights/best.pt"
MODEL_PATH1 = r"E:/Desktop/yolo/HelmetDetect/mycode/HelmetDetect_Exp/yolo11_base/weights/best.pt"
MODEL_PATH2 = r"E:/Desktop/yolo/HelmetDetect/my_baseTraining_runs/helmet_baselineV2/weights/best.pt"
# 2. 图片路径
IMAGE_PATH = r"/mycode/photo/img_6.png"


# 3. 置信度阈值（太低会出很多框）
CONF_THRESHOLD = 0.3

# 4. 类别颜色（BGR 格式）
COLORS = {
    'two_wheeler': (0, 255, 0),      # 绿色
    'helmet': (255, 255, 0),         # 青色
    'without_helmet': (0, 0, 255)    # 红色
}
# ==============================================================

def main():
    # 检查路径
    if not Path(MODEL_PATH).exists():
        print(f"[错误] 模型不存在: {MODEL_PATH}")
        input("按回车退出...")
        return
    if not Path(IMAGE_PATH).exists():
        print(f"[错误] 图片不存在: {IMAGE_PATH}")
        input("按回车退出...")
        return

    # 加载模型
    print("正在加载模型...")
    model = YOLO(MODEL_PATH2)

    # 读取图片
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("[错误] 无法读取图片！")
        return

    # 预测
    print("正在预测...")
    results = model(IMAGE_PATH, conf=CONF_THRESHOLD, verbose=False)[0]

    # 绘制预测框
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf.item()
        cls_id = int(box.cls.item())
        label = results.names[cls_id]
        color = COLORS.get(label, (255, 255, 255))

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 写文字
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 转 RGB 显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 显示（关键：plt.show() 会阻塞，直到你手动关闭）
    plt.figure(figsize=(14, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"YOLO11 Prediction - {Path(IMAGE_PATH).name}", fontsize=16, pad=20)
    plt.tight_layout()

    print("预测完成！正在显示结果...")
    print("请手动关闭窗口以结束程序。")

    # 这一行会阻塞程序，直到你关掉窗口
    plt.show()

    print("窗口已关闭，程序结束。")

if __name__ == "__main__":
    main()