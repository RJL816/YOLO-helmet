import cv2
from ultralytics import YOLO
import time
import numpy as np

# --- 1. 配置区域  ---

# 1.1) 你的模型路径
MODEL_PATH = 'E:/Desktop/yolo/HelmetDetect/my_baseTraining_runs/helmet_baselineV4/weights/best.pt'  # 替换为你的 .pt 文件路径

# 1.2) 输入视频路径 (或摄像头)
VIDEO_SOURCE_PATH = 'E:/Desktop/yolo/HelmetDetect/mycode/video/video1.mp4'  # 替换为你的输入视频路径
# VIDEO_SOURCE_PATH = 0  # 使用 0 表示默认摄像头

# 1.3) 是否保存视频
SAVE_VIDEO = False
VIDEO_OUTPUT_PATH = 'output_video.mp4'

# 1.4) 置信度阈值
CONFIDENCE_THRESHOLD = 0.4

# 1.5) 自定义绘制设置
CUSTOM_LINE_THICKNESS = 2  # 框的粗细

# 1.6) 自定义类别颜色 (BGR格式)
CUSTOM_COLORS = {
    0: (255, 0, 0),  # 0: two_wheeler (蓝色)
    1: (0, 0, 255),  # 1: without_helmet (红色)
    2: (0, 255, 0)  # 2: helmet (绿色)
}
DEFAULT_COLOR = (255, 255, 255)

# --- 2. 显示窗口配置 ---

# 2.1) 是否调整显示窗口大小
RESIZE_DISPLAY = True

# 2.2) 调整后的显示窗口大小 (宽度, 高度)
# 无论你的视频多大，都会被缩放到这个尺寸来显示
# (这不会影响你保存的视频，保存的视频仍然是原始分辨率)
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# 2.3) 是否显示标签和置信度
SHOW_LABELS = True
LABEL_FONT_SCALE = 0.5  # 标签字体大小
LABEL_FONT_THICKNESS = 1  # 标签字体粗细


def main():
    print(f"正在加载模型: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"错误: 无法加载模型。 {e}")
        return

    class_names = model.names
    print(f"模型加载成功。检测类别: {class_names}")

    # 打开视频文件或摄像头
    cap = cv2.VideoCapture(VIDEO_SOURCE_PATH)
    if not cap.isOpened():
        print(f"错误: 无法打开视频源 {VIDEO_SOURCE_PATH}")
        return

    # 获取视频原始属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"视频原始信息: {frame_width}x{frame_height} @ {fps:.2f} FPS")

    # (可选) 设置视频写入器
    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 注意: 保存时使用原始的 frame_width 和 frame_height
        out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
        print(f"视频将以 {frame_width}x{frame_height} 分辨率保存到: {VIDEO_OUTPUT_PATH}")
    else:
        print("实时显示模式: 视频将不会被保存。")

    # 为了让窗口可以被用户手动缩放，我们提前创建窗口
    cv2.namedWindow('YOLOv11 实时预测', cv2.WINDOW_NORMAL)
    if RESIZE_DISPLAY:
        print(f"显示窗口将调整为: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    # -------------------

    print("开始逐帧处理... (按 'q' 键退出)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = frame.copy()

        # --- YOLO 预测 ---
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]

        # --- 手动绘制检测框 ---
        for box in result.boxes:
            # 1. 获取坐标
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # 2. 获取类别ID
            class_id = int(box.cls[0].cpu().numpy())

            # 3. 获取颜色
            color = CUSTOM_COLORS.get(class_id, DEFAULT_COLOR)

            # 4. 绘制矩形框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, CUSTOM_LINE_THICKNESS)

            # --- [!! 新增 !!] 绘制标签和置信度 ---
            if SHOW_LABELS:
                # 5. 获取置信度 和 类别名称
                conf = float(box.conf[0].cpu().numpy())
                label_text = f"{class_names[class_id]} {conf:.2f}"

                # 6. 计算标签文本的大小
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
                                            LABEL_FONT_THICKNESS)

                # 7. 计算标签和背景的位置
                # 默认放在框的上方
                label_bg_x1 = x1
                label_bg_y1 = y1 - h - 10  # 在文字上方留 10 像素空间
                label_bg_x2 = x1 + w
                label_bg_y2 = y1

                label_text_x = x1
                label_text_y = y1 - 5  # 文字基线

                # 8. 检查是否超出屏幕顶端
                if label_bg_y1 < 0:
                    # 如果超出，则移动到框的下方
                    label_bg_y1 = y2
                    label_bg_y2 = y2 + h + 10
                    label_text_y = y2 + h + 5

                # 9. 绘制标签背景
                cv2.rectangle(annotated_frame, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color,
                              cv2.FILLED)

                # 10. 绘制标签文字 (使用白色)
                cv2.putText(annotated_frame, label_text, (label_text_x, label_text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            LABEL_FONT_SCALE, (255, 255, 255), LABEL_FONT_THICKNESS)


        # (可选) 写入原始分辨率的帧
        if SAVE_VIDEO and out is not None:
            out.write(annotated_frame)

        # --- 调整显示帧的大小 ---
        if RESIZE_DISPLAY:
            # 将处理后的帧缩放到指定大小
            display_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        else:
            display_frame = annotated_frame

        # 实时显示处理中的视频 (显示缩放后的帧)
        cv2.imshow('YOLOv11 实时预测', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户中途退出...")
            break

    # --- 清理工作 ---
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("-------------------------")
    print("处理完成，资源已释放。")
    if SAVE_VIDEO and out is not None:
        print(f"结果已保存到: {VIDEO_OUTPUT_PATH}")
    print("-------------------------")


if __name__ == '__main__':
    main()