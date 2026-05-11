"""
智能头盔检测监控大屏 - FastAPI 后端
========================================
功能:
  - SQLite 持久化检测记录
  - 视频文件上传
  - MJPEG 推流 (浏览器 <img> 直接播放)
  - WebSocket 实时推送检测数据
  - RESTful 历史查询接口
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 避免 OpenMP 多实例冲突警告

import torch
import asyncio
import threading
import time
import cv2
import json
import shutil
import uuid
from collections import deque, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    UploadFile, File, Query, HTTPException
)
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ─────────────────────────────────────────────────────────────
#  0. 基础路径 & 配置
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent          # HelmetDetect/
APP_DIR  = Path(__file__).parent                 # HelmetDetect/app/

MODEL_PATH  = BASE_DIR / "my_baseTraining_runs" / "WiseIou_DySampleV1" / "weights" / "best.pt"
UPLOAD_DIR  = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD  = 0.4   # 检测置信度阈值
SAVE_INTERVAL   = 1.0   # 数据库写入 & WS 推送间隔（秒）
JPEG_QUALITY    = 80    # MJPEG 流的 JPEG 质量

# 类别映射 (与训练时 data.yaml 对应)
#   0 = two_wheeler  (两轮车)
#   1 = without_helmet (未戴头盔) ← 违规
#   2 = helmet         (已戴头盔)
CLASS_NAMES = {0: "two_wheeler", 1: "no_helmet", 2: "helmet"}
CLASS_COLORS_BGR = {
    0: (255, 120,   0),   # 橙色
    1: (  0,   0, 220),   # 红色  ← 违规高亮
    2: (  0, 200,   0),   # 绿色
}

# ——— 骑车人-头盔空间关联参数 ———
ASSOC_MARGIN_RATIO = 0.3   # 水平容差 = vehicle_width * ratio
ASSOC_VERT_RATIO   = 1.0   # 头盔框底部允许在车辆框顶部上方 vehicle_height * ratio 内

# ——— 滑动窗口投票参数 ———
VOTE_WINDOW = 7   # 维护每个 tracker ID 最近 N 帧的类别历史，取众数作为最终类别

# ——— 推理性能参数 ———
INFER_EVERY_N = 2    # 每 N 帧做一次跟踪推理，中间帧复用上次结果（1=每帧都推理）
TARGET_FPS    = 40   # MJPEG 输出目标帧率上限
INFER_IMGSZ   = 640  # 推理输入分辨率；改为 416/320 可显著提速（精度略降）
# 自动选择 GPU；若无 CUDA 则退到 CPU（CPU 推理会很慢！）
INFER_DEVICE  = 0 if torch.cuda.is_available() else "cpu"
INFER_HALF    = torch.cuda.is_available()  # GPU 时使用 FP16 半精度，速度约提升 30%

# ——— 违章抓拍参数 ———
VIOLATION_CONFIRM_FRAMES = 5    # 连续 N 帧投票结果为 no_helmet 才触发抓拍
VIOLATION_DIR = APP_DIR / "violations"
VIOLATION_DIR.mkdir(parents=True, exist_ok=True)

# ——— 难例挖掘参数 ———
HARD_EXAMPLE_DIR  = APP_DIR / "hard_examples"
HARD_IMAGES_DIR   = HARD_EXAMPLE_DIR / "images"
HARD_LABELS_DIR   = HARD_EXAMPLE_DIR / "labels"
HARD_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
HARD_LABELS_DIR.mkdir(parents=True, exist_ok=True)
HARD_CONF_LOW     = 0.40   # 置信度低于此值不检出; 大于等于此值且小于 HIGH 为难例
HARD_CONF_HIGH    = 0.60   # 简单样本阈值上界
HARD_SAVE_INTERVAL = 3.0   # 难例保存最小间隔（秒），避免磁盘写满

# ─────────────────────────────────────────────────────────────
#  1. 数据库 & ORM 模型
# ─────────────────────────────────────────────────────────────
DB_PATH = APP_DIR / "detections.db"
engine  = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class DetectionRecord(Base):
    """每隔 SAVE_INTERVAL 秒写入一条检测统计记录"""
    __tablename__ = "detection_records"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    timestamp           = Column(DateTime, nullable=False, default=datetime.now)
    without_helmet_count = Column(Integer, nullable=False, default=0)   # 未戴头盔人数
    with_helmet_count   = Column(Integer, nullable=False, default=0)    # 已戴头盔人数


class ViolationRecord(Base):
    """违章抓拍记录：某个跟踪目标被确认为 no_helmet 后保存"""
    __tablename__ = "violation_records"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    timestamp  = Column(DateTime, nullable=False, default=datetime.now)
    tracker_id = Column(Integer, nullable=False)           # ByteTrack 跟踪 ID
    image_path = Column(String(512), nullable=False)       # 抓拍图片相对路径
    confidence = Column(Float, nullable=False, default=0)  # 检测置信度
    plate_text = Column(String(64), nullable=True)         # OCR 车牌识别结果（预留）
    face_id    = Column(String(128), nullable=True)        # 人脸识别 ID（预留）
    status     = Column(String(32), nullable=False, default="pending")  # pending/confirmed/dismissed


class HardExampleRecord(Base):
    """难例挖掘记录"""
    __tablename__ = "hard_example_records"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    timestamp   = Column(DateTime, nullable=False, default=datetime.now)
    image_name  = Column(String(256), nullable=False)      # 图片文件名
    label_name  = Column(String(256), nullable=False)      # 标注文件名
    avg_conf    = Column(Float, nullable=False, default=0) # 该帧中触发难例的平均置信度
    num_objects = Column(Integer, nullable=False, default=0)


# 应用启动时自动建表
Base.metadata.create_all(bind=engine)


# ─────────────────────────────────────────────────────────────
#  2. FastAPI 实例 & CORS
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="智能头盔检测监控大屏", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
#  3. WebSocket 连接管理器
# ─────────────────────────────────────────────────────────────
class ConnectionManager:
    """线程安全的 WebSocket 广播管理器"""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        print(f"[WS] 客户端接入，当前连接数: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        print(f"[WS] 客户端断开，当前连接数: {len(self.active)}")

    async def broadcast(self, data: dict):
        """向所有活跃客户端广播 JSON 消息"""
        if not self.active:
            return
        msg = json.dumps(data, ensure_ascii=False)
        dead: list[WebSocket] = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


ws_manager = ConnectionManager()


# ─────────────────────────────────────────────────────────────
#  4. 全局推理状态 (线程安全)
# ─────────────────────────────────────────────────────────────
class InferenceState:
    """用锁保护的全局推理共享状态"""

    def __init__(self):
        self._lock           = threading.Lock()
        self._frame_bytes: bytes = b""           # 最新 JPEG 帧
        self._without: int   = 0                 # 未戴头盔人数
        self._with_hm: int   = 0                 # 已戴头盔人数
        self.running: bool   = False             # 推理线程是否在运行
        self.thread: Optional[threading.Thread] = None
        self.source: Optional[str | int] = None  # 路径 or 0(摄像头)

    # ── 写 ──────────────────────────────────────────────────────
    def update(self, frame_bytes: bytes, without: int, with_hm: int):
        with self._lock:
            self._frame_bytes = frame_bytes
            self._without     = without
            self._with_hm     = with_hm

    # ── 读 ──────────────────────────────────────────────────────
    def get_frame(self) -> bytes:
        with self._lock:
            return self._frame_bytes

    def get_counts(self) -> tuple[int, int]:
        with self._lock:
            return self._without, self._with_hm


inf_state = InferenceState()
_event_loop: Optional[asyncio.AbstractEventLoop] = None   # 主事件循环引用


# ─────────────────────────────────────────────────────────────
#  5. YOLO 推理线程函数（目标跟踪 + 滑动窗口投票）
# ─────────────────────────────────────────────────────────────
def _is_on_vehicle(hx1, hy1, hx2, hy2, vx1, vy1, vx2, vy2):
    """判断头盔框是否在车辆框的上方区域内（垂直投影法）"""
    vw = vx2 - vx1
    vh = vy2 - vy1
    margin = vw * ASSOC_MARGIN_RATIO
    head_cx = (hx1 + hx2) / 2.0
    head_bottom = hy2
    if head_cx < vx1 - margin or head_cx > vx2 + margin:
        return False
    if head_bottom < vy1 - vh * ASSOC_VERT_RATIO or head_bottom > vy2:
        return False
    return True


def _inference_worker(source: str | int, loop: asyncio.AbstractEventLoop):
    """
    独立线程：循环读取视频帧 → YOLO 跟踪推理 → 滑动窗口投票 → 编码 JPEG →
    定期写 DB + 通过 asyncio 广播 WebSocket
    """
    print(f"[推理] 加载模型: {MODEL_PATH}")
    try:
        model = YOLO(str(MODEL_PATH))
    except Exception as e:
        print(f"[推理] 模型加载失败: {e}")
        inf_state.running = False
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[推理] 无法打开视频源: {source}")
        inf_state.running = False
        return

    db = SessionLocal()
    last_save = time.time()

    # 每个 tracker ID 的类别历史队列，用于滑动窗口投票
    track_cls_history: dict[int, deque] = {}   # {tid: deque([cls_id, ...], maxlen=VOTE_WINDOW)}

    # 违章抓拍：连续被判定为 no_helmet 的帧数 & 已抓拍集合
    violation_streak: dict[int, int] = {}      # {tid: consecutive_no_helmet_frames}
    captured_tids: set[int] = set()            # 已经抓拍过的 tid，避免重复

    # 难例挖掘：上次保存时间
    last_hard_save = 0.0

    # 帧率控制 & 隔帧推理
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = 1.0 / min(video_fps, TARGET_FPS)   # 每帧最短间隔
    frame_count = 0
    last_annotated_bytes: bytes = b""   # 上一次推理的 JPEG 结果
    last_without = 0
    last_with    = 0

    device_name = f"GPU:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU（警告：CPU推理很慢！）"
    print(f"[推理] 推理设备: {device_name}, imgsz={INFER_IMGSZ}, half={INFER_HALF}")
    print(f"[推理] 推理启动（跟踪+投票 N={VOTE_WINDOW}, 每{INFER_EVERY_N}帧推理1次, 目标{TARGET_FPS}fps），视频源: {source}")

    try:
        while inf_state.running:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                # 视频文件播放完毕 → 循环
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    track_cls_history.clear()   # 视频循环时重置跟踪历史
                    violation_streak.clear()
                    captured_tids.clear()
                    frame_count = 0
                    continue
                else:
                    break

            frame_count += 1

            # ── 隔帧推理：非推理帧直接复用上次结果 ────────────
            if frame_count % INFER_EVERY_N != 1 and INFER_EVERY_N > 1 and last_annotated_bytes:
                inf_state.update(last_annotated_bytes, last_without, last_with)
                # 帧率控制 - 等待到下一帧时间点
                elapsed = time.time() - t_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                continue

            # ── YOLO 跟踪推理（ByteTrack）────────────────────
            results = model.track(
                frame,
                conf=CONF_THRESHOLD,
                persist=True,
                verbose=False,
                device=INFER_DEVICE,
                imgsz=INFER_IMGSZ,
                half=INFER_HALF,
            )
            boxes   = results[0].boxes

            # ── 分离车辆框和头盔框，提取 tracker ID ──────────
            vehicle_dets = []   # (x1,y1,x2,y2,conf,tid)
            head_dets    = []   # (x1,y1,x2,y2,conf,cls_id,tid)

            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                conf   = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                # tracker ID（某些帧可能无 ID，则为 -1）
                tid = int(box.id[0].cpu().numpy()) if box.id is not None else -1
                if cls_id == 0:
                    vehicle_dets.append((x1, y1, x2, y2, conf, tid))
                elif cls_id in (1, 2):
                    head_dets.append((x1, y1, x2, y2, conf, cls_id, tid))

            # ── 骑车人-头盔空间关联 ──────────────────────────
            associated = []  # (x1,y1,x2,y2,conf,cls_id,tid)
            for h in head_dets:
                hx1, hy1, hx2, hy2, hconf, hcls, htid = h
                for v in vehicle_dets:
                    vx1, vy1, vx2, vy2, _, _ = v
                    if _is_on_vehicle(hx1, hy1, hx2, hy2, vx1, vy1, vx2, vy2):
                        associated.append(h)
                        break

            # ── 滑动窗口投票 & 绘制 & 计数 ──────────────────
            without_cnt = 0
            with_cnt    = 0
            annotated   = frame.copy()

            # 画车辆框
            for vx1, vy1, vx2, vy2, vconf, vtid in vehicle_dets:
                color = CLASS_COLORS_BGR[0]
                cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), color, 2)
                label = f"{CLASS_NAMES[0]} {vconf:.2f}"
                if vtid >= 0:
                    label = f"ID{vtid} {label}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ly1 = vy1 - th - 8 if vy1 - th - 8 >= 0 else vy2
                ly2 = ly1 + th + 8
                cv2.rectangle(annotated, (vx1, ly1), (vx1 + tw + 6, ly2), color, cv2.FILLED)
                cv2.putText(annotated, label, (vx1 + 3, ly2 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 画关联到车辆的头盔框（经投票平滑后）
            for hx1, hy1, hx2, hy2, hconf, hcls, htid in associated:
                # --- 滑动窗口投票 ---
                if htid >= 0:
                    if htid not in track_cls_history:
                        track_cls_history[htid] = deque(maxlen=VOTE_WINDOW)
                    track_cls_history[htid].append(hcls)
                    # 取众数作为最终类别
                    final_cls = Counter(track_cls_history[htid]).most_common(1)[0][0]
                else:
                    # 无 tracker ID 时退化为单帧结果
                    final_cls = hcls

                color = CLASS_COLORS_BGR.get(final_cls, (255, 255, 255))
                cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), color, 2)
                label = f"{CLASS_NAMES.get(final_cls, str(final_cls))} {hconf:.2f}"
                if htid >= 0:
                    label = f"ID{htid} {label}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ly1 = hy1 - th - 8 if hy1 - th - 8 >= 0 else hy2
                ly2 = ly1 + th + 8
                cv2.rectangle(annotated, (hx1, ly1), (hx1 + tw + 6, ly2), color, cv2.FILLED)
                cv2.putText(annotated, label, (hx1 + 3, ly2 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if final_cls == 1:
                    without_cnt += 1
                elif final_cls == 2:
                    with_cnt += 1

                # ── 违章抓拍判定 ─────────────────────────────
                if htid >= 0:
                    if final_cls == 1:
                        violation_streak[htid] = violation_streak.get(htid, 0) + 1
                        if (violation_streak[htid] >= VIOLATION_CONFIRM_FRAMES
                                and htid not in captured_tids):
                            # 触发抓拍
                            captured_tids.add(htid)
                            vio_filename = f"vio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_tid{htid}.jpg"
                            vio_path = VIOLATION_DIR / vio_filename
                            # 在原图上高亮违章区域
                            vio_img = frame.copy()
                            cv2.rectangle(vio_img, (hx1, hy1), (hx2, hy2), (0, 0, 255), 3)
                            cv2.putText(vio_img, f"VIOLATION ID{htid}",
                                        (hx1, max(hy1 - 10, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.imwrite(str(vio_path), vio_img)
                            # 写 DB
                            try:
                                vio_record = ViolationRecord(
                                    timestamp=datetime.now(),
                                    tracker_id=htid,
                                    image_path=vio_filename,
                                    confidence=round(hconf, 4),
                                    status="pending",
                                )
                                db.add(vio_record)
                                db.commit()
                                vio_id = vio_record.id
                            except Exception as e:
                                print(f"[抓拍] DB 写入失败: {e}")
                                db.rollback()
                                vio_id = None
                            # 广播违章事件
                            vio_payload = {
                                "event": "violation",
                                "id": vio_id,
                                "tracker_id": htid,
                                "confidence": round(hconf, 4),
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "image": vio_filename,
                            }
                            asyncio.run_coroutine_threadsafe(
                                ws_manager.broadcast(vio_payload), loop
                            )
                            print(f"[抓拍] 违章抓拍 tid={htid}, conf={hconf:.3f}, file={vio_filename}")
                    else:
                        # 投票结果不再是 no_helmet，重置连续计数
                        violation_streak[htid] = 0

            # ── 难例挖掘 ─────────────────────────────────────
            now_hard = time.time()
            if now_hard - last_hard_save >= HARD_SAVE_INTERVAL:
                hard_boxes_for_save = []  # (cls_id, cx, cy, bw, bh, conf)
                img_h, img_w = frame.shape[:2]
                for box in boxes:
                    raw_conf = float(box.conf[0].cpu().numpy())
                    raw_cls  = int(box.cls[0].cpu().numpy())
                    if HARD_CONF_LOW <= raw_conf < HARD_CONF_HIGH:
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                        cx = ((bx1 + bx2) / 2.0) / img_w
                        cy = ((by1 + by2) / 2.0) / img_h
                        bw = (bx2 - bx1) / img_w
                        bh = (by2 - by1) / img_h
                        hard_boxes_for_save.append((raw_cls, cx, cy, bw, bh, raw_conf))
                if hard_boxes_for_save:
                    last_hard_save = now_hard
                    he_name = f"hard_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                    he_img_path = HARD_IMAGES_DIR / f"{he_name}.jpg"
                    he_lbl_path = HARD_LABELS_DIR / f"{he_name}.txt"
                    cv2.imwrite(str(he_img_path), frame)
                    with open(he_lbl_path, "w") as lf:
                        for (hc, hcx, hcy, hbw, hbh, _) in hard_boxes_for_save:
                            lf.write(f"{hc} {hcx:.6f} {hcy:.6f} {hbw:.6f} {hbh:.6f}\n")
                    avg_c = sum(b[5] for b in hard_boxes_for_save) / len(hard_boxes_for_save)
                    try:
                        he_record = HardExampleRecord(
                            timestamp=datetime.now(),
                            image_name=f"{he_name}.jpg",
                            label_name=f"{he_name}.txt",
                            avg_conf=round(avg_c, 4),
                            num_objects=len(hard_boxes_for_save),
                        )
                        db.add(he_record)
                        db.commit()
                    except Exception as e:
                        print(f"[难例] DB 写入失败: {e}")
                        db.rollback()
                    print(f"[难例] 保存难例 {he_name}, {len(hard_boxes_for_save)} objects, avg_conf={avg_c:.3f}")

            # 清理长时间未出现的 tracker ID（防止内存泄漏）
            active_tids = set(h[6] for h in head_dets if h[6] >= 0)
            active_tids.update(v[5] for v in vehicle_dets if v[5] >= 0)
            stale = [tid for tid in track_cls_history if tid not in active_tids]
            for tid in stale:
                del track_cls_history[tid]
                violation_streak.pop(tid, None)
                captured_tids.discard(tid)

            # ── 编码 JPEG 存入共享状态 ────────────────────────────
            ok, buf = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if ok:
                frame_bytes = buf.tobytes()
                inf_state.update(frame_bytes, without_cnt, with_cnt)
                # 缓存本次结果供跳帧时复用
                last_annotated_bytes = frame_bytes
                last_without = without_cnt
                last_with    = with_cnt

            # ── 定期写 DB & 广播 WS ───────────────────────────────
            now = time.time()
            if now - last_save >= SAVE_INTERVAL:
                last_save = now
                ts = datetime.now()

                # 写数据库
                try:
                    record = DetectionRecord(
                        timestamp=ts,
                        without_helmet_count=without_cnt,
                        with_helmet_count=with_cnt,
                    )
                    db.add(record)
                    db.commit()
                except Exception as e:
                    print(f"[推理] DB 写入失败: {e}")
                    db.rollback()

                # 广播 WebSocket（跨线程调度到主事件循环）
                payload = {
                    "timestamp":     ts.strftime("%H:%M:%S"),
                    "without_helmet": without_cnt,
                    "with_helmet":    with_cnt,
                }
                asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast(payload), loop
                )

            # ── 帧率控制 ─────────────────────────────────────────
            elapsed = time.time() - t_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    finally:
        cap.release()
        db.close()
        inf_state.running = False
        print("[推理] 推理线程结束")


# ─────────────────────────────────────────────────────────────
#  6. 生命周期
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    print("[App] FastAPI 启动完成，事件循环已就绪。")
    print("[App] 监控大屏地址: http://127.0.0.1:8000/")
    print("[App] API  文档地址: http://127.0.0.1:8000/docs")


@app.on_event("shutdown")
async def on_shutdown():
    inf_state.running = False
    if inf_state.thread and inf_state.thread.is_alive():
        inf_state.thread.join(timeout=3)
    print("[App] FastAPI 已关闭。")


# ─────────────────────────────────────────────────────────────
#  7. API 路由
# ─────────────────────────────────────────────────────────────

# ── 7.1 视频文件上传 ───────────────────────────────────────────
@app.post("/api/upload-video", summary="上传本地视频文件")
async def upload_video(file: UploadFile = File(...)):
    """
    接收视频文件并保存到 app/uploads/ 目录。
    返回服务端文件名，前端可用此文件名调用 /api/start-inference。
    """
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv"}
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(400, f"不支持的文件格式: {ext}，允许: {allowed_exts}")

    save_path = UPLOAD_DIR / file.filename
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    print(f"[上传] 视频已保存: {save_path} ({len(content)//1024} KB)")
    return {"message": "上传成功", "filename": file.filename}


# ── 7.2 启动推理 ──────────────────────────────────────────────
@app.post("/api/start-inference", summary="启动 YOLO 推理")
async def start_inference(source: str = Query("camera", description="'camera' 或已上传的文件名")):
    """
    - source='camera' → 使用系统默认摄像头 (cv2.VideoCapture(0))
    - source='xxx.mp4' → 使用已上传的视频文件
    """
    global _event_loop

    if inf_state.running:
        return {"message": "推理已在运行中", "source": str(inf_state.source)}

    if source == "camera":
        video_source: str | int = 0
    else:
        video_source = str(UPLOAD_DIR / source)
        if not Path(video_source).exists():
            raise HTTPException(404, f"文件不存在: {source}，请先上传。")

    inf_state.running = True
    inf_state.source  = video_source
    inf_state.thread  = threading.Thread(
        target=_inference_worker,
        args=(video_source, _event_loop),
        daemon=True
    )
    inf_state.thread.start()
    return {"message": "推理已启动", "source": str(video_source)}


# ── 7.3 停止推理 ──────────────────────────────────────────────
@app.post("/api/stop-inference", summary="停止 YOLO 推理")
async def stop_inference():
    if not inf_state.running:
        return {"message": "推理未在运行"}
    inf_state.running = False
    return {"message": "正在停止推理..."}


# ── 7.4 查询可用视频列表 ──────────────────────────────────────
@app.get("/api/videos", summary="获取已上传的视频文件列表")
async def list_videos():
    videos = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    return {"videos": videos}


# ── 7.5 MJPEG 视频流 ──────────────────────────────────────────
def _mjpeg_generator():
    """
    生成器函数：持续读取最新 JPEG 帧，包装成 multipart/x-mixed-replace 格式。
    当没有帧时返回一张占位图。
    """
    placeholder = _make_placeholder()
    while True:
        frame_bytes = inf_state.get_frame()
        if not frame_bytes:
            frame_bytes = placeholder
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )
        time.sleep(0.033)   # ~30 fps 上限


def _make_placeholder() -> bytes:
    """生成一张纯黑占位图，写上提示文字"""
    import numpy as np
    img = np.zeros((360, 640, 3), dtype="uint8")
    cv2.putText(img, "Waiting for stream...", (120, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@app.get("/api/video-feed", summary="MJPEG 实时视频流")
async def video_feed():
    """
    浏览器端使用 <img src="/api/video-feed"> 即可播放实时画面。
    格式: multipart/x-mixed-replace
    """
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ── 7.6 WebSocket 实时数据 ────────────────────────────────────
@app.websocket("/ws/detections")
async def ws_detections(websocket: WebSocket):
    """
    客户端连接后，每次推理写入 DB 时会收到 JSON：
    {
      "timestamp":      "14:23:05",
      "without_helmet": 2,
      "with_helmet":    3
    }
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # 保持连接，等待客户端 ping 或断开
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        ws_manager.disconnect(websocket)


# ── 7.7 历史记录查询 ──────────────────────────────────────────
@app.get("/api/history", summary="获取最近 N 分钟的检测历史")
async def get_history(minutes: int = Query(5, ge=1, le=60)):
    """
    返回最近 `minutes` 分钟内的检测记录列表，用于页面初始化时预填图表。
    """
    since = datetime.now() - timedelta(minutes=minutes)
    db = SessionLocal()
    try:
        records = (
            db.query(DetectionRecord)
            .filter(DetectionRecord.timestamp >= since)
            .order_by(DetectionRecord.timestamp.asc())
            .all()
        )
        data = [
            {
                "timestamp":      r.timestamp.strftime("%H:%M:%S"),
                "without_helmet": r.without_helmet_count,
                "with_helmet":    r.with_helmet_count,
            }
            for r in records
        ]
    finally:
        db.close()

    return {"data": data, "count": len(data)}


# ── 7.8 推理状态查询 ──────────────────────────────────────────
@app.get("/api/status", summary="查询当前推理运行状态")
async def get_status():
    without, with_hm = inf_state.get_counts()
    return {
        "running":        inf_state.running,
        "source":         str(inf_state.source) if inf_state.source is not None else None,
        "without_helmet": without,
        "with_helmet":    with_hm,
    }


# ── 7.9 违章抓拍相关 API ─────────────────────────────────────
@app.get("/api/violations", summary="获取违章记录列表")
async def get_violations(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="按状态筛选: pending/confirmed/dismissed"),
):
    db = SessionLocal()
    try:
        q = db.query(ViolationRecord).order_by(ViolationRecord.timestamp.desc())
        if status:
            q = q.filter(ViolationRecord.status == status)
        total = q.count()
        records = q.offset(offset).limit(limit).all()
        data = [
            {
                "id":         r.id,
                "timestamp":  r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "tracker_id": r.tracker_id,
                "image":      r.image_path,
                "confidence": r.confidence,
                "plate_text": r.plate_text,
                "face_id":    r.face_id,
                "status":     r.status,
            }
            for r in records
        ]
    finally:
        db.close()
    return {"data": data, "total": total}


@app.get("/api/violations/{vid}/image", summary="获取违章抓拍图片")
async def get_violation_image(vid: int):
    db = SessionLocal()
    try:
        rec = db.query(ViolationRecord).filter(ViolationRecord.id == vid).first()
        if not rec:
            raise HTTPException(404, "记录不存在")
        img_path = VIOLATION_DIR / rec.image_path
        if not img_path.exists():
            raise HTTPException(404, "图片文件不存在")
    finally:
        db.close()
    return FileResponse(str(img_path), media_type="image/jpeg")


@app.post("/api/violations/{vid}/recognize", summary="触发 OCR/人脸识别（预留接口）")
async def recognize_violation(vid: int):
    """
    预留接口：将来对接 OCR 车牌识别或人脸识别服务。
    当前返回 mock 占位结果。
    """
    db = SessionLocal()
    try:
        rec = db.query(ViolationRecord).filter(ViolationRecord.id == vid).first()
        if not rec:
            raise HTTPException(404, "记录不存在")
        # --- mock: 实际对接时替换为真实 OCR/人脸识别调用 ---
        rec.plate_text = "待接入OCR"
        rec.face_id    = "待接入FaceID"
        db.commit()
        result = {
            "id":         rec.id,
            "plate_text": rec.plate_text,
            "face_id":    rec.face_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"识别失败: {e}")
    finally:
        db.close()
    return result


@app.put("/api/violations/{vid}/status", summary="更新违章记录状态")
async def update_violation_status(
    vid: int,
    new_status: str = Query(..., description="新状态: confirmed / dismissed"),
):
    if new_status not in ("confirmed", "dismissed"):
        raise HTTPException(400, "状态只能是 confirmed 或 dismissed")
    db = SessionLocal()
    try:
        rec = db.query(ViolationRecord).filter(ViolationRecord.id == vid).first()
        if not rec:
            raise HTTPException(404, "记录不存在")
        rec.status = new_status
        db.commit()
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, str(e))
    finally:
        db.close()
    return {"id": vid, "status": new_status}


# ── 7.10 难例挖掘相关 API ────────────────────────────────────
@app.get("/api/hard-examples/stats", summary="获取难例统计信息")
async def hard_examples_stats():
    db = SessionLocal()
    try:
        total = db.query(HardExampleRecord).count()
        # 最近 24 小时数量
        since_24h = datetime.now() - timedelta(hours=24)
        recent = db.query(HardExampleRecord).filter(
            HardExampleRecord.timestamp >= since_24h
        ).count()
        # 平均置信度
        from sqlalchemy import func
        avg_row = db.query(func.avg(HardExampleRecord.avg_conf)).scalar()
        avg_conf = round(float(avg_row), 4) if avg_row else 0.0
    finally:
        db.close()
    # 磁盘占用
    img_files = list(HARD_IMAGES_DIR.glob("*.jpg"))
    disk_mb = sum(f.stat().st_size for f in img_files) / (1024 * 1024)
    return {
        "total":     total,
        "recent_24h": recent,
        "avg_conf":  avg_conf,
        "disk_mb":   round(disk_mb, 2),
        "conf_range": [HARD_CONF_LOW, HARD_CONF_HIGH],
    }


@app.get("/api/hard-examples/list", summary="获取难例列表")
async def hard_examples_list(
    limit: int = Query(30, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    db = SessionLocal()
    try:
        q = db.query(HardExampleRecord).order_by(HardExampleRecord.timestamp.desc())
        total = q.count()
        records = q.offset(offset).limit(limit).all()
        data = [
            {
                "id":          r.id,
                "timestamp":   r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "image_name":  r.image_name,
                "label_name":  r.label_name,
                "avg_conf":    r.avg_conf,
                "num_objects": r.num_objects,
            }
            for r in records
        ]
    finally:
        db.close()
    return {"data": data, "total": total}


@app.get("/api/hard-examples/{image_name}/image", summary="获取难例图片")
async def hard_example_image(image_name: str):
    img_path = HARD_IMAGES_DIR / image_name
    if not img_path.exists():
        raise HTTPException(404, "图片不存在")
    # 安全检查：确保路径在预期目录内
    if not img_path.resolve().is_relative_to(HARD_IMAGES_DIR.resolve()):
        raise HTTPException(400, "非法路径")
    return FileResponse(str(img_path), media_type="image/jpeg")


@app.post("/api/hard-examples/export", summary="导出难例数据集（打包为zip）")
async def export_hard_examples():
    """将 hard_examples/ 目录打包为 zip 返回，可直接用于增量训练。"""
    import zipfile
    import io

    img_files = sorted(HARD_IMAGES_DIR.glob("*.jpg"))
    lbl_files = sorted(HARD_LABELS_DIR.glob("*.txt"))
    if not img_files:
        raise HTTPException(404, "暂无难例数据")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in img_files:
            zf.write(f, f"images/{f.name}")
        for f in lbl_files:
            zf.write(f, f"labels/{f.name}")
        # 生成 data.yaml
        yaml_content = (
            f"path: ./\n"
            f"train: images\n"
            f"val: images\n"
            f"nc: 3\n"
            f"names: ['two_wheeler', 'without_helmet', 'helmet']\n"
        )
        zf.writestr("data.yaml", yaml_content)
    buf.seek(0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=hard_examples_{ts}.zip"},
    )


# ─────────────────────────────────────────────────────────────
#  8. 前端静态文件挂载
#    必须写在所有 @app.route / @app.websocket 之后
#     这样 FastAPI 内部路由优先于 StaticFiles 通配符
# ─────────────────────────────────────────────────────────────
_frontend_dir = BASE_DIR / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
    print(f"[App] 前端静态文件已挂载: {_frontend_dir}")


# ─────────────────────────────────────────────────────────────
#  9. 直接运行入口
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # 推理线程使用 threading，不兼容 --reload
        log_level="info"
    )
