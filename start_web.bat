@echo off
chcp 65001 > nul
title 智能头盔检测监控大屏

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║     智能头盔检测监控大屏  启动中...       ║
echo  ╚══════════════════════════════════════════╝
echo.

:: ── 切换到项目根目录 ──────────────────────────────────────────
cd /d "%~dp0"
echo [路径] 当前目录: %CD%

:: ── 检查 Conda 虚拟环境 ──────────────────────────────────────
where conda >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [INFO] 检测到 Conda，尝试激活 yolov11 环境...
    call conda activate yolov11 >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo [OK]  已激活 Conda 环境: yolov11
    ) else (
        echo [WARN] 未找到 yolov11 环境，使用当前 Python
    )
)

:: ── 检查 Python ───────────────────────────────────────────────
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [错误] 未检测到 Python，请先安装并加入 PATH。
    echo        推荐: conda create -n yolov11 python=3.10
    pause & exit /b 1
)

:: ── 安装依赖（首次运行或依赖有更新时执行）────────────────────────
echo.
echo [1/3] 正在检查并安装 Web 依赖...
pip install -r requirements_web.txt -q
if %ERRORLEVEL% neq 0 (
    echo [警告] 部分依赖安装失败，请手动执行:
    echo       pip install -r requirements_web.txt
)

:: ── 安装本地自定义 ultralytics（WiseIoU/DySample 等改进）───────
echo [INFO] 安装本地自定义 ultralytics 源码...
pip install -e . -q >nul 2>&1

:: ── 检查模型文件 ──────────────────────────────────────────────
echo.
echo [2/3] 检查模型文件...
set MODEL=my_baseTraining_runs\WiseIou_DySampleV1\weights\best.pt
if not exist "%MODEL%" (
    echo [错误] 模型文件不存在: %MODEL%
    echo        请确认训练已完成或手动修改 app\main.py 中的 MODEL_PATH。
    pause & exit /b 1
) else (
    echo [OK]  模型文件存在: %MODEL%
)

:: ── 启动 FastAPI 后端 ─────────────────────────────────────────
echo.
echo [3/3] 启动 FastAPI 后端（端口 8000）...
echo.
echo  ► 服务地址： http://127.0.0.1:8000
echo  ► 监控大屏： http://127.0.0.1:8000/index.html
echo  ► API 文档： http://127.0.0.1:8000/docs
echo.
echo  提示：按 Ctrl+C 可停止服务
echo ─────────────────────────────────────────────────────────────
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
