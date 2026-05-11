param([switch]$background)

# 切换到项目目录
Set-Location "e:\Desktop\yolo\HelmetDetect"

# 激活 conda 环境并运行后端
& conda.exe run -n yolov11 python backend_simple.py
