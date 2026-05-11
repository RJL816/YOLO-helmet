@echo off
cd /d "e:\Desktop\yolo\HelmetDetect"
echo Starting YOLO Helmet Detection Backend...
echo Using environment: yolov11
"E:\Anaconda\Scripts\conda.exe" run -p "D:\Anaconda\envs\yolov11" python backend_simple.py
pause
