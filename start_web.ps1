# 智能头盔检测监控大屏 - PowerShell 启动脚本
# 用法: 在项目根目录右键 "用 PowerShell 运行"
# 或终端中执行: .\start_web.ps1

Set-Location $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host ""
Write-Host " ╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host " ║   智能头盔检测监控大屏  启动中...         ║" -ForegroundColor Cyan
Write-Host " ╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── 1. 检查 Python ────────────────────────────────────────────
try {
    $pyVer = python --version 2>&1
    Write-Host "[OK]  Python: $pyVer" -ForegroundColor Green
} catch {
    Write-Host "[错误] 未检测到 Python，请安装后重试。" -ForegroundColor Red
    Read-Host "按 Enter 退出"; exit 1
}

# ── 2. 安装 Web 依赖 ──────────────────────────────────────────
Write-Host ""
Write-Host "[1/3] 安装 Web 依赖 (requirements_web.txt)..." -ForegroundColor Yellow
pip install -r requirements_web.txt -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "[警告] 部分依赖安装失败，请手动执行: pip install -r requirements_web.txt" -ForegroundColor Yellow
}

# ── 3. 检查模型文件 ───────────────────────────────────────────
Write-Host ""
Write-Host "[2/3] 检查模型权重文件..." -ForegroundColor Yellow
$modelPath = "my_baseTraining_runs\WiseIou_DySampleV1\weights\best.pt"
if (-not (Test-Path $modelPath)) {
    Write-Host "[错误] 模型不存在: $modelPath" -ForegroundColor Red
    Write-Host "       请确认训练已完成，或修改 app\main.py 中的 MODEL_PATH 变量。" -ForegroundColor Red
    Read-Host "按 Enter 退出"; exit 1
}
Write-Host "[OK]  模型文件存在: $modelPath" -ForegroundColor Green

# ── 4. 启动服务 ───────────────────────────────────────────────
Write-Host ""
Write-Host "[3/3] 启动 FastAPI 后端服务..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  ► 监控大屏: " -NoNewline; Write-Host "http://127.0.0.1:8000/" -ForegroundColor Cyan
Write-Host "  ► API 文档: " -NoNewline; Write-Host "http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Ctrl+C 可停止服务" -ForegroundColor DarkGray
Write-Host "─────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ""

# 以模块方式运行，确保相对路径正确
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
