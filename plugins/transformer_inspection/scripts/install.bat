@echo off
REM Windows 一键安装脚本

echo ==========================================
echo 输变电激光星芒破夜绘明监测平台 - 安装脚本
echo ==========================================

REM 检查Python
python --version
if errorlevel 1 (
    echo 错误: 未找到Python,请先安装Python 3.10+
    pause
    exit /b 1
)

REM 创建虚拟环境
if not exist "venv" (
    echo 创建虚拟环境...
    python -m venv venv
)

REM 激活虚拟环境
call venv\Scripts\activate.bat

REM 安装依赖
echo 安装依赖...
pip install --upgrade pip
pip install -r requirements.txt

REM 创建必要目录
echo 创建目录结构...
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "evidence\runs" mkdir evidence\runs
if not exist "evidence\exports" mkdir evidence\exports

echo.
echo ==========================================
echo 安装完成!
echo.
echo 启动命令:
echo   venv\Scripts\activate.bat
echo   python run.py
echo.
echo 访问地址: http://127.0.0.1:8080
echo ==========================================
pause
