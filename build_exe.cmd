@echo off
REM 构建独立可执行的 RicardoAgent.exe（单文件，双击/命令行均可）
REM 前置: pip install pyinstaller
python -m PyInstaller --onefile --name RicardoAgent ^
  --hidden-import matplotlib --hidden-import scipy --hidden-import sklearn ^
  --clean -y ricardo_app.py
echo.
echo 产物: dist\RicardoAgent.exe
