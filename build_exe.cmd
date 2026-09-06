@echo off
REM 构建独立可执行文件（单文件，双击/命令行均可）
REM 前置: pip install pyinstaller

REM RicardoAgent.exe —— 终端版（CLI，含全部子命令）
python -m PyInstaller --onefile --name RicardoAgent ^
  --hidden-import matplotlib --hidden-import scipy --hidden-import sklearn ^
  --clean -y ricardo_app.py

REM RicardoAgentGUI.exe —— 桌面版（无控制台黑窗，双击直接开聊天窗口）
python -m PyInstaller --onefile --noconsole --name RicardoAgentGUI ^
  --hidden-import matplotlib --hidden-import scipy --hidden-import sklearn ^
  --clean -y ricardo_gui_app.py

echo.
echo 产物: dist\RicardoAgent.exe（终端版）, dist\RicardoAgentGUI.exe（桌面版）
