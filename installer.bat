@echo off
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo === Installation Sonic Cam (Python 3.12 requis pour MediaPipe) ===
if not exist .venv (
    py -3.12 -m venv .venv || (echo Python 3.12 introuvable & pause & exit /b 1)
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe prepare_sonic.py
echo.
echo Termine. Lancez lancer_sonic_cam.bat
pause
