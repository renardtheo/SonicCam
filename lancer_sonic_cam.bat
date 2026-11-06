@echo off
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
rem Options utiles : --camera 1  --min-gap 300 --max-gap 900 (secondes)  --size 1.0  --slap-ratio 0.5  --mirror
.venv\Scripts\python.exe sonic_cam.py %*
pause
