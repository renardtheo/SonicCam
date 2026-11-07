@echo off
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
rem MODE DEBUG : aucune limite de fps, webcam au max, mesures a l ecran (touche D) et dans la console.
rem Pour l usage normal (Teams / Discord), utilisez lancer_sonic_cam.bat
.venv\Scripts\python.exe sonic_cam.py --debug %*
pause
