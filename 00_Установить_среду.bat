@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_environment.ps1"
if errorlevel 1 echo Installation failed. See messages above.
pause
