@echo off
chcp 65001 >nul
setlocal
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH="
set "PYTHONHOME=%~dp0runtime"
cd /d "%~dp0"
"%~dp0runtime\python.exe" -s -E "%~dp0start.py" check %*
if errorlevel 1 pause
