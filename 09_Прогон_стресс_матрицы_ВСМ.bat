@echo off
chcp 65001 >nul
setlocal
set "HERE=%~dp0"
cd /d "%HERE%"

if not exist "%HERE%runtime\python.exe" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%link_runtime.ps1"
)

set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH="
set "PYTHONHOME=%HERE%runtime"
cd /d "%HERE%"
"%HERE%runtime\python.exe" -s -E "%HERE%start.py" stress_matrix %*
pause
