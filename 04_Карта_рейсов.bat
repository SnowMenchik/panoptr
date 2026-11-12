@echo off
chcp 65001 >nul
setlocal
set "HERE=%~dp0"
cd /d "%HERE%"
call "%HERE%find_environment.bat"
if errorlevel 1 (
    pause
    exit /b 1
)
"%PANOPTR_PYTHON%" -s "%HERE%start.py" routes %*
if errorlevel 1 pause
