@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
call "%~dp0find_environment.bat"
if errorlevel 1 (pause & exit /b 1)
"%PANOPTR_PYTHON%" -s "%~dp0download_models.py"
pause
