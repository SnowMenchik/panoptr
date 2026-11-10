@echo off
set "PANOPTR_PYTHON="
if exist "%~dp0data\python-path.txt" set /p PANOPTR_PYTHON=<"%~dp0data\python-path.txt"
if not defined PANOPTR_PYTHON (
    echo Run 00_Установить_среду.bat first.
    exit /b 1
)
set "PANOPTR_PYTHON=%~dp0%PANOPTR_PYTHON%"
if not exist "%PANOPTR_PYTHON%" (
    echo Python environment not found. Run 00_Установить_среду.bat again.
    exit /b 1
)
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
exit /b 0
