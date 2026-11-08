@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
set "HERE=%~dp0"
set "RUN_ROOT=%HERE:~0,-1%"
set "PY_EXE=%RUN_ROOT%\runtime\python.exe"

:: Check if runtime\python.exe exists and is not a 130-byte Git LFS pointer from GitHub Download ZIP
set "VALID_PY=0"
if exist "%PY_EXE%" (
    for %%A in ("%PY_EXE%") do if %%~zA GTR 10000 set "VALID_PY=1"
)

if "%VALID_PY%"=="0" (
    echo [PANOPTR] В текущей папке нет бинарного runtime\python.exe ^(скачан ZIP только с кодом^).
    echo [PANOPTR] Поиск рабочей сборки PANOPTR_FINAL_2026-10-31 на компьютере и обновление кода...
    for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command ^
        "$here = '%HERE:~0,-1%'; " ^
        "$roots = @((Split-Path $here -Parent), (Join-Path $env:USERPROFILE 'Desktop'), (Join-Path $env:USERPROFILE 'Downloads'), 'C:\', 'D:\') | Select-Object -Unique; " ^
        "$found = $null; " ^
        "foreach ($r in $roots) { if (Test-Path $r) { " ^
        "  $cand = Get-ChildItem -Path $r -Filter 'python.exe' -Recurse -Depth 4 -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like '*runtime\python.exe' -and $_.Length -gt 10000 } | Select-Object -First 1; " ^
        "  if ($cand) { $found = Split-Path (Split-Path $cand.FullName -Parent) -Parent; break } " ^
        "} } " ^
        "if ($found) { Write-Output $found }"`) do (
        set "RUN_ROOT=%%F"
    )
    if exist "!RUN_ROOT!\runtime\python.exe" (
        echo [PANOPTR] Найдена рабочая сборка: !RUN_ROOT!
        echo [PANOPTR] Копирование исправленных файлов в !RUN_ROOT!\02_Реализация\panoptr ...
        for %%M in (main.py final_capture.py finale_ui.py path_tracking.py test_final.py) do (
            if exist "%HERE%02_Реализация\panoptr\%%M" copy /Y "%HERE%02_Реализация\panoptr\%%M" "!RUN_ROOT!\02_Реализация\panoptr\%%M" >nul
        )
        set "PY_EXE=!RUN_ROOT!\runtime\python.exe"
        set "VALID_PY=1"
    )
)

if "%VALID_PY%"=="1" (
    set "PYTHONNOUSERSITE=1"
    set "PYTHONUTF8=1"
    set "PYTHONDONTWRITEBYTECODE=1"
    set "PYTHONPATH="
    set "PYTHONHOME=%RUN_ROOT%\runtime"
    cd /d "%RUN_ROOT%"
    "%PY_EXE%" -s -E "%RUN_ROOT%\verify_development.py"  %*
    if errorlevel 1 pause
    exit /b 0
)

:: Fallback to system python if installed
where python >nul 2>nul
if %errorlevel%==0 (
    echo [PANOPTR] Запуск через системный Python...
    cd /d "%HERE%02_Реализация\panoptr"
    python launch_final.py %*
    if errorlevel 1 pause
    exit /b 0
)

echo [!] Не найдена папка runtime с Python. Скопируйте файлы из 02_Реализация\panoptr в вашу рабочую папку PANOPTR_FINAL_2026-10-31\02_Реализация\panoptr.
pause
