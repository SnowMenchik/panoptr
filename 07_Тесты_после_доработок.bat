@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
set "HERE=%~dp0"
set "RUN_ROOT=%HERE:~0,-1%"
set "PY_EXE=%RUN_ROOT%\runtime\python.exe"

set "VALID_PY=0"
if exist "%PY_EXE%" (
    for %%A in ("%PY_EXE%") do if %%~zA GTR 10000 set "VALID_PY=1"
)

if "%VALID_PY%"=="0" (
    for %%D in ("%HERE%..\PANOPTR_FINAL_2026-10-31" "D:\PANOPTR\PANOPTR_FINAL_2026-10-31" "C:\PANOPTR\PANOPTR_FINAL_2026-10-31") do (
        if "!VALID_PY!"=="0" if exist "%%~fD\runtime\python.exe" (
            set "RUN_ROOT=%%~fD"
            set "PY_EXE=%%~fD\runtime\python.exe"
            set "VALID_PY=1"
        )
    )
)

if "%VALID_PY%"=="0" (
    for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command ^
        "$here = '%HERE:~0,-1%'; " ^
        "$roots = @((Split-Path $here -Parent), 'D:\PANOPTR', 'D:\', 'C:\') | Select-Object -Unique; " ^
        "foreach ($r in $roots) { if (Test-Path $r) { " ^
        "  $cand = Get-ChildItem -Path $r -Filter 'python.exe' -Recurse -Depth 3 -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like '*runtime\python.exe' -and $_.Length -gt 10000 } | Select-Object -First 1; " ^
        "  if ($cand) { Write-Output (Split-Path (Split-Path $cand.FullName -Parent) -Parent); break } " ^
        "} }"`) do (
        set "RUN_ROOT=%%F"
        set "PY_EXE=%%F\runtime\python.exe"
        set "VALID_PY=1"
    )
)

if "%VALID_PY%"=="1" (
    if /I not "%RUN_ROOT%"=="%HERE:~0,-1%" (
        echo [PANOPTR] Используется рабочая сборка: !RUN_ROOT!
        echo [PANOPTR] Синхронизация исправленного кода в !RUN_ROOT!\02_Реализация\panoptr ...
        for %%M in (main.py final_capture.py finale_ui.py path_tracking.py test_final.py) do (
            if exist "%HERE%02_Реализация\panoptr\%%M" copy /Y "%HERE%02_Реализация\panoptr\%%M" "!RUN_ROOT!\02_Реализация\panoptr\%%M" >nul
        )
    )
    set "PYTHONNOUSERSITE=1"
    set "PYTHONUTF8=1"
    set "PYTHONDONTWRITEBYTECODE=1"
    set "PYTHONPATH="
    set "PYTHONHOME=!RUN_ROOT!\runtime"
    cd /d "!RUN_ROOT!"
    "%PY_EXE%" -s -E "%RUN_ROOT%\verify_development.py"  %*
    if errorlevel 1 pause
    exit /b 0
)

echo [!] Не найдена папка runtime\python.exe.
pause
