@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
set "HERE=%~dp0"
cd /d "%HERE%"

:: Ensure runtime and binary model folders are linked inside THIS branch folder (%HERE%)
if not exist "%HERE%runtime\python.exe" (
    powershell -NoProfile -Command ^
        "$here = '%HERE:~0,-1%'; " ^
        "$roots = @((Split-Path $here -Parent), 'D:\PANOPTR', 'C:\PANOPTR') | Select-Object -Unique; " ^
        "$src = $null; " ^
        "foreach ($r in $roots) { if (Test-Path $r) { " ^
        "  $cand = Get-ChildItem -Path $r -Filter 'python.exe' -Recurse -Depth 3 -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like '*runtime\python.exe' -and $_.FullName -notlike ($here + '*') -and $_.Length -gt 10000 } | Select-Object -First 1; " ^
        "  if ($cand) { $src = Split-Path (Split-Path $cand.FullName -Parent) -Parent; break } " ^
        "} } " ^
        "if ($src) { " ^
        "  Write-Host ('[PANOPTR] Подключение бинарных компонентов (runtime и весов .pt) в текущую ветку ' + $here); " ^
        "  $dirs = @('runtime', 'examples', '02_Реализация\panoptr\models', '02_Реализация\panoptr\dataset_lab\annotation_vendor', '08_Обучение\models', '09_Дополнительные_данные_2026-10-11', '10_Лаборатория_данных\annotation_models'); " ^
        "  foreach ($d in $dirs) { " ^
        "    $s = Join-Path $src $d; $t = Join-Path $here $d; " ^
        "    if ((Test-Path $s) -and -not (Test-Path $t)) { " ^
        "      $parent = Split-Path $t -Parent; if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }; " ^
        "      cmd /c mklink /J `"$t`" `"$s`" | Out-Null " ^
        "    } " ^
        "  } " ^
        "  foreach ($pt in @('yolo26n.pt', 'yolo26s-seg.pt', 'yolo26n-pose.pt')) { " ^
        "    $sp = Join-Path $src ('02_Реализация\panoptr\' + $pt); $tp = Join-Path $here ('02_Реализация\panoptr\' + $pt); " ^
        "    if ((Test-Path $sp) -and -not (Test-Path $tp)) { Copy-Item $sp $tp -Force } " ^
        "  } " ^
        "}"
)

set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH="
set "PYTHONHOME=%HERE%runtime"
cd /d "%HERE%"
"%HERE%runtime\python.exe" -s -E "%HERE%start.py" main %*
if errorlevel 1 pause
