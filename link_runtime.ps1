$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot

$dirs = @(
    'runtime',
    'examples',
    '02_Реализация\panoptr\models',
    '02_Реализация\panoptr\dataset_lab\annotation_vendor',
    '08_Обучение\models',
    '08_Обучение\runs',
    '09_Дополнительные_данные_2026-10-11',
    '10_Лаборатория_данных\annotation_models'
)

$pts = @('yolo26n.pt', 'yolo26s-seg.pt', 'yolo26n-pose.pt')

$needLink = $false
foreach ($d in $dirs) {
    if (-not (Test-Path (Join-Path $here $d))) { $needLink = $true; break }
}
foreach ($pt in $pts) {
    if (-not (Test-Path (Join-Path $here ('02_Реализация\panoptr\' + $pt)))) { $needLink = $true; break }
}
if (-not $needLink) { exit 0 }

$roots = @(
    (Split-Path $here -Parent),
    (Join-Path $env:USERPROFILE 'Downloads\2\PANOPTR_FINAL_2026-10-31'),
    (Join-Path $env:USERPROFILE 'Downloads\2'),
    (Join-Path $env:USERPROFILE 'Downloads'),
    'D:\PANOPTR',
    'C:\PANOPTR'
) | Select-Object -Unique

$src = $null
foreach ($r in $roots) {
    if (Test-Path $r) {
        $cand = Get-ChildItem -Path $r -Filter 'rail9_finetuned_2026.pt' -Recurse -Depth 5 -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notlike ($here + '*') -and $_.Length -gt 100000 } |
            Select-Object -First 1
        if ($cand) {
            $src = Split-Path (Split-Path (Split-Path (Split-Path $cand.FullName -Parent) -Parent) -Parent) -Parent
            break
        }
    }
}

if ($src) {
    Write-Host ('[PANOPTR] Подключение весов .pt и данных из: ' + $src) -ForegroundColor Cyan
    foreach ($d in $dirs) {
        if ($d -eq 'runtime') {
            $cv2Pyd = Join-Path $src 'runtime\Lib\site-packages\cv2\cv2.pyd'
            if (-not ((Test-Path $cv2Pyd) -and ((Get-Item $cv2Pyd).Length -gt 10000))) {
                continue
            }
        }
        $s = Join-Path $src $d
        $t = Join-Path $here $d
        if ((Test-Path $s) -and -not (Test-Path $t)) {
            $parent = Split-Path $t -Parent
            if (-not (Test-Path $parent)) {
                New-Item -ItemType Directory -Path $parent -Force | Out-Null
            }
            New-Item -ItemType Junction -Path $t -Target $s -Force | Out-Null
        }
    }
    foreach ($pt in $pts) {
        $sp = Join-Path $src ('02_Реализация\panoptr\' + $pt)
        $tp = Join-Path $here ('02_Реализация\panoptr\' + $pt)
        if ((Test-Path $sp) -and -not (Test-Path $tp)) {
            Copy-Item -Path $sp -Destination $tp -Force
        }
    }
}

$rtTarget = Join-Path $here 'runtime'
if (-not (Test-Path (Join-Path $rtTarget 'python.exe'))) {
    try {
        $sysPy = (& python -c "import sys; print(sys.prefix)" 2>$null)
        if ($sysPy -and (Test-Path (Join-Path $sysPy 'python.exe'))) {
            Write-Host ('[PANOPTR] Подключение системного Python в runtime: ' + $sysPy) -ForegroundColor Green
            New-Item -ItemType Junction -Path $rtTarget -Target $sysPy -Force | Out-Null
        }
    } catch {}
}

if (-not (Test-Path (Join-Path $rtTarget 'python.exe'))) {
    Write-Host '[!] Не найдена папка с бинарным runtime\python.exe.' -ForegroundColor Red
    exit 1
}
