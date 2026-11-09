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
    'D:\PANOPTR',
    'C:\PANOPTR'
) | Select-Object -Unique

$src = $null
foreach ($r in $roots) {
    if (Test-Path $r) {
        $cand = Get-ChildItem -Path $r -Filter 'python.exe' -Recurse -Depth 4 -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like '*runtime\python.exe' -and $_.FullName -notlike ($here + '*') -and $_.Length -gt 10000 } |
            Select-Object -First 1
        if ($cand) {
            $src = Split-Path (Split-Path $cand.FullName -Parent) -Parent
            break
        }
    }
}

if (-not $src) {
    Write-Host '[!] Не найдена папка с бинарным runtime\python.exe.' -ForegroundColor Red
    exit 1
}

Write-Host ('[PANOPTR] Подключение бинарных компонентов (runtime и весов .pt) в текущую ветку: ' + $here)

foreach ($d in $dirs) {
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
