$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root
$env:PYTHONHOME = $null
$env:PYTHONPATH = $null
$env:PYTHONNOUSERSITE = '1'
Write-Host 'Создание новой изолированной среды PANOPTR (Python 3.12 или 3.11 x64). Нужен интернет.'
$version = $null
foreach ($v in @('3.12', '3.11')) {
    & py "-$v" -c 'import sys,struct; sys.exit(0 if struct.calcsize("P")==8 else 1)' 2>$null
    if ($LASTEXITCODE -eq 0) { $version = $v; break }
}
if (-not $version) { Write-Host 'Установите Python 3.12 x64 с python.org и запустите установку снова.'; exit 1 }
$envName = 'env-' + [Guid]::NewGuid().ToString('N')
$envDir = Join-Path $root ('data\envs\' + $envName)
& py "-$version" -m venv $envDir
if ($LASTEXITCODE -ne 0) { Write-Host 'Не удалось создать окружение.'; exit 1 }
$python = Join-Path $envDir 'Scripts\python.exe'
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Host 'Не удалось обновить pip.'; exit 1 }
& $python -m pip install -r (Join-Path $root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { Write-Host 'Установка зависимостей не завершена. Старая среда не изменена.'; exit 1 }
& $python -c 'import cv2,numpy,torch,torchvision,ultralytics,PySide6; assert hasattr(cv2,"VideoCapture"); print("OpenCV", cv2.__version__, cv2.__file__)'
if ($LASTEXITCODE -ne 0) { Write-Host 'Проверка библиотек не прошла. Старая среда не изменена.'; exit 1 }
# Only switch after all checks passed. Never alter existing runtime or global packages.
$relative = 'data\envs\' + $envName + '\Scripts\python.exe'
[IO.File]::WriteAllText((Join-Path $root 'data\python-path.txt'), $relative + "`r`n", [Text.Encoding]::ASCII)
Write-Host 'Среда готова. Для загрузки весов запустите 00_Скачать_модели.bat.'
exit 0
