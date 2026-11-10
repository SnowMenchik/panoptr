$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONHOME = $null
$env:PYTHONPATH = $null
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'

try {
    if (-not [Environment]::Is64BitOperatingSystem) {
        throw 'Нужна 64-битная Windows.'
    }
    Write-Host '[1/5] Подготовка отдельного Python 3.12 x64. Команда py не нужна.'
    Write-Host 'Для загрузки Python и библиотек нужен интернет. Системный Python не изменяется.'

    $data = Join-Path $root 'data'
    $bases = Join-Path $data 'python-bases'
    $envs = Join-Path $data 'envs'
    New-Item -ItemType Directory -Path $bases, $envs -Force | Out-Null
    $id = [Guid]::NewGuid().ToString('N')
    $baseDir = Join-Path $bases ('python312-' + $id)
    $archive = Join-Path $bases ('python312-' + $id + '.zip')
    $url = 'https://api.nuget.org/v3-flatcontainer/python/3.12.10/python.3.12.10.nupkg'
    $expected = '0EB85C2DFCCCCF1B17352DE4C397F69194035B7D37149EACC16F1147D93DE3B8'
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    try {
        Invoke-WebRequest -Uri $url -OutFile $archive -UseBasicParsing
        $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
        if ($actual -ne $expected) { throw 'Контрольная сумма Python не совпадает. Установка остановлена.' }
        Expand-Archive -LiteralPath $archive -DestinationPath $baseDir
    } finally {
        if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue }
    }
    $basePython = Join-Path $baseDir 'tools\python.exe'
    if (-not (Test-Path -LiteralPath $basePython -PathType Leaf)) { throw 'В архиве отсутствует tools\python.exe.' }
    & $basePython --version
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось запустить загруженный Python.' }

    Write-Host '[2/5] Создание изолированного окружения...'
    $envName = 'env-' + $id
    $envDir = Join-Path $envs $envName
    & $basePython -m venv $envDir
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось создать окружение Python.' }
    $python = Join-Path $envDir 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Python окружения не создан.' }

    Write-Host '[3/5] Установка pip...'
    & $python -m pip --isolated install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить pip. Проверьте интернет и повторите установку.' }
    Write-Host '[4/5] Установка библиотек; это может занять несколько минут...'
    & $python -m pip --isolated install -r (Join-Path $root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить библиотеки. Сохраните сообщения pip выше.' }
    Write-Host '[5/5] Проверка импортов OpenCV, PyTorch, Qt и других библиотек...'
    & $python (Join-Path $root 'install_verify.py')
    if ($LASTEXITCODE -ne 0) { throw 'Проверка библиотек не прошла. Сохраните ошибку выше; при ошибках DLL проверьте установленный Microsoft Visual C++ Redistributable x64.' }

    # Activate only after every step succeeded. Existing environments are not changed.
    $relative = 'data\envs\' + $envName + '\Scripts\python.exe'
    $pointer = Join-Path $data 'python-path.txt'
    $pending = Join-Path $data ('python-path-' + $id + '.tmp')
    [IO.File]::WriteAllText($pending, $relative + "`r`n", [Text.Encoding]::ASCII)
    if (Test-Path -LiteralPath $pointer) {
        $backup = Join-Path $data ('python-path-' + $id + '.backup')
        [IO.File]::Replace($pending, $pointer, $backup)
        Remove-Item -LiteralPath $backup -Force -ErrorAction SilentlyContinue
    } else {
        [IO.File]::Move($pending, $pointer)
    }
    Write-Host 'ГОТОВО: среда PANOPTR установлена. Теперь запустите 00_Скачать_модели.bat.' -ForegroundColor Green
    exit 0
} catch {
    Write-Host ('ОШИБКА УСТАНОВКИ: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'Предыдущая активная среда не изменена. Сохраните текст окна и запустите установку снова.'
    exit 1
}
