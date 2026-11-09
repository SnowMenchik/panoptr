# PANOPTR — Инструкция по установке и запуску рабочей ветки на ПК

Репозиторий: `https://github.com/SnowMenchik/q.git`  
Рабочие ветки с поддержкой двух камер C920 и обновлениями к финалу:
- `fix/dual-camera-support` — чистая техническая ветка (исправленный код двух камер, UI, 9 `.bat`-лаунчеров, `requirements.txt`, автотесты).
- `feat/vsm-max-score-package` — полная ветка со стресс-матрицей ВСМ (0–400 км/ч) и пакетом документов.

---

## Почему при загрузке использовался Git LFS и как правильно скачивать проект

При первой отправке проекта из папки `C:\Users\stank\Downloads\2` (объём **6,25 ГБ**) использовалась команда:
```bat
git lfs migrate import --above=50MB --everything
```
Из-за этого файлы разделились на две категории:
1. **Код (`.py`, `.bat`, конфиги) и все 19 обученных моделей YOLO (`.pt`, по 5–23 МБ каждая)** — весят **меньше 50 МБ**, поэтому хранятся в **обычном Git** и скачиваются без ограничений.
2. **22 тяжёлых бинарных файла (> 50 МБ)** внутри `runtime\` (`torch_cuda.dll`, `torch_cpu.dll`, `cublas*.dll`, `cudnn*.dll`) и `examples\` (`drone.avi`, `rail.avi`) — попали в **Git LFS (5,2 ГБ)**, что превышает бесплатную квоту GitHub LFS (1 ГБ/мес).

Поэтому есть **два быстрых и надёжных способа** развернуть ветку на ПК.

---

## Способ 1 (За 5 секунд, без скачивания гигабайт из интернета) — если на ПК есть исходная папка `C:\Users\stank\Downloads\2` или `D:\PANOPTR`

### Шаг 1. Скачайте лёгкую ветку кода в любую пустую папку
Откройте `cmd.exe` и выполните:
```bat
mkdir "C:\Users\stank\OneDrive\Рабочий стол\panoptr"
cd /d "C:\Users\stank\OneDrive\Рабочий стол\panoptr"
git init
git remote add origin https://github.com/SnowMenchik/q.git
git fetch --depth 1 origin fix/dual-camera-support
git checkout -f -B fix/dual-camera-support FETCH_HEAD
```

### Шаг 2. Подключите бинарные папки (`runtime`, `models`, `examples`) из `C:\Users\stank\Downloads\2`
Выполните в том же окне `cmd` одну команду:
```bat
powershell -NoProfile -ExecutionPolicy Bypass -Command "$here=(Get-Location).Path; $src='C:\Users\stank\Downloads\2\PANOPTR_FINAL_2026-10-31'; if (-not (Test-Path $src)) { $src='C:\Users\stank\Downloads\2' }; git clean -fd; foreach ($d in 'runtime','examples','02_Реализация\panoptr\models','02_Реализация\panoptr\dataset_lab\annotation_vendor','08_Обучение\models','08_Обучение\runs','09_Дополнительные_данные_2026-10-11','10_Лаборатория_данных\annotation_models') { $s=Join-Path $src $d; $t=Join-Path $here $d; if (Test-Path $t) { cmd /c rmdir "$t" 2>$null }; if ((Test-Path $s) -and -not (Test-Path $t)) { New-Item -ItemType Directory -Path (Split-Path $t -Parent) -Force | Out-Null; New-Item -ItemType Junction -Path $t -Target $s -Force | Out-Null } }; foreach ($pt in 'yolo26n.pt','yolo26s-seg.pt','yolo26n-pose.pt') { $sp=Join-Path $src ('02_Реализация\panoptr\'+$pt); $tp=Join-Path $here ('02_Реализация\panoptr\'+$pt); if ((Test-Path $sp) -and -not (Test-Path $tp)) { Copy-Item $sp $tp -Force } }; Write-Host 'ГОТОВО! Все компоненты подключены!' -ForegroundColor Green"
```

### Шаг 3. Запустите проверку и приложение
```bat
05_Проверить_сборку.bat
07_Тесты_после_доработок.bat
06_Полигон_и_финал.bat
```

---

## Способ 2 (На новом ПК, где нет папки `Downloads\2`) — через системный Python + `requirements.txt`

Если вы клонируете ветку на другой ноутбук, где нет старого архива:

### Шаг 1. Скачайте ветку `fix/dual-camera-support`
```bat
git init
git remote add origin https://github.com/SnowMenchik/q.git
git fetch --depth 1 origin fix/dual-camera-support
git checkout -f -B fix/dual-camera-support FETCH_HEAD
```

### Шаг 2. Установите зависимости в системный Python и привяжите его как `runtime`
```bat
pip install -r requirements.txt
for /f "delims=" %i in ('python -c "import sys; print(sys.prefix)"') do mklink /J runtime "%i"
```

### Шаг 3. Подтяните все 19 обученных весов `.pt` из репозитория (они < 50 МБ и не требуют LFS)
```bat
git fetch --depth 1 origin main
git checkout origin/main -- PANOPTR_FINAL_2026-10-31/02_Реализация/panoptr/models PANOPTR_FINAL_2026-10-31/02_Реализация/panoptr/yolo26n.pt PANOPTR_FINAL_2026-10-31/02_Реализация/panoptr/yolo26s-seg.pt PANOPTR_FINAL_2026-10-31/02_Реализация/panoptr/yolo26n-pose.pt
xcopy /E /I /Y "PANOPTR_FINAL_2026-10-31\02_Реализация\panoptr\*" "02_Реализация\panoptr\"
rmdir /S /Q PANOPTR_FINAL_2026-10-31
```

### Шаг 4. Прогоните тесты разработки
```bat
07_Тесты_после_доработок.bat
```
Оба набора тестов (`application` и `data_lab`) выведут `PASS`.

---

## Как подтягивать свежие изменения из ветки Git
Когда в ветке `fix/dual-camera-support` появляются обновления, выполните в папке проекта:
```bat
git fetch origin fix/dual-camera-support
git reset --hard origin/fix/dual-camera-support
```

---

## Назначение всех 9 `.bat`-файлов в корне ветки
| Файл | Назначение |
|---|---|
| `01_Запустить_CPU.bat` | Запуск главного двухканального окна в режиме CPU |
| `02_Запустить_GPU.bat` | Запуск главного двухканального окна в режиме GPU (CUDA) |
| `03_Лаборатория_данных.bat` | Лаборатория датасетов, авторазметки и обучения YOLO |
| `04_Карта_рейсов.bat` | Накопительная карта дефектов пути по рейсам и километражу |
| `05_Проверить_сборку.bat` | Самопроверка переносной среды `runtime`, CUDA и моделей (`data\last_check.json`) |
| `06_Полигон_и_финал.bat` | Главное окно финала: события, репетиция отказов, расчёт камеры, 3D-стенд, симулятор ВСМ |
| `07_Тесты_после_доработок.bat` | Запуск всех 210 unit-тестов (`verify_development.py` -> `PASS application`, `PASS data_lab`) |
| `08_Диагностика_камер.bat` | Проверка подключённых физических камер DirectShow / MSMF |
| `09_Прогон_стресс_матрицы_ВСМ.bat` | Автоматический прогон 5 скоростных сценариев ВСМ (100–400 км/ч) |
