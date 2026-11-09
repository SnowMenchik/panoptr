# Источники и условия использования

1. Svanström, Englund, Alonso-Fernandez: *A Dataset for Multi-Sensor Drone Detection* (2021). Официальные видео и MATLAB-рамки: https://github.com/DroneDetectionThesis/Drone-detection-dataset . Репозиторий помечен CC0-1.0; копия LICENSE сохранена в sources/svanstrom_LICENSE.txt и исходном архиве. Скачан полный снимок; RGB-набор использует видимый канал, IR и аудио сохранены отдельно в raw и не входят в обучение.
2. Aksoy, Orak, Özkan, Selimoğlu: *Drone Dataset: Amateur Unmanned Air Vehicle Detection*, версия4, DOI10.17632/zcsj2g2m4c.4: https://data.mendeley.com/datasets/zcsj2g2m4c/4 . Авторская страница указывает CC BY4.0: https://creativecommons.org/licenses/by/4.0/ . Прямой HTTP-доступ к Mendeley здесь вернул403; файлы получены из публичного зеркала https://github.com/tarlanahad/drone-detection-dataset , раздел Mendeley. В обучении оставлены только положительные обычные фотографии; пустые метки и все видеокадры исключены после визуальной проверки. Исходники сохранены. Происхождение отдельных интернет-изображений полностью восстановить нельзя.
3. Mehdi Özel: *Drone Dataset (UAV)*: https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav . Автор разрешает академическое/любительское использование и просит ссылку на источник. API указывает «Data files © Original Authors», а не CC0/CC BY. Эти условия сохранены в sources/kaggle_meta.json. Используется YOLO-версия из того же зеркала tarlanahad; дублирующая XML-копия сохранена в raw и повторно не добавлялась. Пакет предназначен для учебного проекта/исследования; коммерческое использование и публичное распространение всего объединённого набора этими условиями не подтверждены.

Зеркало не создаёт новую лицензию на исходные изображения. При публикации результатов укажите всех авторов и ссылки выше. Не публикуйте весь объединённый архив как собственный датасет.

Декодер MCOS: Burak Ozdemir, https://github.com/bozdemir/mcos-decoder , MIT. Исходный код и LICENSE сохранены в raw/mcos_decoder. Используется его MatStream; чтение RGB-таблицы реализовано отдельно в scripts/prepare_drone.py, чтобы выбирать именованный столбец DRONE, обходить лишнюю мета-запись и сохранять все Nx4 рамки. Библиотеки в рабочем окружении обучения09 не устанавливались и не заменялись.

Снимки GitHub закреплены SHA коммитов, архивы проверены CRC ZIP и SHA256. Полные URL, хеши и размеры: sources/downloads.json; ревизии: sources/*.revision.json.

Сведения о настройках обучения: https://docs.ultralytics.com/modes/train/ . Используется уже установленная версия Ultralytics проекта и локальный общий COCO-чекпойнт YOLO26s, а не новая установка библиотек.


## Дополнение 17.10.2026: новые данные

- Yaren Koç, Fixed-Wing UAV Detection Dataset and Model: https://github.com/yarenk/fixed-wing-uav-dataset-and-model . Полный опубликованный dataset.zip выпуска version1 сохранён. Автор указывает для датасета CC BY4.0; ссылка и имя автора обязательны. Код/веса автор отличает от данных и указывает MIT. Чужие best.pt не загружались и не исполнялись. Снимок README и SHA архива сохранены в sources.
- Jie Zhao, Jingshu Zhang, Dongdong Li, Dong Wang. Vision-based Anti-UAV Detection and Tracking, IEEE TITS2022: https://github.com/wangdongdut/DUT-Anti-UAV , https://arxiv.org/abs/2205.10851 . Скачаны полностью три официальных архива detection train/val/test с Google Drive. Tracking не загружался. В проверенном README нет явной универсальной лицензии на перераспространение датасета: не обозначать набор как MIT/CC0 и не переносить лицензию другой реализации на данные. Опубликованный исследовательский набор использован для подготовки конкурсного эксперимента; коммерческие права не подтверждены.
- Hanqing Guo, Ye Zheng, Yin Zhang, Zhi Gao, Shiyu Zhao. Global-Local MAV Detection under Challenging Conditions based on Appearance and Motion, IEEE TITS2024: https://github.com/WindyLab/Global-Local-MAV-Detection , https://arxiv.org/abs/2312.11008 . Из официального открытого ARD-MAV.zip получены полные8 выбранных видео и все XML этих последовательностей. Доступ публичный; форма/авторизация не обходилась. Весь14.6GB архив и остальные52 серии не скачивались. Условия исследовательского использования и ссылка на авторов сохранены; явную лицензию на свободное перераспространение всего датасета в README подтвердить не удалось. Не приписывать данным лицензию статьи или стороннего репозитория.
- Преобразование VOC следует официальной формуле Ultralytics: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml . Ширина/высота равны исходной разности координат, без добавления пикселя. Нулевая высота/ширина не исправляется догадкой.

## Найдены, но не включены

- Drone-vs-Bird: https://github.com/wosdetc/challenge . Есть самолётные БПЛА и дальние цели, однако официальный порядок требует письма авторам и подписанного DUA. Чужие копии вместо этого порядка не использованы; запрос от имени пользователя не отправлялся.
- DroneSwarms: https://hiyuur.github.io/ . Очень малые RGB-цели, но выдача данных по форме. Доступ за пользователя не оформлялся.
- MMFW-UAV: https://github.com/AIUS-LAB/MMFW-UAV , https://www.scidb.cn/s/vmimum . Специализированные самолётные БПЛА, разные ракурсы и сенсоры; полный объём более130GB. В текущий набор не включён. Инфракрасные снимки нельзя просто объявить RGB-примерами.
- YOLOMG/ARD100: https://github.com/Irisky123/YOLOMG . Отдельный набор с малыми целями, опубликованная ссылка ведёт на Baidu; в текущую сборку не включён.

Отчёты о загрузке с точными URL, размерами, CRC и SHA находятся в sources/*_download.json; ARD содержит списки отдельных полученных записей ZIP. Сведения проверены17.10.2026. Это описание источников и ограничений, а не новая лицензия объединённого набора.
