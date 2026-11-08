"""A short demonstration workflow on top of the existing acquisition and evidence path."""
from pathlib import Path
import json,time,uuid
from PySide6.QtCore import QTimer,Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QTabWidget,QWidget,QLabel,QPushButton,QComboBox,
 QDoubleSpinBox,QLineEdit,QPlainTextEdit,QTableWidget,QTableWidgetItem,QFileDialog,QMessageBox,QScrollArea)
from finale import PROFILES,TASKS,profile,configuration,TrialSeries,write,summary_text

class FinaleDialog(QDialog):
    def __init__(self,window,parent=None):
        super().__init__(None);self.window=window
        self.setWindowFlags(Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint);self.series=None;self.read_only_series=False;self.current_config=None;self.last_signature=None;self.last_running=None;self.last_error=None;self._model_cache=None
        self.setWindowTitle('Panoptr — полигон и финал');self.resize(1180,940);self.root=Path(__file__).resolve().parent
        outer=QVBoxLayout(self);note=QLabel('Главная задача: найти видимый признак, сохранить доказательство и передать человеку. Профиль колеи не является калибровкой; без неё метры неизвестны.');note.setWordWrap(True);outer.addWidget(note)
        self.tabs=QTabWidget();outer.addWidget(self.tabs);self.build_setup();self.build_trials();self.build_help()
        self.state=QLabel('Выберите задачу и источник.');self.state.setWordWrap(True);outer.addWidget(self.state)
        self.timer=QTimer(self);self.timer.timeout.connect(self.poll);self.timer.start(300)
    def safe(self,fn):
        try:return fn()
        except Exception as e:QMessageBox.warning(self,'Полигон и финал',str(e))
    def button(self,layout,label,fn):
        b=QPushButton(label);b.clicked.connect(lambda:self.safe(fn));layout.addWidget(b);return b
    def panel(self):return self.window.panels[self.channel.currentIndex()]
    def build_setup(self):
        page=QWidget();layout=QVBoxLayout(page);form=QFormLayout();layout.addLayout(form)
        self.channel=QComboBox();self.channel.addItems(['Канал 1','Канал 2']);form.addRow('Канал',self.channel)
        self.task=QComboBox()
        for key,spec in TASKS.items():self.task.addItem(spec['title'],key)
        form.addRow('Задача',self.task);self.site=QComboBox()
        for key,spec in PROFILES.items():self.site.addItem(spec['title'],key)
        form.addRow('Профиль пути',self.site);self.gauge=QDoubleSpinBox();self.gauge.setRange(10,2000);self.gauge.setValue(750);self.gauge.setEnabled(False);form.addRow('Измеренная колея макета, мм',self.gauge)
        self.site.currentIndexChanged.connect(lambda:self.gauge.setEnabled(self.site.currentData()=='model'))
        self.source_kind=QComboBox();self.source_kind.addItem('Локальная видеозапись','file');self.source_kind.addItem('Подключённая камера','camera');form.addRow('Источник',self.source_kind)
        self.file=QLineEdit();form.addRow('Видео',self.file);self.button(layout,'Выбрать видео',self.choose_video)
        self.task.currentIndexChanged.connect(self._sync_default_video)
        self._sync_default_video()
        self.device=QComboBox();self.device.addItems(['CPU','GPU','Auto']);form.addRow('Обработка',self.device)
        self.size=QComboBox();self.size.addItems(['320','416','512','640','960','1024','1280','1536']);self.size.setCurrentText('640');form.addRow('Размер входа модели',self.size)
        self.conf=QDoubleSpinBox();self.conf.setRange(.05,.95);self.conf.setValue(.25);self.conf.setSingleStep(.05);form.addRow('Порог модели',self.conf)
        row=QHBoxLayout();layout.addLayout(row);self.button(row,'1. Применить профиль',self.apply_profile);self.button(row,'2. Запустить канал',self.start_channel);self.button(row,'Запустить оба канала',self.start_both_channels);self.button(row,'Остановить канал',self.stop_channel)
        self.profile_note=QLabel('Для USB-камеры выберите устройство в основном окне. Для другого ракурса заново задайте зону и проверочную калибровку. Запись или макет не доказывают работу на ВСМ.');self.profile_note.setWordWrap(True);layout.addWidget(self.profile_note)
        row=QHBoxLayout();layout.addLayout(row);self.button(row,'3. Карточки и доказательства',self.window._open_final_dashboard);self.button(row,'Карта и следующий рейс',self.window._open_route_memory);self.button(row,'Лаборатория данных',self.window._open_dataset_lab)
        self.button(layout,'Проверить локальные веса и зависимости',self.check_ready);self.ready=QPlainTextEdit();self.ready.setReadOnly(True);layout.addWidget(self.ready)
        self.tabs.addTab(page,'1. Сквозной показ')
    def _examples_dir(self):
        for candidate in [self.root.parents[1]/'examples',self.root/'examples']:
            if candidate.is_dir():return candidate
        return self.root
    def _default_video_for_task(self,task_key):
        ex=self._examples_dir()
        preferred='drone.avi' if task_key=='drone' else 'rail.avi'
        for name in [preferred,'rail.avi','drone.avi']:
            p=ex/name
            if p.is_file():return str(p.resolve())
        return ''
    def _sync_default_video(self):
        current=self.file.text().strip()
        ex=self._examples_dir()
        if not current or Path(current).parent==ex:
            default_vid=self._default_video_for_task(self.task.currentData())
            if default_vid:self.file.setText(default_vid)
    def choose_video(self):
        start_dir=str(Path(self.file.text()).parent) if self.file.text().strip() and Path(self.file.text()).parent.is_dir() else str(self._examples_dir())
        p=QFileDialog.getOpenFileName(self,'Локальное видео',start_dir,'Видео (*.avi *.mp4 *.mkv *.mov)')[0]
        if p:self.file.setText(p)
    def site_profile(self):return profile(self.site.currentData(),self.gauge.value() if self.site.currentData()=='model' else None)
    def fingerprint(self):
        p=self.panel();o=p.options();return json.dumps({'channel':p.channel,'model':o.model,'size':o.image_size,'device':o.device,'confidence':o.confidence,'fps':p.inference_fps.value(),
          'source_kind':p.source.currentIndex(),'source':p.source_path.text(),'camera':p.selected_camera_index(),'policy':p.policy},sort_keys=True,ensure_ascii=False,default=str)
    def apply_profile(self):
        if self.series and self.series.active:raise ValueError('Завершите опыт перед сменой профиля.')
        p=self.panel();kind=self.source_kind.currentData();source=self.file.text() if kind=='file' else p.selected_camera_index()
        cfg=configuration(self.root,self.task.currentData(),self.site_profile(),kind,source,self.device.currentText(),int(self.size.currentText()),self.conf.value())
        if p.running:p.stop()
        p.source.setCurrentIndex(1 if kind=='file' else 0)
        if kind=='file':p.source_path.setText(str(Path(source).resolve()))
        p.model_preset.setCurrentIndex(max(0,p.model_preset.findText('Пользовательская')));p.model_path.setText(cfg['model']);p.device.setCurrentIndex(p.device.findData(cfg['device']));p.image_size.setCurrentText(str(cfg['image_size']));p.confidence.setValue(cfg['confidence']);p.inference_fps.setValue(4)
        p.policy.update(zone=[],speed_kmh=0.,visible_length_m=0.,geometry_calibration_file='');p._inference_settings_changed()
        self.current_config=cfg;self.truth_class.setText(TASKS[self.task.currentData()]['classes'][0]);self.last_signature=self.fingerprint();self.last_running=p.running
        if self.series:self.series.action('configuration_changed','Применён новый профиль '+cfg['task'])
        write(self.window.event_store.root/f'channel_{p.channel}_finale_profile.json',cfg);self.profile_note.setText(cfg['profile']['title']+f" · колея {cfg['profile']['gauge_mm']:g} мм. Калибровка не применена; расстояния неизвестны. Перед новой серией удалите объект и дождитесь окончания старого сопровождения.");self.state.setText('Профиль готов. Запустите канал, затем создайте серию испытаний.')
    def start_channel(self):
        if not self.current_config:self.apply_profile()
        if self.series:self.series.action('start_channel','Запуск канала испытателем')
        if not self.panel().running:self.panel().toggle_camera()
        self.last_running=self.panel().running
    def start_both_channels(self):
        if self.series and self.series.active:raise ValueError('Завершите опыт перед перезапуском обоих каналов.')
        saved_channel=self.channel.currentIndex()
        primary_task=self.task.currentData()
        secondary_task='drone' if primary_task!='drone' else 'rail'
        kind=self.source_kind.currentData()
        for ch_idx,t_key in [(0,primary_task),(1,secondary_task)]:
            p=self.window.panels[ch_idx]
            src=self.file.text().strip() if kind=='file' and ch_idx==saved_channel else (self._default_video_for_task(t_key) or self.file.text().strip()) if kind=='file' else p.selected_camera_index()
            cfg=configuration(self.root,t_key,self.site_profile(),kind,src,self.device.currentText(),int(self.size.currentText()),self.conf.value())
            if p.running:p.stop()
            p.source.setCurrentIndex(1 if kind=='file' else 0)
            if kind=='file':p.source_path.setText(str(Path(src).resolve()))
            p.model_preset.setCurrentIndex(max(0,p.model_preset.findText('Пользовательская')));p.model_path.setText(cfg['model']);p.device.setCurrentIndex(p.device.findData(cfg['device']));p.image_size.setCurrentText(str(cfg['image_size']));p.confidence.setValue(cfg['confidence']);p.inference_fps.setValue(4)
            p.policy.update(zone=[],speed_kmh=0.,visible_length_m=0.,geometry_calibration_file='');p._inference_settings_changed()
            write(self.window.event_store.root/f'channel_{p.channel}_finale_profile.json',cfg)
            if ch_idx==saved_channel:
                self.current_config=cfg;self.truth_class.setText(TASKS[t_key]['classes'][0])
        for p in self.window.panels:
            if not p.running:p.toggle_camera()
        self.last_signature=self.fingerprint();self.last_running=self.panel().running
        self.state.setText('Оба канала (Канал 1 и Канал 2) запущены одновременно.')
        self.window.showNormal();self.window.raise_();self.window.activateWindow()
    def stop_channel(self):
        if self.series:self.series.action('stop_channel','Остановка канала испытателем')
        if self.panel().running:self.panel().stop()
        self.last_running=self.panel().running
    def check_ready(self):
        from final_tools import readiness
        report=readiness(self.root,[self.panel().options().model]);self.ready.setPlainText(json.dumps(report,ensure_ascii=False,indent=2))
    def build_trials(self):
        page=QWidget();layout=QVBoxLayout(page);form=QFormLayout();layout.addLayout(form)
        self.operator=QLineEdit();form.addRow('Испытатель',self.operator);self.expected=QComboBox();self.expected.addItem('Один целевой объект','target');self.expected.addItem('Нормальные элементы / фон','negative');form.addRow('Тип опыта',self.expected)
        self.truth_class=QLineEdit('missing');form.addRow('Ожидаемый класс как у модели',self.truth_class);self.conditions=QLineEdit();form.addRow('Свет, положение и условия',self.conditions)
        row=QHBoxLayout();layout.addLayout(row);self.button(row,'Создать новую серию',self.new_series);self.button(row,'Открыть серию',self.open_series);self.button(row,'Начать опыт / отметить появление цели',self.begin_trial);self.button(row,'Завершить опыт',self.end_trial)
        note=QLabel('Одна цель на положительный опыт. Время начала отмечается испытателем. Сначала уберите предыдущую цель. Для метрик вручную оцените каждую карточку; автоматические предложения не являются GT.');note.setWordWrap(True);layout.addWidget(note)
        self.trials=QTableWidget(0,5);self.trials.setHorizontalHeaderLabels(['Опыт','Цель / фон','Класс','Карточек','Проверка']);self.trials.setSelectionBehavior(QTableWidget.SelectRows);self.trials.itemSelectionChanged.connect(self.refresh_cards);layout.addWidget(self.trials)
        self.cards=QComboBox();layout.addWidget(self.cards);row=QHBoxLayout();layout.addLayout(row);self.judgment=QComboBox()
        for title,key in [('Обнаружение цели','match'),('Ложное событие','false'),('Повтор уже найденной цели','duplicate'),('Не считать тревогой / кандидат','ignore')]:self.judgment.addItem(title,key)
        row.addWidget(self.judgment);self.reason=QLineEdit();self.reason.setPlaceholderText('Обоснование оценки или причина вмешательства');row.addWidget(self.reason);self.button(row,'Оценить карточку',self.judge);self.button(row,'Опыт проверен',self.review)
        row=QHBoxLayout();layout.addLayout(row);self.button(row,'Записать помощь человека',self.assistance);self.button(row,'Экспортировать протокол и доказательства',self.export)
        self.summary=QPlainTextEdit();self.summary.setReadOnly(True);self.summary.setMaximumHeight(170);layout.addWidget(self.summary);self.tabs.addTab(page,'2. Серия испытаний')
    def new_series(self):
        if not self.current_config:raise ValueError('Сначала примените профиль.')
        if self.series and self.series.active:raise ValueError('Завершите текущий опыт.')
        self.operator.setReadOnly(False)
        from finale import digest
        from copy import deepcopy
        p=self.panel();opts=p.options();cfg=deepcopy(self.current_config);cfg.update(model=opts.model,model_sha256=digest(opts.model),device=opts.device,image_size=opts.image_size,confidence=opts.confidence,
          channel=p.channel,source_kind='file' if p.source.currentIndex()==1 else 'camera',source=p.source_path.text() if p.source.currentIndex()==1 else str(p.selected_camera_index()),policy=deepcopy(p.policy),inference_fps=p.inference_fps.value())
        self.series=TrialSeries(self.window.event_store.root/'finale'/uuid.uuid4().hex,cfg,self.operator.text(),self.window.event_store.root)
        self.read_only_series=False
        self.last_signature=self.fingerprint();self.last_running=self.panel().running;self.operator.setReadOnly(True);self.refresh_trials();self.state.setText('Серия создана. Начните целевой или отрицательный опыт.')
    def begin_trial(self):
        if not self.series:raise ValueError('Создайте серию.')
        if self.read_only_series:raise ValueError('Старая серия открыта для проверки и экспорта. Создайте новую для текущего запуска.')
        if not self.panel().running:raise ValueError('Сначала запустите канал.')
        self.series.begin(self.expected.currentData(),self.truth_class.text(),self.conditions.text(),self.window.event_store.events());self.refresh_trials();self.trials.selectRow(len(self.series.data['trials'])-1)
    def end_trial(self):
        if not self.series:raise ValueError('Нет серии.')
        self.poll();self.series.finish();self.refresh_trials();self.state.setText('Опыт завершён. Оцените каждую карточку и нажмите «Опыт проверен».')
    def open_series(self):
        if self.series and self.series.active:raise ValueError('Завершите текущий опыт.')
        p=QFileDialog.getExistingDirectory(self,'Папка серии с protocol.json')
        if p:
            self.series=TrialSeries.open(p);self.read_only_series=True;self.operator.setText(self.series.data['operator']);self.operator.setReadOnly(True);self.last_signature=self.fingerprint();self.last_running=self.panel().running;self.refresh_trials();self.state.setText('Серия открыта для проверки и экспорта. Перед новыми опытами создайте новую серию.')
    def selected_trial(self):
        i=self.trials.currentRow()
        if not self.series or not 0<=i<len(self.series.data['trials']):raise ValueError('Выберите опыт.')
        return self.series.data['trials'][i]
    def refresh_trials(self):
        if not self.series:return
        current=self.trials.currentRow();self.trials.blockSignals(True);self.trials.setRowCount(len(self.series.data['trials']))
        for i,t in enumerate(self.series.data['trials']):
            for j,value in enumerate([i+1,t['expected'],t['class_name'],len(t['events']),'проверен' if t['reviewed'] else t['status']]):self.trials.setItem(i,j,QTableWidgetItem(str(value)))
        self.trials.blockSignals(False)
        if current>=0:self.trials.selectRow(current)
        self.refresh_cards();self.summary.setPlainText(summary_text(self.series.summary()))
    def refresh_cards(self):
        key=self.cards.currentData();self.cards.clear()
        try:t=self.selected_trial()
        except ValueError:return
        for eid,e in t['events'].items():self.cards.addItem(eid[:10]+' · '+e['event']['class_name']+' · '+t['judgments'].get(eid,{}).get('result','не проверено'),eid)
        self.cards.setCurrentIndex(max(0,self.cards.findData(key)))
    def judge(self):
        t=self.selected_trial();self.series.judge(t['id'],self.cards.currentData(),self.judgment.currentData(),self.reason.text());self.refresh_trials()
    def review(self):self.series.review(self.selected_trial()['id']);self.refresh_trials()
    def assistance(self):
        if not self.series:raise ValueError('Создайте серию.')
        self.series.action('manual_assistance',self.reason.text());self.refresh_trials()
    def poll(self):
        if not self.series:return
        if self.read_only_series:return
        try:
            p=self.panel();signature=self.fingerprint()
            if signature!=self.last_signature:self.series.action('configuration_changed','Изменены источник, модель, порог, FPS или зона в основном окне');self.last_signature=signature
            if p.running!=self.last_running:self.series.action('channel_state','Канал '+('запущен' if p.running else 'остановлен'),assistance=False);self.last_running=p.running
            error=str(getattr(p.capture,'error','') or '') if getattr(p,'capture',None) else ''
            if error and error!=self.last_error:self.series.action('source_error',error,assistance=False)
            self.last_error=error
            packet=p.detections;fresh=bool(p.running and packet is not None and not error and not getattr(p.capture,'finished',False) and (time.monotonic()-packet.captured_at)*1000<=p.policy['max_age_ms'])
            assessed=fresh and bool(getattr(p,'health',{}).get('assessed')) and bool(not p.capture_health or p.capture_health.get('assessed'))
            self.series.observe([r for r in self.window.event_store.events() if r['channel']==p.channel],assessed);self.refresh_trials()
            self.state.setText('Опыт идёт · '+('кадр пригоден по заданным проверкам' if assessed else 'наблюдение непригодно или отсутствует') if self.series.active else 'Серия сохранена; завершённые опыты требуют ручной проверки.')
        except Exception as exc:self.state.setText('Протокол: '+str(exc))
    def export(self):
        if not self.series:raise ValueError('Нет серии.')
        parent=QFileDialog.getExistingDirectory(self,'Куда сохранить новую папку отчёта')
        if parent:
            out=self.series.export(Path(parent)/('Panoptr_испытания_'+uuid.uuid4().hex[:8]));self.state.setText('Сохранено: '+str(out)+' и ZIP')
    def build_help(self):
        page=QPlainTextEdit();page.setReadOnly(True);page.setPlainText('САМОСТОЯТЕЛЬНАЯ РЕПЕТИЦИЯ\n\n1. Проверьте комплект без интернета, начните с CPU. Не обновляйте зависимости перед показом.\n2. Выберите одну задачу и источник. Проверьте нормальную сцену, цель и возвращение к норме.\n3. Сохраните карточку и докажите происхождение кадра.\n4. Проведите несколько опытов с фиксированными порогами; отметьте пропуски, ложные события и помощь человека.\n5. Отключите и верните камеру. Закрытый объектив и отсутствие находок не доказывают свободный путь.\n6. Проверьте ZIP на другом ноутбуке.\n\nНОВОЕ ЗАДАНИЕ\nИзображения и метки → проверка пар/классов/групп → подготовка набора → совместимые локальные веса → отдельный запуск обучения → оценка и экспорт. Сначала небольшая проба по разрешению команды, рабочие веса не перезаписывать. Предложения авторазметки проверять вручную. SAM 3 требует отдельно доступных весов.\n\nКОЛЕЯ\n750/1520/макет — профиль испытания, не калибровка. Для метров нужны независимые проверочные точки конкретной камеры. Физические 400 км/ч, промышленные камеры и климат этой программной репетицией не подтверждаются.\n\nОБМЕН\nКарта рейсов принимает panoptr.route.v1. Пример пакета и запуск проверки находятся в папке finale_examples. Без достоверного места событие остаётся без координаты. Это демонстрационный контракт, не готовая интеграция с внутренними системами РЖД.');self.tabs.addTab(page,'3. Подготовка без наставника')
    def closeEvent(self,event):
        if self.series and self.series.active:QMessageBox.information(self,'Опыт идёт','Завершите опыт, чтобы сохранить его длительность.');event.ignore();return
        self.timer.stop();event.accept()
