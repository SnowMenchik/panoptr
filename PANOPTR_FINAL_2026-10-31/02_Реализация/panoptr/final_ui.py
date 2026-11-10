"""Reviewable event cards, CPU rehearsals and read-only preparation tools."""
from pathlib import Path
from datetime import datetime
import json,sys,time
import cv2
from PySide6.QtCore import QProcess,QTimer,Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QTabWidget,QWidget,QLabel,QPushButton,QLineEdit,QPlainTextEdit,
    QTableWidget,QTableWidgetItem,QFileDialog,QMessageBox,QFormLayout,QSpinBox,QComboBox,QSplitter,QScrollArea,QHeaderView)
from PySide6.QtGui import QImage,QPixmap
from final_core import save_json,read_image
from final_store import ArchiveStore

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]/'11_Финальный_модуль'

class FinalDashboard(QDialog):
    def __init__(self,panels,store,parent=None):
        super().__init__(parent); self.panels=panels; self.store=store; self.view_store=store; self.setWindowTitle('Panoptr — события и подготовка финала'); self.resize(1180,850)
        self.process=QProcess(self); self.process.setProcessChannelMode(QProcess.MergedChannels); self.process.readyReadStandardOutput.connect(self.read_log); self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error); self.busy=[]; self.rows=[]; layout=QVBoxLayout(self); self.tabs=QTabWidget(); layout.addWidget(self.tabs)
        self.build_events(); self.build_storage(); self.build_rehearsal(); self.build_review(); self.build_readiness(); self.build_compare()
        self.button(layout,'Разметка и независимая оценка событий',self.open_event_benchmark)
        self.button(layout,'Инженерный расчёт камеры и покрытия',self.open_camera_design)
        self.button(layout,'Виртуальный 3D-стенд и компоновка оборудования',self.open_virtual_stand)
        self.button(layout,'Понимание пути, балласт и сравнение сопровождения',self.open_path_understanding)
        self.button(layout,'Справочник дефектов рельсов / запись обследования',self.open_rail_reference)
        self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(100); layout.addWidget(self.log)
        self.state=QLabel('Инструменты не запускают обучение. Репетиция использует CPU.'); self.state.setWordWrap(True); layout.addWidget(self.state)
        stop=QPushButton('Остановить инструмент'); stop.clicked.connect(self.process.terminate); layout.addWidget(stop)
        self.timer=QTimer(self); self.timer.timeout.connect(self.refresh_events); self.timer.start(1000)
    def open_rail_reference(self):
        from rail_reference_ui import RailReferenceDialog
        dialog=RailReferenceDialog(self.view_store,self.selected(),self);dialog.exec();self.refresh_events();self.show_event()
    def safe(self,fn):
        try: return fn()
        except Exception as exc: QMessageBox.warning(self,'Финальный модуль',str(exc))
    def open_event_benchmark(self):
        from event_benchmark_ui import EventBenchmarkDialog
        if not hasattr(self,'event_benchmark_dialog'): self.event_benchmark_dialog=EventBenchmarkDialog(self)
        self.event_benchmark_dialog.show(); self.event_benchmark_dialog.raise_()
    def open_camera_design(self):
        from camera_design_ui import CameraDesignDialog
        if not hasattr(self,'camera_design_dialog'): self.camera_design_dialog=CameraDesignDialog(self)
        self.camera_design_dialog.show(); self.camera_design_dialog.raise_()
    def open_virtual_stand(self):
        from virtual_stand import VirtualStandDialog
        if not hasattr(self,'virtual_stand_dialog') or not self.virtual_stand_dialog.isVisible(): self.virtual_stand_dialog=VirtualStandDialog(self)
        self.virtual_stand_dialog.show(); self.virtual_stand_dialog.raise_()
    def open_path_understanding(self):
        from path_ui import PathUnderstandingDialog
        if not hasattr(self,'path_dialog'): self.path_dialog=PathUnderstandingDialog(self.panels,self)
        self.path_dialog.show(); self.path_dialog.raise_()
    def button(self,layout,title,fn):
        b=QPushButton(title); b.clicked.connect(lambda:self.safe(fn)); layout.addWidget(b); self.busy.append(b); return b
    def field(self,form,label,directory=False):
        row=QWidget(); layout=QHBoxLayout(row); layout.setContentsMargins(0,0,0,0); edit=QLineEdit(); layout.addWidget(edit)
        def browse():
            path=QFileDialog.getExistingDirectory(self,label) if directory else QFileDialog.getOpenFileName(self,label)[0]
            if path: edit.setText(path)
        self.button(layout,'Выбрать',browse); form.addRow(label,row); return edit
    def build_events(self):
        page=QWidget(); layout=QVBoxLayout(page); self.session_label=QLabel(str(self.store.root)); self.session_label.setWordWrap(True); layout.addWidget(self.session_label)
        controls=QHBoxLayout(); layout.addLayout(controls); self.button(controls,'Обновить',self.refresh_events); self.button(controls,'Открыть прошлый журнал',self.open_archive)
        self.button(controls,'Текущий журнал',lambda:self.set_store(self.store)); self.button(controls,'Экспорт JSON / CSV / ZIP',self.export_events)
        split=QSplitter(); self.events_table=QTableWidget(0,7); self.events_table.setHorizontalHeaderLabels(['Время UTC','Кам.','Класс','Наблюдение','Решение','Зона','PNG']); self.events_table.setSelectionBehavior(QTableWidget.SelectRows); self.events_table.setEditTriggers(QTableWidget.NoEditTriggers); self.events_table.itemSelectionChanged.connect(self.show_event); self.events_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); split.addWidget(self.events_table)
        details=QWidget(); right=QVBoxLayout(details); self.image=QLabel('Выберите событие'); self.image.setAlignment(Qt.AlignCenter); self.image.setMinimumSize(350,230); right.addWidget(self.image)
        self.details=QPlainTextEdit(); self.details.setReadOnly(True); right.addWidget(self.details); split.addWidget(details); split.setSizes([680,460]); split.setStretchFactor(0,3); split.setStretchFactor(1,2); layout.addWidget(split)
        self.events_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        for index,width in enumerate([85,45,100,100,125,100,45]): self.events_table.setColumnWidth(index,width)
        row=QHBoxLayout(); layout.addLayout(row); self.note=QLineEdit(); self.note.setPlaceholderText('Комментарий оператора'); row.addWidget(self.note)
        for title,status in [('Подтвердить','accepted'),('Отклонить','rejected'),('Закрыть','closed')]: self.button(row,title,lambda status=status:self.decide(status))
        info=QLabel('Повторные наблюдения не заменяют решение оператора. Нет независимого GT — Recall и FP/мин не вычисляются.'); info.setWordWrap(True); layout.addWidget(info); self.tabs.addTab(page,'События')
    def build_storage(self):
        page=QWidget(); layout=QVBoxLayout(page)
        info=QLabel('Узлы хранения: ACTIVE — доступен, UNAVAILABLE — ошибка связи, DISABLED — отключён вручную, RECOVERING — проверяется. Ручное отключение исключает узел из маршрутизации и не отменяется автоматической проверкой. Включение возвращает узел только после успешной проверки. Запись считается сохранённой, если её принял хотя бы один доступный узел; состояние всех узлов указано в отчёте.'); info.setWordWrap(True); layout.addWidget(info)
        self.storage_table=QTableWidget(0,6); self.storage_table.setHorizontalHeaderLabels(['Узел','Состояние','Папка','Ошибка','Запись/чтение','Следующая проверка']); self.storage_table.setEditTriggers(QTableWidget.NoEditTriggers); self.storage_table.setSelectionBehavior(QTableWidget.SelectRows); self.storage_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); layout.addWidget(self.storage_table,1)
        row=QHBoxLayout(); layout.addLayout(row)
        self.button(row,'Обновить состояние',self.refresh_storage); self.button(row,'Отключить выбранный',self.disable_storage_node); self.button(row,'Включить и проверить',self.enable_storage_node); self.button(row,'Синхронизировать',self.sync_storage_node)
        self.storage_history=QPlainTextEdit(); self.storage_history.setReadOnly(True); self.storage_history.setMaximumHeight(180); layout.addWidget(self.storage_history)
        self.tabs.addTab(page,'Узлы хранения')
    def refresh_storage(self):
        try: rows=self.view_store.node_statuses(); history=self.view_store.storage_history(30)
        except Exception as exc:
            self.storage_history.setPlainText(str(exc)); return
        self.storage_table.setRowCount(len(rows))
        for i,row in enumerate(rows):
            delay=max(0.,row.get('next_probe_at',0.)-time.monotonic()) if row.get('next_probe_at') else 0.
            values=[row['id'],row['state'],row['root'],row.get('last_error',''),f"{row.get('writes_ok',0)}/{row.get('reads_ok',0)}",'сразу' if delay<=0 else f'через {delay:.0f} с']
            for j,value in enumerate(values): self.storage_table.setItem(i,j,QTableWidgetItem(str(value)))
        self.storage_history.setPlainText('\n'.join(json.dumps(item,ensure_ascii=False) for item in history))
    def selected_storage_node(self):
        index=self.storage_table.currentRow()
        if not 0<=index<len(self.view_store.node_statuses()): raise ValueError('Выберите узел хранения.')
        return self.view_store.node_statuses()[index]['id']
    def disable_storage_node(self):
        try:
            node=self.selected_storage_node(); self.view_store.disable_node(node,'Отключён оператором в панели хранения.'); self.refresh_storage(); self.refresh_events()
        except Exception as exc: QMessageBox.warning(self,'Узлы хранения',str(exc))
    def enable_storage_node(self):
        try:
            node=self.selected_storage_node(); status=self.view_store.enable_node(node); self.refresh_storage(); self.refresh_events()
            if status['state']!='ACTIVE': QMessageBox.warning(self,'Узлы хранения',f"Узел {node} не прошёл проверку: {status.get('last_error','')}")
        except Exception as exc: QMessageBox.warning(self,'Узлы хранения',str(exc))
    def sync_storage_node(self):
        try:
            node=self.selected_storage_node(); self.view_store.sync_node(node); self.refresh_storage()
        except Exception as exc: QMessageBox.warning(self,'Узлы хранения',str(exc))
    def set_store(self,store): self.view_store=store; self.session_label.setText(str(getattr(store,'active_root',store.root))); self.refresh_events(); self.refresh_storage()
    def open_archive(self):
        path=QFileDialog.getExistingDirectory(self,'Журнал с session.json',str(WORK/'sessions'))
        if path: self.set_store(ArchiveStore(path))
    def refresh_events(self):
        if not self.isVisible(): return
        if self.tabs.tabText(self.tabs.currentIndex())=='Узлы хранения':
            self.refresh_storage(); return
        selected=self.selected(); identifier=selected['event_id'] if selected else None; self.rows=self.view_store.events()
        self.events_table.blockSignals(True); self.events_table.setRowCount(len(self.rows))
        for i,r in enumerate(self.rows):
            observation={'candidate':'Кандидат','observed_repeatedly':'Повторные','unassessed_candidate':'Плохие данные'}.get(r['observation_state'],r['observation_state'])
            decision={'unreviewed':'Не проверено','accepted':'Подтверждено','rejected':'Отклонено','closed':'Закрыто'}.get(r['operator_status'],r['operator_status'])
            zone={'inside':'В зоне','near':'Рядом','outside':'Вне зоны','zone_not_set':'Не задана','geometry_unknown':'Геометрия неизвестна','not_applicable_overhead':'Контактная сеть','not_applicable_aerial':'Воздушное наблюдение'}.get(r['zone']['location'],r['zone']['location'])
            values=[r['created_utc'].split('T')[-1][:8],r['channel']+1,{'person':'Человек','drone':'БПЛА','rock':'Камень','missing':'Отсутствует','displaced':'Смещено','inverted':'Перевёрнуто','deformed':'Деформация','fractured':'Разрушено'}.get(r['class_name'],r['class_name']),observation,decision,zone,'Да' if r.get('evidence_saved') else 'Нет']
            for j,value in enumerate(values):
                item=QTableWidgetItem(str(value)); item.setToolTip(r['created_utc']+' | '+r['event_id']+' | '+r['class_name']); self.events_table.setItem(i,j,item)
        self.events_table.blockSignals(False)
        if identifier:
            index=next((i for i,r in enumerate(self.rows) if r['event_id']==identifier),-1)
            if index>=0: self.events_table.selectRow(index)
        if self.tabs.currentIndex()==0: self.state.setText(self.view_store.error if self.view_store.error else f'Событий: {len(self.rows)}. Исходная разметка моделей сохранена, решение оператора отдельно.')
    def selected(self):
        i=self.events_table.currentRow(); return self.rows[i] if 0<=i<len(self.rows) else None
    def show_event(self):
        r=self.selected()
        if not r: return
        path=self.view_store.evidence_path(r['event_id']); image=read_image(path) if path.exists() else None
        if image is not None:
            box=r.get('evidence_observation',r)['box']; x1,y1,x2,y2=map(int,box); cv2.rectangle(image,(x1,y1),(x2,y2),(0,200,255),2)
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]; q=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy(); self.image.setPixmap(QPixmap.fromImage(q).scaled(self.image.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        else: self.image.setText('Доказательство не сохранено; проверьте журнал ошибки записи.')
        from rail_reference import advice
        self.details.setPlainText(advice(r['class_name'])['text']+'\n\n'+json.dumps(r,ensure_ascii=False,indent=2))
    def decide(self,status):
        r=self.selected()
        if not r: raise ValueError('Выберите событие.')
        self.view_store.decide(r['event_id'],status,self.note.text()); self.refresh_events(); self.show_event()
    def export_events(self):
        path=QFileDialog.getSaveFileName(self,'Новая папка отчёта',str(WORK/'event_report'))[0]
        if path: self.view_store.export(path); self.state.setText('Сохранён отчёт и ZIP: '+path)
    def build_rehearsal(self):
        page=QWidget(); layout=QVBoxLayout(page); form=QFormLayout(); layout.addLayout(form)
        self.video=self.field(form,'Локальное видео'); self.model=self.field(form,'Локальные веса'); self.model.setText(str(ROOT/'models/person_security_yolo26n.pt'))
        self.channel=QComboBox(); self.channel.addItems(['Профиль камеры 1','Профиль камеры 2']); form.addRow('Контроль и роли классов',self.channel)
        self.limit=QSpinBox(); self.limit.setRange(3,300); self.limit.setValue(12); form.addRow('Число отобранных кадров',self.limit)
        self.every=QSpinBox(); self.every.setRange(1,10000); self.every.setValue(1); form.addRow('Шаг кадров',self.every)
        self.size=QSpinBox(); self.size.setRange(32,1280); self.size.setSingleStep(32); self.size.setValue(320); form.addRow('Размер CPU-пробы',self.size)
        self.output=QLineEdit(str(WORK/'rehearsal_new')); form.addRow('Новая папка результата',self.output)
        self.button(layout,'Запустить все шесть сценариев на CPU',self.rehearse)
        info=QLabel('Нормальная запись → смаз → темнота → потеря кадров → имитация ошибки predict → восстановление. Это программная репетиция, не испытание настоящей камеры, температуры или скорости ВСМ. Без независимого GT точность не заявляется.'); info.setWordWrap(True); layout.addWidget(info)
        self.scenario=QComboBox(); self.scenario.currentIndexChanged.connect(self.show_scenario); layout.addWidget(self.scenario)
        self.scenario_image=QLabel('После прогона можно просмотреть образец каждого сценария. В сценариях отказа образец показывает доступный кадр; сам отказ отмечен в frames.json.'); self.scenario_image.setWordWrap(True); self.scenario_image.setMinimumHeight(150); self.scenario_image.setAlignment(Qt.AlignCenter); layout.addWidget(self.scenario_image,1)
        self.scenario_summary=QLabel(); self.scenario_summary.setWordWrap(True); layout.addWidget(self.scenario_summary); self.rehearsal_report=None; self.tabs.addTab(page,'Репетиция')
    def rehearse(self):
        self.worker('rehearse',{'video':self.video.text(),'model':self.model.text(),'limit':self.limit.value(),'every':self.every.value(),'image_size':self.size.value(),
            'confidence':self.panels[self.channel.currentIndex()].confidence.value(),'policy':self.panels[self.channel.currentIndex()].policy,'output':self.output.text()})
    def build_review(self):
        page=QWidget(); page_layout=QVBoxLayout(page); controls=QWidget(); layout=QVBoxLayout(controls); form=QFormLayout(); layout.addLayout(form)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(controls); scroll.setMaximumHeight(320); page_layout.addWidget(scroll)
        self.train_images=self.field(form,'Изображения train',True); self.val_images=self.field(form,'Изображения val/test',True)
        self.scan_limit=QSpinBox(); self.scan_limit.setRange(1,5000); self.scan_limit.setValue(500); form.addRow('Максимум кадров каждой папки',self.scan_limit)
        self.button(layout,'Найти точные и похожие кадры между выборками',lambda:self.worker('duplicates',{'train_images':self.train_images.text(),'val_images':self.val_images.text(),'limit':self.scan_limit.value()}))
        self.predictions=self.field(form,'JSON предсказаний'); self.truth=self.field(form,'JSON независимого GT (необязательно)'); self.second=self.field(form,'JSON второй модели (необязательно)')
        self.button(layout,'Составить очередь сложных кадров',lambda:self.worker('review',{'predictions':self.predictions.text(),'ground_truth':self.truth.text() or None,'second':self.second.text() or None}))
        self.button(layout,'Собрать результаты обучений 08/09',lambda:self.worker('registry',{}))
        info=QLabel('Отчёт отмечает предел проверки. Ничего не удаляется в датасетах. Для JSON нужны одинаковые ID классов. CSV разных обучений не определяют общего победителя.'); info.setWordWrap(True); page_layout.addWidget(info)
        self.result_table=QTableWidget(0,3); self.result_table.setHorizontalHeaderLabels(['Источник / кадр','Показатель / причина','Дополнительно']); self.result_table.setEditTriggers(QTableWidget.NoEditTriggers); self.result_table.setSelectionBehavior(QTableWidget.SelectRows); self.result_table.itemSelectionChanged.connect(self.preview_review); page_layout.addWidget(self.result_table,1)
        self.review_image=QLabel('Изображения открываются по абсолютному пути из отчёта или относительно выбранной папки.'); self.review_image.setWordWrap(True); self.review_image.setMaximumHeight(170); page_layout.addWidget(self.review_image)
        self.review_image_root=self.field(form,'Папка изображений для очереди JSON',True); self.review_rows=[]
        self.tabs.addTab(page,'Данные и обучения')
    def build_readiness(self):
        page=QWidget(); layout=QVBoxLayout(page); self.button(layout,'Проверить локальные зависимости и веса двух каналов',lambda:self.worker('ready',{'models':[p.options().model for p in self.panels]}))
        info=QLabel('Проверка не открывает камеры и не запускает GPU. Локальный BAT работает без pip/скачивания; на другом компьютере зависимости нужно установить заранее. Эта проверка не подтверждает CUDA, физические камеры и аппаратные испытания.'); info.setWordWrap(True); layout.addWidget(info); layout.addStretch(); self.tabs.addTab(page,'Готовность')
    def build_compare(self):
        page=QWidget(); layout=QVBoxLayout(page); form=QFormLayout(); layout.addLayout(form)
        self.compare_yaml=self.field(form,'YOLO data.yaml контрольного набора'); self.compare_first=self.field(form,'Первая модель'); self.compare_second=self.field(form,'Вторая модель')
        self.compare_split=QComboBox(); self.compare_split.addItems(['val','test']); form.addRow('Выборка',self.compare_split)
        self.compare_limit=QSpinBox(); self.compare_limit.setRange(0,1000000); self.compare_limit.setValue(0); form.addRow('Предел кадров (0 = весь split)',self.compare_limit)
        self.compare_size=QSpinBox(); self.compare_size.setRange(32,1280); self.compare_size.setSingleStep(32); self.compare_size.setValue(640); form.addRow('Одинаковый размер для обеих моделей',self.compare_size)
        self.button(layout,'Сравнить на CPU после завершения обучения',lambda:self.worker('compare',{'data_yaml':self.compare_yaml.text(),'first':self.compare_first.text(),
            'second':self.compare_second.text(),'split':self.compare_split.currentText(),'limit':self.compare_limit.value(),'image_size':self.compare_size.value()}))
        info=QLabel('Входной split копируется в отдельную папку, исходники и кэши обучения не меняются. Обе модели должны иметь одинаковые классы и задачу. Без GT-файла кадр не считается фоном. P/R относятся к выбранной валидатором точке F1, AP — к общей процедуре проверки. Название test не доказывает независимость.'); info.setWordWrap(True); layout.addWidget(info)
        self.compare_table=QTableWidget(0,9); self.compare_table.setHorizontalHeaderLabels(['Модель','Класс','GT','P','R','AP50','AP50–95','Mask AP50','Mask AP50–95']); self.compare_table.setEditTriggers(QTableWidget.NoEditTriggers); self.compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); layout.addWidget(self.compare_table,1)
        self.tabs.addTab(page,'Сравнение моделей')
    def worker(self,action,extra):
        if self.process.state()!=QProcess.NotRunning: raise ValueError('Инструмент ещё работает.')
        session=WORK/'tool_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'); session.mkdir(parents=True)
        cfg={'action':action,'output':str(session/'comparison' if action=='compare' else session/'report.json'),**extra}; save_json(session/'config.json',cfg); self.active_output=Path(cfg['output']); self.active_action=action
        for b in self.busy: b.setEnabled(False)
        self.log.clear(); self.state.setText('Выполняется '+action+'; обучение не запускается.')
        self.process.start(sys.executable,[str(ROOT/'final_cli.py'),str(session/'config.json')])
    def read_log(self): self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace'))
    def process_error(self,error):
        self.log.appendPlainText(self.process.errorString())
        if error==QProcess.FailedToStart:
            for b in self.busy: b.setEnabled(True)
    def process_finished(self,code,status):
        self.read_log()
        for b in self.busy: b.setEnabled(True)
        self.state.setText('Готово: '+str(self.active_output) if code==0 else 'Ошибка инструмента; подробности в журнале.')
        if code==0 and self.active_action in ('duplicates','review','registry'):
            self.show_review_report(json.loads(self.active_output.read_text(encoding='utf-8')),self.active_action)
        elif code==0 and self.active_action=='rehearse':
            self.rehearsal_report=json.loads((self.active_output/'report.json').read_text(encoding='utf-8')); self.rehearsal_output=self.active_output
            self.scenario.clear(); self.scenario.addItems([r['scenario'] for r in self.rehearsal_report['scenarios']]); self.show_scenario()
        elif code==0 and self.active_action=='compare':
            report=json.loads((self.active_output/'comparison.json').read_text(encoding='utf-8')); rows=[]
            for i,m in enumerate(report['models']):
                for c in m['per_class']:
                    box=c['box'] or {}; mask=c['mask'] or {}
                    rows.append([('A' if i==0 else 'B'),c['class_name'],c['gt_instances'],box.get('precision'),box.get('recall'),box.get('ap50'),box.get('ap50_95'),mask.get('ap50'),mask.get('ap50_95')])
            self.compare_table.setRowCount(len(rows))
            for i,row in enumerate(rows):
                for j,value in enumerate(row): self.compare_table.setItem(i,j,QTableWidgetItem('N/A' if value is None else f'{value:.3f}' if isinstance(value,float) else str(value)))
            self.state.setText('Общий набор: '+report['input']['dataset_signature'][:12]+' · независимость не подтверждена · '+str(self.active_output))
    def show_scenario(self,*_):
        if not self.rehearsal_report or self.scenario.currentIndex()<0: return
        row=self.rehearsal_report['scenarios'][self.scenario.currentIndex()]; path=self.rehearsal_output/row['scenario']/'preview.png'
        if path.is_file():
            image=read_image(path); rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]; q=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy()
            self.scenario_image.setPixmap(QPixmap.fromImage(q).scaled(self.scenario_image.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        else: self.scenario_image.setText('Нет успешно обработанного образца.')
        latency=f"{row['p95_inference_ms']:.1f} мс" if row['p95_inference_ms'] is not None else 'N/A'
        self.scenario_summary.setText(f"Обработано {row['processed']} / {row['source_frames']}; имитаций отказа {row['injected_unavailable']}; P95 predict {latency}; Recall / FP в минуту: N/A без GT.")
    def show_review_report(self,report,action):
        self.review_rows=[]
        if action=='duplicates':
            for r in report['candidates']: self.review_rows.append({'columns':[r['train'],'точная копия' if r['exact_duplicate'] else 'похожий pHash',f"d={r['phash_distance']} | {r['val']}"],'images':[r['train'],r['val']]})
        elif action=='review':
            for r in report['queue']:
                path=Path(r['image']); path=path if path.is_absolute() else Path(self.review_image_root.text())/path
                self.review_rows.append({'columns':[r['image'],', '.join(r['reasons']),r['priority']],'images':[str(path)]})
        else:
            for r in report['runs']:
                metrics=r.get('metrics',{}); values={k:v for k,v in metrics.items() if 'metrics/' in k}
                self.review_rows.append({'columns':[r['run'],r.get('last_epoch',r.get('error','')),json.dumps(values,ensure_ascii=False)],'images':[]})
        self.result_table.setRowCount(min(1000,len(self.review_rows)))
        for i,r in enumerate(self.review_rows[:1000]):
            for j,value in enumerate(r['columns']): self.result_table.setItem(i,j,QTableWidgetItem(str(value)))
        self.state.setText(f"Записей в отчёте {len(self.review_rows)}; показано {min(1000,len(self.review_rows))}. Полный JSON: {self.active_output}")
    def preview_review(self):
        index=self.result_table.currentRow()
        if not 0<=index<len(self.review_rows): return
        images=[]
        for path in self.review_rows[index]['images']:
            image=read_image(path) if Path(path).is_file() else None
            if image is not None:
                h,w=image.shape[:2]; image=cv2.resize(image,(max(1,int(w*130/h)),130)); images.append(image)
        if not images: self.review_image.setText('Изображение не найдено или запись относится к обучению.'); return
        import numpy as np
        image=np.concatenate(images,axis=1); rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]
        q=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy(); self.review_image.setPixmap(QPixmap.fromImage(q).scaled(self.review_image.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
    def closeEvent(self,event):
        if hasattr(self,'path_dialog') and self.path_dialog.process.state()!=QProcess.NotRunning:
            QMessageBox.information(self,'Понимание пути','Сначала завершите или остановите CPU-инструмент.'); event.ignore(); return
        if hasattr(self,'event_benchmark_dialog') and self.event_benchmark_dialog.process.state()!=QProcess.NotRunning:
            QMessageBox.information(self,'Оценка выполняется','Сначала завершите или остановите инструмент оценки событий.'); event.ignore(); return
        if self.process.state()!=QProcess.NotRunning: QMessageBox.information(self,'Инструмент выполняется','Дождитесь завершения или нажмите «Остановить инструмент».'); event.ignore()
        else:
            if hasattr(self,'virtual_stand_dialog'): self.virtual_stand_dialog.close()
            event.accept()
