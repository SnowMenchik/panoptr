"""Reviewable event cards, CPU rehearsals and read-only preparation tools."""
from pathlib import Path
from datetime import datetime
import json,os,sys
import cv2
from PySide6.QtCore import QProcess,QTimer,QUrl,Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QTabWidget,QWidget,QLabel,QPushButton,QLineEdit,QPlainTextEdit,
    QTableWidget,QTableWidgetItem,QFileDialog,QMessageBox,QFormLayout,QSpinBox,QComboBox,QSplitter,QScrollArea,QHeaderView)
from PySide6.QtGui import QBrush,QColor,QDesktopServices,QImage,QPixmap
from final_core import save_json,read_image
from final_store import ArchiveStore
from ui_theme import decision_style, decision_legend_html, paint_row, paint_status_cell

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]/'data'

class FinalDashboard(QDialog):
    def __init__(self,panels,store,parent=None):
        super().__init__(None); self.setWindowFlags(Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint); self.panels=panels; self.store=store; self.view_store=store; self.setWindowTitle('Panoptr — события и подготовка финала'); self.resize(1180,850)
        self.process=QProcess(self); self.process.setProcessChannelMode(QProcess.MergedChannels); self.process.readyReadStandardOutput.connect(self.read_log); self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error); from ui_theme import apply_competition_theme, create_competition_banner
        self.busy=[]; self.rows=[]; layout=QVBoxLayout(self); layout.addWidget(create_competition_banner('СОБЫТИЯ, ДОКАЗАТЕЛЬСТВА И ПОДГОТОВКА ФИНАЛА', 'Журнал карточек оператора • Репетиция отказов • Инженерные инструменты ВСМ')); self.tabs=QTabWidget(); layout.addWidget(self.tabs)
        self.build_events(); self.build_rehearsal(); self.build_review(); self.build_readiness(); self.build_compare()
        self.button(layout,'Разметка и независимая оценка событий',self.open_event_benchmark)
        self.button(layout,'Инженерный расчёт камеры и покрытия',self.open_camera_design)
        self.button(layout,'Виртуальный 3D-стенд и компоновка оборудования',self.open_virtual_stand)
        self.button(layout,'Понимание пути, балласт и сравнение сопровождения',self.open_path_understanding)
        self.button(layout,'Справочник дефектов рельсов / запись обследования',self.open_rail_reference)
        self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(100); layout.addWidget(self.log)
        self.state=QLabel('Инструменты не запускают обучение. Репетиция использует CPU.'); self.state.setWordWrap(True); layout.addWidget(self.state)
        stop=QPushButton('Остановить инструмент'); stop.clicked.connect(self.process.terminate); layout.addWidget(stop)
        apply_competition_theme(self); self.timer=QTimer(self); self.timer.timeout.connect(self.refresh_events); self.timer.start(1000)
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
        self.button(controls,'Текущий журнал',lambda:self.set_store(self.store)); self.button(controls,'Экспорт ZIP (CSV + JSON + PNG)',self.export_events)
        self.button(controls,'Что означают столбцы',self.explain_columns)
        split=QSplitter(); self.events_table=QTableWidget(0,8)
        self.events_table.setHorizontalHeaderLabels(['Время UTC','Кам.','Класс','Наблюдение','Решение оператора','Зона','Снимок','Комментарий'])
        self.events_table.setSelectionBehavior(QTableWidget.SelectRows); self.events_table.setEditTriggers(QTableWidget.NoEditTriggers); self.events_table.itemSelectionChanged.connect(self.show_event)
        split.addWidget(self.events_table)
        details=QWidget(); right=QVBoxLayout(details); self.image=QLabel('Выберите событие'); self.image.setAlignment(Qt.AlignCenter); self.image.setMinimumSize(320,220); right.addWidget(self.image)
        self.details=QPlainTextEdit(); self.details.setReadOnly(True); right.addWidget(self.details); split.addWidget(details); split.setSizes([820,360]); split.setStretchFactor(0,3); split.setStretchFactor(1,2); layout.addWidget(split)
        self.events_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        for index,width in enumerate([84,46,100,100,150,92,66,160]): self.events_table.setColumnWidth(index,width)
        self.events_table.horizontalHeader().setStretchLastSection(True)
        legend=QLabel(); legend.setTextFormat(Qt.RichText); legend.setWordWrap(True)
        legend.setText('Цвет строки — это решение оператора (текст решения остаётся в столбце «Решение оператора»): '+decision_legend_html()+'. Наблюдение модели показано отдельно: событие ИИ само по себе не является подтверждением дефекта.')
        layout.addWidget(legend)
        row=QHBoxLayout(); layout.addLayout(row)
        self.notes_path=WORK/'ui_state'/'operator_notes.json'; self.notes_history=self.load_notes()
        self.note=QComboBox(); self.note.setEditable(True); self.note.setInsertPolicy(QComboBox.NoInsert); self.note.setMinimumWidth(320)
        self.note.addItems(self.notes_history)
        if self.note.lineEdit() is not None: self.note.lineEdit().setPlaceholderText('Комментарий оператора: основание решения (↑ / ↓ — прошлые комментарии)')
        self.note.setToolTip('Комментарий сохраняется вместе с решением и остаётся в столбце «Комментарий». После нажатия кнопки решения поле очищается; стрелки ↑ и ↓ в поле подставляют прошлые комментарии (последние 8).')
        row.addWidget(self.note,1)
        for title,status,tip in [('Подтвердить','accepted','Объект или дефект действительно виден на снимке. Хранится отдельно от вывода модели.'),
                                 ('Отклонить','rejected','Событие ложное: объекта/дефекта нет либо это другой класс. Укажите причину в комментарии.'),
                                 ('Закрыть без решения','closed','Вы посмотрели карточку, но не подтверждаете и не опровергаете: например, кадр нечитаемый или нужен осмотр на месте. Это НЕ «путь свободен» и НЕ подтверждение дефекта.')]:
            button=self.button(row,title,lambda status=status:self.decide(status)); button.setToolTip(tip)
        info=QLabel('Повторные наблюдения не заменяют решение оператора. Нет независимого GT — Recall и FP/мин не вычисляются. «Закрыть без решения» нужно для карточек, которые нельзя оценить по снимку: такие строки видны серым, но не считаются ни подтверждёнными, ни ложными.'); info.setWordWrap(True); layout.addWidget(info)
        self.tabs.addTab(page,'События')
    def load_notes(self):
        try:
            values=json.loads(self.notes_path.read_text(encoding='utf-8'))
            return [str(v) for v in values if str(v).strip()][:8]
        except (OSError,ValueError): return []
    def remember_note(self,text):
        text=(text or '').strip()
        if not text: return
        self.notes_history=[text]+[v for v in self.notes_history if v!=text]
        self.notes_history=self.notes_history[:8]
        try:
            self.notes_path.parent.mkdir(parents=True,exist_ok=True); save_json(self.notes_path,self.notes_history)
        except OSError: pass
        self.note.blockSignals(True); self.note.clear(); self.note.addItems(self.notes_history); self.note.setCurrentIndex(-1); self.note.blockSignals(False)
        if self.note.lineEdit() is not None: self.note.lineEdit().setPlaceholderText('Комментарий оператора: основание решения (↑ / ↓ — прошлые комментарии)')
        self.note.setToolTip('Прошлые комментарии (стрелки ↑ / ↓): '+' | '.join(self.notes_history[:8]) if self.notes_history else 'История комментариев пока пуста.')
    def explain_columns(self):
        QMessageBox.information(self,'Что означают столбцы', (
            'Время UTC — момент первого появления карточки (всемирное время).\n\n'
            'Кам. — номер канала камеры (1 или 2).\n\n'
            'Класс — что нашла модель. Это вывод модели, а не подтверждённый диагноз.\n\n'
            'Наблюдение — сколько кадров подтвердили кандидата:\n'
            '  • Кандидат — 1 кадр, ещё не подтверждён повторами;\n'
            '  • Повторные — объект виден в нескольких пригодных кадрах;\n'
            '  • Плохие данные — кадры устарели, слишком тёмные/смазанные, кандидат сохранён как свидетельство.\n\n'
            'Решение оператора — ваш вывод: Подтверждено / Отклонено / Закрыто без решения / Не проверено. Цвет всей строки соответствует этому решению.\n\n'
            'Зона — где объект относительно заданной зоны пути:\n'
            '  • «Не задана» — зона не настроена, поэтому оценивать «в пути / вне пути» нельзя;\n'
            '  • «В зоне» — объект перекрывает зону пути; «Рядом» — у границы; «Вне зоны» — вне коридора;\n'
            '  • «Контактная сеть» — объект над путём (надземный класс), наземная зона к нему не применяется;\n'
            '  • «Воздушное наблюдение» — БПЛА: коридор пути и метры по земле к нему не применяются.\n\n'
            'Снимок — сохранён ли кадр-доказательство (evidence.png). «Есть» — файл лежит в журнале и попадёт в ZIP; «Нет» — запись не удалась или кадр не получен, это надо считать потерей доказательства.\n\n'
            'Комментарий — текст, который вы ввели при решении. Он виден здесь у каждой строки, поэтому не теряется.\n\n'
            '«Закрыть без решения» — это пропуск, а не подтверждение и не отклонение: карточка просмотрена, но по снимку нельзя ни подтвердить событие, ни назвать его ложным. Такое событие не должно попадать в отчёт как подтверждённое.'))
    def set_store(self,store): self.view_store=store; self.session_label.setText(str(store.root)); self.refresh_events()
    def open_archive(self):
        path=QFileDialog.getExistingDirectory(self,'Журнал с session.json',str(WORK/'sessions'))
        if path: self.set_store(ArchiveStore(path))
    def zone_text(self,r):
        return {'inside':'В зоне','near':'Рядом','outside':'Вне зоны','zone_not_set':'Не задана','geometry_unknown':'Геометрия неизвестна','not_applicable_overhead':'Контактная сеть','not_applicable_aerial':'Воздушное наблюдение'}.get(r['zone']['location'],r['zone']['location'])
    def refresh_events(self):
        if not self.isVisible(): return
        selected=self.selected(); identifier=selected['event_id'] if selected else None; self.rows=self.view_store.events()
        self.events_table.blockSignals(True); self.events_table.setRowCount(len(self.rows))
        for i,r in enumerate(self.rows):
            observation={'candidate':'Кандидат','observed_repeatedly':'Повторные','unassessed_candidate':'Плохие данные'}.get(r['observation_state'],r['observation_state'])
            status=r.get('operator_status','unreviewed'); style=decision_style(status)
            values=[r['created_utc'].split('T')[-1][:8],r['channel']+1,{'person':'Человек','drone':'БПЛА','rock':'Камень','missing':'Отсутствует','displaced':'Смещено','inverted':'Перевёрнуто','deformed':'Деформация','fractured':'Разрушено'}.get(r['class_name'],r['class_name']),observation,style['label'],self.zone_text(r),'Есть' if r.get('evidence_saved') else 'Нет',r.get('operator_note','') or '—']
            for j,value in enumerate(values):
                item=QTableWidgetItem(str(value)); item.setToolTip(r['created_utc']+' | '+r['event_id']+' | '+r['class_name']); self.events_table.setItem(i,j,item)
            paint_row(self.events_table,i,style)
            paint_status_cell(self.events_table.item(i,4),style,style['label'])
            observation_item=self.events_table.item(i,3); observation_item.setForeground(QBrush(QColor('#475569')))
            self.events_table.item(i,7).setToolTip(r.get('operator_note','') or 'Комментарий не введён')
            self.events_table.item(i,6).setToolTip('Кадр-доказательство сохранён' if r.get('evidence_saved') else 'Кадр не сохранён: это потеря доказательства, укажите в отчёте')
        self.events_table.blockSignals(False)
        if identifier:
            index=next((i for i,r in enumerate(self.rows) if r['event_id']==identifier),-1)
            if index>=0: self.events_table.selectRow(index)
        if self.tabs.currentIndex()==0: self.state.setText(self.view_store.error if self.view_store.error else f'Событий: {len(self.rows)} · подтверждено {sum(r.get("operator_status")=="accepted" for r in self.rows)} · отклонено {sum(r.get("operator_status")=="rejected" for r in self.rows)} · закрыто без решения {sum(r.get("operator_status")=="closed" for r in self.rows)} · ждут решения {sum(r.get("operator_status","unreviewed")=="unreviewed" for r in self.rows)}. Исходная разметка моделей сохранена, решение оператора отдельно.')
    def selected(self):
        i=self.events_table.currentRow(); return self.rows[i] if 0<=i<len(self.rows) else None
    def show_event(self):
        r=self.selected()
        if not r: return
        path=self.view_store.root/'events'/r['event_id']/'evidence.png'; image=read_image(path) if path.exists() else None
        if image is not None:
            box=r.get('evidence_observation',r)['box']; x1,y1,x2,y2=map(int,box); cv2.rectangle(image,(x1,y1),(x2,y2),(0,200,255),2)
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]; q=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy(); self.image.setPixmap(QPixmap.fromImage(q).scaled(self.image.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        else: self.image.setText('Доказательство не сохранено; проверьте журнал ошибки записи.')
        from rail_reference import advice
        status=r.get('operator_status','unreviewed'); style=decision_style(status)
        header=['Решение оператора: '+style['label']+('  ·  комментарий: '+r['operator_note'] if r.get('operator_note') else '  ·  комментарий не введён'),
            'Наблюдение модели: '+{'candidate':'кандидат (один кадр)','observed_repeatedly':'повторные пригодные кадры','unassessed_candidate':'кадры непригодны'}.get(r['observation_state'],r['observation_state']),
            'Зона: '+self.zone_text(r)+self.zone_hint(r),
            'Снимок-доказательство: '+('сохранён (evidence.png)' if r.get('evidence_saved') else 'не сохранён — потеря доказательства'),
            'Событие: '+r['event_id'],'']
        self.details.setPlainText('\n'.join(header)+advice(r['class_name'])['text']+'\n\n'+json.dumps(r,ensure_ascii=False,indent=2))
    def zone_hint(self,r):
        return {'inside':' — объект перекрывает настроенную зону пути.','near':' — объект у границы зоны; для решения нужен глаз оператора.','outside':' — объект вне настроенной зоны пути.','zone_not_set':' — зона не настроена: «в пути / вне пути» не определяется.','not_applicable_overhead':' — объект над путём, наземная зона к нему не применяется.','not_applicable_aerial':' — БПЛА, коридор пути и метры по земле не применяются.','geometry_unknown':' — геометрия не подтверждена.'}.get(r['zone']['location'],'')
    def decide(self,status):
        r=self.selected()
        if not r: raise ValueError('Выберите событие в таблице.')
        text=self.note.currentText().strip(); self.view_store.decide(r['event_id'],status,text); self.remember_note(text)
        self.note.setCurrentIndex(-1); self.note.clearEditText(); self.refresh_events(); self.show_event()
    def export_events(self):
        parent=QFileDialog.getExistingDirectory(self,'Куда сохранить новую папку отчёта',str(WORK))
        if not parent: return
        target=Path(parent)/('Panoptr_события_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        output=self.view_store.export(target); archive=Path(str(output)+'.zip'); rows=self.view_store.events()
        images=sum(1 for r in rows if r.get('evidence_saved')); size=archive.stat().st_size/1024/1024 if archive.is_file() else 0.
        QMessageBox.information(self,'Экспорт готов',
            f'Папка: {output}\nАрхив: {archive}\n\n'
            f'Внутри ZIP всегда одни и те же файлы:\n'
            f'• events.csv — таблица событий для Excel ({len(rows)} строк)\n'
            f'• events.json — те же события полностью, с зоной и решениями\n'
            f'• summary.json — сводка, определения задержек и честные N/A для Recall / FP\n'
            f'• telemetry.json — телеметрия кадров\n'
            f'• evidence/<ID>/evidence.png и context.avi — снимок и короткий клип события (снимков: {images} из {len(rows)})\n\n'
            f'Размер ZIP: {size:.1f} МБ. Отдельного выбора «только CSV» или «только JSON» нет: ZIP — это тот же отчёт вместе с доказательствами.')
        self.state.setText('Сохранены отчёт и ZIP: '+str(archive))
    def build_rehearsal(self):
        page=QWidget(); layout=QVBoxLayout(page); form=QFormLayout(); layout.addLayout(form)
        self.video=self.field(form,'Локальное видео'); def_v=ROOT.parents[1]/'examples/rail.avi'; self.video.setText(os.path.abspath(def_v) if def_v.is_file() else ''); self.model=self.field(form,'Локальные веса'); self.model.setText(os.path.abspath(ROOT/'models/person_security_yolo26n.pt'))
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
        page=QWidget(); layout=QVBoxLayout(page)
        self.button(layout,'Проверить среду и веса (то же, что 05_Проверить_сборку.bat)',self.readiness_check)
        info=QLabel('Это ровно та же проверка, что и 05_Проверить_сборку.bat: импорты библиотек, дочерний процесс Python, наличие весов. Результат пишется в data/last_check.json и разбирается словами ниже, поэтому отдельный BAT для проверки не обязателен. Проверка не открывает камеры, не запускает инференс и не подтверждает CUDA, кодеки и качество модели на путях.')
        info.setWordWrap(True); layout.addWidget(info)
        self.ready_summary=QPlainTextEdit(); self.ready_summary.setReadOnly(True); self.ready_summary.setMinimumHeight(200)
        self.ready_summary.setPlainText('Нажмите «Проверить среду и веса». Здесь появится разбор: итог, версия Python, каждая библиотека, дочерний процесс, CUDA, веса двух каналов, ошибки и путь к файлу результата.')
        layout.addWidget(self.ready_summary,1)
        row=QHBoxLayout(); layout.addLayout(row); self.button(row,'Открыть папку data',self.open_data_folder); self.button(row,'Показать подробный JSON',self.toggle_raw)
        self.ready_raw=QPlainTextEdit(); self.ready_raw.setReadOnly(True); self.ready_raw.setMaximumHeight(150); self.ready_raw.setVisible(False); self.ready_raw.setPlainText('Подробный JSON появится после проверки.')
        layout.addWidget(self.ready_raw); self.tabs.addTab(page,'Готовность')
    def open_data_folder(self):
        WORK.mkdir(parents=True,exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(WORK)))
    def toggle_raw(self):
        self.ready_raw.setVisible(not self.ready_raw.isVisible())
    def readiness_check(self):
        if self.process.state()!=QProcess.NotRunning: raise ValueError('Инструмент ещё работает.')
        self.active_action='ready'; self.active_output=WORK/'last_check.json'
        try: self.active_output.unlink()
        except OSError: pass
        for b in self.busy: b.setEnabled(False)
        self.log.clear(); self.state.setText('Проверка среды и весов: 10–60 секунд, камеры не открываются.')
        self.process.start(sys.executable,[str(ROOT/'portable_check.py')])
    def apply_readiness(self,code=None):
        try: report=json.loads(self.active_output.read_text(encoding='utf-8'))
        except (OSError,ValueError):
            self.ready_summary.setPlainText('Файл результата не создан: '+str(self.active_output)+'. Проверка не завершилась, подробности в журнале ниже.'); self.state.setText('Проверка не создала data/last_check.json.'); return
        lines=['ИТОГ: '+('ВСЁ ГОТОВО' if report.get('passed') else 'ЕСТЬ ПРОБЛЕМЫ — список ошибок ниже')+f"   (код завершения {code})",
            'Python: '+str(report.get('executable') or report.get('python'))]
        modules=report.get('modules') or {}; dependencies=report.get('dependencies') or {}
        if modules: lines.append('Библиотеки: '+', '.join(f"{name} — OK" for name in modules))
        elif dependencies: lines.append('Библиотеки: '+', '.join(f"{name} — {'OK' if value else 'НЕТ'}" for name,value in dependencies.items()))
        if report.get('worker_ok') is not None: lines.append('Дочерний процесс Python: '+('OK' if report['worker_ok'] else 'ОШИБКА (проверьте антивирус и права на папку)'))
        cuda=report.get('cuda')
        if isinstance(cuda,dict): lines.append('CUDA: '+('доступна — '+str(cuda.get('device','GPU')) if cuda.get('available') else 'недоступна, приложение будет считать на CPU (это не ошибка установки)'))
        weights=report.get('weights') or []
        for weight in weights: lines.append('Веса проверки: '+('OK  ' if weight.get('exists') else 'НЕТ  ')+str(weight.get('path'))+('  sha256 '+str(weight['sha256'])[:16] if weight.get('sha256') else ''))
        for index,panel in enumerate(self.panels):
            model=Path(panel.options().model)
            lines.append(f'Веса канала {index+1}: '+('OK  ' if model.is_file() else 'НЕТ  ')+str(model))
        errors=report.get('errors') or []
        lines.append('Ошибки: '+('нет' if not errors else str(len(errors))))
        lines.extend('  • '+str(item) for item in errors)
        if report.get('scope'): lines.append('Что НЕ проверено: '+str(report['scope']))
        lines.append('Файл результата: '+str(self.active_output))
        self.ready_summary.setPlainText('\n'.join(lines)); self.ready_raw.setPlainText(json.dumps(report,ensure_ascii=False,indent=2))
        self.state.setText(('Проверка пройдена: ' if report.get('passed') else 'Проверка нашла проблемы: ')+str(self.active_output))
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
        if self.active_action=='ready':
            self.apply_readiness(code); return
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
