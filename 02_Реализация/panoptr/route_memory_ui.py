"""Dispatcher view: chainage map, human review, run history and next-run warnings."""
from pathlib import Path
import json
import secrets
from PySide6.QtCore import Qt,QTimer
from PySide6.QtGui import QColor,QPen,QBrush,QPixmap,QFont,QFontDatabase
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,
    QPushButton,QTableWidget,QTableWidgetItem,QTabWidget,QPlainTextEdit,QFileDialog,QMessageBox,
    QDialog,QDialogButtonBox,QFormLayout,QLineEdit,QDoubleSpinBox,QComboBox,QGraphicsView,QGraphicsScene,QGraphicsItem,QSplitter)
from route_memory import RouteMemory,LABELS,stamp,session_packets,import_file,seed_demo
from ui_theme import ROUTE_STYLES,paint_row,paint_status_cell,legend_html

DEFAULT_DB=Path(__file__).resolve().parents[2]/'data/route_memory/route_memory.sqlite'
STATES={'pending':'На проверке','confirmed':'Подтверждено','rejected':'Отклонено','closed':'Закрыто'}

def configure_fonts():
    app=QApplication.instance()
    if app and not app.property('route_fonts_ready'):
        font_path=Path('C:/Windows/Fonts/segoeui.ttf')
        if font_path.is_file():
            identifier=QFontDatabase.addApplicationFont(str(font_path))
            families=QFontDatabase.applicationFontFamilies(identifier)
            if families:app.setFont(QFont(families[0],10))
        app.setProperty('route_fonts_ready',True)

def readable_time(value):
    return stamp(value).strftime('%Y-%m-%d %H:%M:%S UTC') if value else 'нет'

def form(parent,title,fields):
    dialog=QDialog(parent); dialog.setWindowTitle(title); layout=QFormLayout(dialog)
    widgets={}
    for key,label,default in fields:
        widget=QLineEdit(str(default)); layout.addRow(label,widget); widgets[key]=widget
    buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addRow(buttons)
    return {key:w.text().strip() for key,w in widgets.items()} if dialog.exec()==QDialog.Accepted else None

class RouteMemoryWindow(QMainWindow):
    def __init__(self,db=None,session=None,parent=None,demo=False):
        super().__init__(parent)
        configure_fonts()
        self.setWindowTitle('Panoptr — Карта рейсов'+(' — ДЕМО' if demo else ''))
        self.resize(1240,840); self.memory=RouteMemory(db or DEFAULT_DB)
        self.receiver=None; self.session=None; self.session_cache={}; self.demo_windows=[]; self.demo=demo
        self.rows=[]; self.unknown=[]; self.refreshing=False
        central=QWidget(); self.setCentralWidget(central); outer=QVBoxLayout(central)
        from ui_theme import create_competition_banner
        outer.addWidget(create_competition_banner('КАРТА РЕЙСОВ И ПАМЯТЬ МАРШРУТА ВСМ', 'Наблюдения → верификация инженера → история участка по километражу → задание на следующий рейс'))
        note=QLabel('Карта по километражу, без подложки. GNSS поезда не равен месту объекта. Отсутствие предупреждений не подтверждает свободный путь.')
        note.setWordWrap(True); outer.addWidget(note)
        bar=QHBoxLayout(); outer.addLayout(bar)
        for label,callback in [('Импорт JSON / JSONL',self.import_data),('Подключить журнал',self.connect_session),
                               ('Приём от поезда',self.toggle_receiver),('Демо двух рейсов',self.open_demo),('Экспорт архива',self.export)]:
            b=QPushButton(label); b.clicked.connect(callback); bar.addWidget(b)
        self.status=QLabel('Хранилище: '+str(self.memory.path)); self.status.setWordWrap(True); outer.addWidget(self.status)
        filters=QHBoxLayout(); outer.addLayout(filters)
        self.route=QComboBox(); self.track=QComboBox(); self.route.setMinimumWidth(180)
        filters.addWidget(QLabel('Линия')); filters.addWidget(self.route)
        filters.addWidget(QLabel('Путь')); filters.addWidget(self.track)
        self.route.currentTextChanged.connect(self.change_route); self.track.currentTextChanged.connect(self.draw)
        filters.addWidget(QLabel('Статус')); self.state=QComboBox(); self.state.addItems(['Все','На проверке','Подтверждено','Устарело','Отклонено','Закрыто'])
        self.state.currentTextChanged.connect(self.draw); filters.addWidget(self.state); filters.addStretch()
        self.tabs=QTabWidget(); outer.addWidget(self.tabs,1)
        map_page=QWidget(); map_layout=QVBoxLayout(map_page)
        self.scene=QGraphicsScene(); self.scene.selectionChanged.connect(self.map_selected)
        self.map=QGraphicsView(self.scene); self.map.setMinimumHeight(190)
        self.map.setMaximumHeight(260); map_layout.addWidget(self.map)
        split=QSplitter(Qt.Horizontal); map_layout.addWidget(split,1)
        self.table=QTableWidget(0,7); self.table.setHorizontalHeaderLabels(['Тип','Км','Статус','Новое / повтор','Рейсов','Наблюдений','Последнее UTC'])
        self.table.setSelectionBehavior(QTableWidget.SelectRows); self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.show_detail); self.table.cellClicked.connect(self.select_marker)
        split.addWidget(self.table)
        details=QWidget(); dl=QVBoxLayout(details); self.detail=QPlainTextEdit(); self.detail.setReadOnly(True); dl.addWidget(self.detail)
        self.evidence=QLabel('Кадр из подключённого журнала'); self.evidence.setAlignment(Qt.AlignCenter); self.evidence.setWordWrap(True); self.evidence.setMinimumHeight(110); dl.addWidget(self.evidence)
        split.addWidget(details); split.setSizes([780,400])
        self.legend=QLabel(); self.legend.setTextFormat(Qt.RichText); self.legend.setWordWrap(True)
        self.legend.setText('Цвет строки — статус проверки (текст статуса остаётся в столбце «Статус»): '+legend_html([(ROUTE_STYLES[k]['bg'],ROUTE_STYLES[k]['fg'],k) for k in ['На проверке','Подтверждено','Устарело','Отклонено','Закрыто']])+'. «Устарело» — подтверждение истекло, объект нужно проверить снова.')
        map_layout.addWidget(self.legend)
        actions=QHBoxLayout(); map_layout.addLayout(actions)
        for label,action in [('Подтвердить','confirm'),('Отклонить','reject'),('Закрыть по проверке','close'),('На повторную проверку','reopen')]:
            button=QPushButton(label); button.clicked.connect(lambda checked=False,a=action:self.decide(a)); actions.addWidget(button)
        self.tabs.addTab(map_page,'Карта и события')
        unknown_page=QWidget(); ul=QVBoxLayout(unknown_page)
        ul.addWidget(QLabel('Без проверенного места объекта: сохраняются в журнале, не участвуют в предупреждениях.'))
        self.unlocated_table=QTableWidget(0,5); self.unlocated_table.setHorizontalHeaderLabels(['Тип','Поезд','Рейс','Время UTC','Причина'])
        self.unlocated_table.setSelectionBehavior(QTableWidget.SelectRows); self.unlocated_table.setEditTriggers(QTableWidget.NoEditTriggers)
        ul.addWidget(self.unlocated_table); b=QPushButton('Указать проверенное место выбранного наблюдения'); b.clicked.connect(self.locate); ul.addWidget(b)
        self.tabs.addTab(unknown_page,'Без места')
        next_page=QWidget(); nl=QVBoxLayout(next_page); nf=QFormLayout(); nl.addLayout(nf)
        self.run=QLineEdit('Следующий-рейс'); nf.addRow('Рейс получателя',self.run)
        self.position=QDoubleSpinBox(); self.position.setRange(0,10000); self.position.setDecimals(3); self.position.setSuffix(' км'); nf.addRow('Положение поезда',self.position)
        self.direction=QComboBox(); self.direction.addItems(['По возрастанию километража','По убыванию километража']); nf.addRow('Направление',self.direction)
        self.distance=QDoubleSpinBox(); self.distance.setRange(.001,100); self.distance.setValue(10); self.distance.setSuffix(' км'); nf.addRow('Глубина просмотра',self.distance)
        button=QPushButton('Показать актуальные предупреждения по выбранной линии и пути'); button.clicked.connect(self.preview); nl.addWidget(button)
        self.preview_text=QPlainTextEdit(); self.preview_text.setReadOnly(True); nl.addWidget(self.preview_text)
        self.tabs.addTab(next_page,'Следующий рейс')
        self.timer=QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(3000)
        self.refresh()
        if session: self.status.setText('Текущий журнал доступен. Нажмите «Подключить журнал»: '+str(session)); self.initial_session=Path(session)
        else: self.initial_session=None
        QTimer.singleShot(0,self.fit_map)

        from ui_theme import apply_competition_theme
        apply_competition_theme(self)

    def fail(self,exc): QMessageBox.warning(self,'Карта рейсов',str(exc))
    def import_data(self):
        path,_=QFileDialog.getOpenFileName(self,'Пакет поезда','','JSON (*.json *.jsonl)')
        if not path:return
        try:
            counts=import_file(self.memory,path); self.refresh(); self.status.setText('Импорт: '+json.dumps(counts,ensure_ascii=False))
        except Exception as e:self.fail(e)

    def connect_session(self):
        def_sess=self.initial_session or (Path(__file__).resolve().parents[2]/'data/sessions'); session=QFileDialog.getExistingDirectory(self,'Журнал поезда с session.json',str(def_sess))
        if not session:return
        values=form(self,'Идентификация журнала',[('train','Поезд','Поезд-01'),('run','Уникальный ID рейса',Path(session).name)])
        if not values:return
        self.session=(Path(session),values['train'],values['run']); self.session_cache={}
        self.tick()
        self.update_status()
        QMessageBox.information(self,'Журнал подключён',
            f'Журнал: {self.session[0]}\nПоезд / рейс: {values["train"]} / {values["run"]}\n'
            f'Принято наблюдений: {len(self.session_cache)}\nОбъектов с проверенным местом: {len(getattr(self,"all_rows",[]))}\nНаблюдений без места: {len(self.unknown)}\n\n'
            'Что делать дальше: если все наблюдения без координат, откройте вкладку «Без места», выберите строку и нажмите «Указать проверенное место выбранного наблюдения». '
            'Координата поезда не считается местом объекта, поэтому такие события не появляются на карте, пока место не указано вручную.')

    def tick(self):
        try:
            if self.session:
                path,train,run=self.session; imported=0
                for packet in session_packets(path,train,run):
                    key=packet['observations'][0]['observation_id']
                    if key in self.session_cache: continue
                    self.memory.ingest(packet); self.session_cache[key]=packet; imported+=1
                    if imported>=200: break
                self.update_status(prefix=f'Журнал подключён: {path} | Принято событий: {len(self.session_cache)}. ')
            self.refresh()
            if self.tabs.currentIndex()==2:self.preview()
        except Exception as e: self.status.setText('Ошибка приёма, повтор через 3 с: '+str(e))

    def refresh(self):
        self.all_rows=self.memory.objects(); self.refreshing=True
        previous=self.route.currentText(); routes=sorted({r['route'] for r in self.all_rows})
        self.route.clear(); self.route.addItems(routes)
        if previous in routes:self.route.setCurrentText(previous)
        self.refreshing=False; self.change_route()
        previous_index=self.unlocated_table.currentRow()
        selected_unknown=self.unknown[previous_index]['id'] if 0<=previous_index<len(self.unknown) else None
        self.unknown=self.memory.unlocated()
        self.unlocated_table.blockSignals(True); self.unlocated_table.clearSelection(); self.unlocated_table.setCurrentCell(-1,-1)
        self.unlocated_table.setRowCount(len(self.unknown))
        for i,obs in enumerate(self.unknown):
            row=json.loads(obs['payload']); loc=row.get('location')
            reason='Нет координат объекта' if not loc else ('Координаты поезда' if loc['position_kind']=='train' else 'Место не проверено / погрешность >25 м')
            for j,v in enumerate([LABELS[row['kind']],obs['train'],obs['run'],row['observed_at'],reason]):self.unlocated_table.setItem(i,j,QTableWidgetItem(str(v)))
            if obs['id']==selected_unknown:self.unlocated_table.selectRow(i)
        self.unlocated_table.blockSignals(False)
        self.unlocated_table.resizeColumnsToContents()
        self.update_status()
        if self.session and not getattr(self,'all_rows',[]) and self.unknown and self.tabs.currentIndex()==0:
            # Nothing can be drawn on the chainage map: show where the data actually is.
            self.tabs.setCurrentIndex(1)

    def update_status(self,prefix=''):
        located=len(getattr(self,'all_rows',[])); unknown=len(getattr(self,'unknown',[]))
        text=f'{prefix}Хранилище: {self.memory.path} · объектов с проверенным местом: {located} · наблюдений без места: {unknown}'
        if self.session: text+=f' · журнал: {self.session[0]}'
        if unknown and not located: text+='. Все принятые наблюдения без координат объекта: откройте вкладку «Без места» и укажите место — только после этого событие появится на карте.'
        self.status.setText(text)

    def change_route(self,*args):
        if self.refreshing:return
        previous=self.track.currentText(); self.track.blockSignals(True)
        self.track.clear(); tracks=sorted({r['track'] for r in self.all_rows if r['route']==self.route.currentText()}); self.track.addItems(tracks)
        if previous in tracks:self.track.setCurrentText(previous)
        self.track.blockSignals(False); self.draw()

    def draw(self,*args):
        if self.refreshing:return
        selected=self.selected_id(); self.rows=[]
        current_route=self.route.currentText(); current_track=self.track.currentText()
        for r in self.all_rows:
            if current_route and r['route']!=current_route:continue
            if current_track and r['track']!=current_track:continue
            state='Устарело' if r['state'] in ('confirmed','pending') and r['stale'] else STATES[r['state']]
            if self.state.currentText() not in ('Все',state):continue
            self.rows.append({**r,'display_state':state})
        self.table.blockSignals(True); self.table.clearSelection(); self.table.setCurrentCell(-1,-1)
        self.table.setRowCount(len(self.rows)); self.scene.blockSignals(True); self.scene.clear()
        low=min((r['chainage'] for r in self.rows),default=0)-100
        high=max((r['chainage'] for r in self.rows),default=1000)+100
        self.scene.setSceneRect(0,0,1100,175); self.scene.addLine(50,95,1050,95,QPen(QColor('#305BA7'),4))
        for i,r in enumerate(self.rows):
            repeated=('Повторная зона' if r['kind'] in ('person','drone') else 'Повторное') if r['observations']>1 else 'Новое'
            values=[LABELS[r['kind']],f"{r['chainage']/1000:.3f}",r['display_state'],repeated,r['runs'],r['observations'],readable_time(r['last_seen'])]
            color='#c6382d' if r['fresh'] else ('#68706a' if r['state'] in ('closed','rejected') or r['stale'] else '#b77b20')
            for j,v in enumerate(values):
                item=QTableWidgetItem(str(v)); item.setToolTip(str(v)); self.table.setItem(i,j,item)
            style=ROUTE_STYLES.get(r['display_state'])
            if style:
                paint_row(self.table,i,style); paint_status_cell(self.table.item(i,2),style,r['display_state'])
            x=50+(r['chainage']-low)/(high-low)*1000
            marker=self.scene.addEllipse(x-6,89,12,12,QPen(QColor(color)),QBrush(QColor(color)))
            marker.setData(0,r['id']); marker.setFlag(QGraphicsItem.ItemIsSelectable,True)
            marker.setToolTip(f"{LABELS[r['kind']]} · {r['chainage']/1000:.3f} км · {r['display_state']}")
            label=self.scene.addText(f"{r['chainage']/1000:.3f} км\n{LABELS[r['kind']]}"); label.setPos(x-32,110 if i%2 else 30)
            if r['id']==selected:self.table.selectRow(i)
        self.scene.addText(f'{current_route or "нет линии"} / путь {current_track or "нет пути"} · положение по километражу').setPos(45,0)
        if not self.rows:
            hint='Нет объектов с проверенным местом.'+(f' Наблюдений без места: {len(self.unknown)} — откройте вкладку «Без места» и укажите место объекта.' if self.unknown else ' Загрузите демо или подключите журнал.')
            text=self.scene.addText(hint); text.setTextWidth(980); text.setPos(45,35)
        self.scene.blockSignals(False); self.table.blockSignals(False)
        for column,width in enumerate((110,70,125,115,60,85,190)):self.table.setColumnWidth(column,width)
        self.map.fitInView(self.scene.sceneRect(),Qt.KeepAspectRatio)
        self.show_detail()

    def fit_map(self):self.map.fitInView(self.scene.sceneRect(),Qt.KeepAspectRatio)
    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'map'):QTimer.singleShot(0,self.fit_map)

    def selected_id(self):
        index=self.table.currentRow()
        return self.rows[index]['id'] if 0<=index<len(self.rows) else None
    def select_marker(self,*args):self.show_detail()
    def map_selected(self):
        items=self.scene.selectedItems()
        if items:
            for index,row in enumerate(self.rows):
                if row['id']==items[0].data(0):self.table.selectRow(index);break
    def show_detail(self):
        obj_id=self.selected_id()
        if not obj_id:self.detail.clear(); return
        obj=next(r for r in self.rows if r['id']==obj_id); obs=self.memory.observations(obj_id)
        lines=[LABELS[obj['kind']],obj['display_state'],
            f"Линия: {obj['route']} / путь {obj['track']}",
            f"Место: {obj['chainage']/1000:.3f} км ± {obj['uncertainty']:.0f} м",
            f"Впервые: {readable_time(obj['first_seen'])}",f"Последнее: {readable_time(obj['last_seen'])}",
            f"Подтверждение до: {readable_time(obj['valid_until'])}",f"Примечание: {obj['note'] or 'нет'}",'',
            'История наблюдений (время UTC):']
        for item in obs:
            row=json.loads(item['payload']);quality='пригодно' if row['quality_ok'] else 'качество не подтверждено'
            lines.extend([readable_time(row['observed_at']),f"{item['train']} / {item['run']} · {quality}"])
        lines.extend(['',f"Идентификатор: {obj_id}"])
        self.detail.setPlainText('\n'.join(lines))
        self.evidence.clear(); self.evidence.setText('Кадр доступен для событий подключённого журнала')
        if self.session:
            path,train,run=self.session
            for item in obs:
                key=item['external_id']
                if item['train']!=train or item['run']!=run or len(key)!=32 or any(ch not in '0123456789abcdef' for ch in key):continue
                image=path/'events'/key/'evidence.png'
                if image.is_file(): self.evidence.setPixmap(QPixmap(str(image)).scaled(320,180,Qt.KeepAspectRatio,Qt.SmoothTransformation)); break

    def decide(self,action):
        obj_id=self.selected_id()
        if not obj_id:self.fail('Выберите карточку.');return
        values=form(self,'Решение оператора',[('actor','Оператор',''),('note','Основание / примечание','')])
        if not values:return
        try:self.memory.decide(obj_id,action,values['actor'],values['note']); self.refresh()
        except Exception as e:self.fail(e)

    def locate(self):
        index=self.unlocated_table.currentRow()
        if not 0<=index<len(self.unknown):self.fail('Выберите наблюдение без места.');return
        observation_id=self.unknown[index]['id']
        values=form(self,'Проверенное место ОБЪЕКТА, не поезда',[('actor','Оператор',''),('route','Линия',''),('track','Путь',''),('km','Положение объекта, км',''),('error','Погрешность, м','5')])
        if not values:return
        try:
            loc={'route_id':values['route'],'track_id':values['track'],'chainage_m':float(values['km'].replace(',','.'))*1000,
                 'uncertainty_m':float(values['error'].replace(',','.')),'position_kind':'object','verified':True}
            self.memory.locate(observation_id,loc,values['actor']); self.refresh()
        except Exception as e:self.fail(e)

    def preview(self):
        if not self.route.currentText() or not self.track.currentText():
            self.preview_text.setPlainText('Сначала появились бы линия и путь: сейчас нет ни одного объекта с проверенным местом. Подключите журнал, укажите места наблюдений на вкладке «Без места» или нажмите «Демо двух рейсов», чтобы увидеть пример. Отсутствие предупреждений не подтверждает свободный путь.')
            return
        try:
            rows=self.memory.warnings(self.route.currentText(),self.track.currentText(),self.position.value()*1000,
                1 if self.direction.currentIndex()==0 else -1,self.distance.value()*1000,self.run.text())
            self.preview_text.setPlainText('\n'.join(f"{LABELS[r['kind']]} · км {r['chainage']/1000:.3f} · через {r['distance_m']:.0f} м · актуально до {readable_time(r['valid_until'])}" for r in rows)
                or 'Актуальных подтверждённых предупреждений нет. Это не подтверждает свободный путь.')
        except Exception as e:self.fail(e)

    def toggle_receiver(self):
        if self.receiver:
            self.receiver.close(); self.receiver=None; self.status.setText('Приём API остановлен.');return
        from route_memory_api import Receiver
        try:
            token=secrets.token_urlsafe(32); self.receiver=Receiver(self.memory,token)
            dialog=QDialog(self); dialog.setWindowTitle('Приём данных поезда'); layout=QVBoxLayout(dialog)
            layout.addWidget(QLabel(f'Адрес: http://127.0.0.1:{self.receiver.port}/v1/observations\nТолько локальный приём; удалённый поезд подключается через защищённый шлюз.'))
            secret=QLineEdit(token); secret.setReadOnly(True); layout.addWidget(QLabel('Токен для Authorization: Bearer')); layout.addWidget(secret)
            close=QPushButton('Готово'); close.clicked.connect(dialog.accept); layout.addWidget(close); dialog.exec()
            self.status.setText(f'API принимает пакеты на 127.0.0.1:{self.receiver.port}. Повторное нажатие остановит приём.')
        except Exception as e:self.fail(e)

    def open_demo(self):
        try:
            import tempfile
            temp=tempfile.TemporaryDirectory(prefix='panoptr_demo_'); memory=RouteMemory(Path(temp.name)/'demo.sqlite'); seed_demo(memory)
            window=RouteMemoryWindow(memory.path,demo=True); window.demo_temp=temp
            window.position.setValue(124); window.show(); self.demo_windows.append(window)
        except Exception as e:self.fail(e)
    def export(self):
        path,_=QFileDialog.getSaveFileName(self,'Архив карты рейсов','route_archive.json','JSON (*.json)')
        if path:
            try:self.memory.export(path); self.status.setText('Архив с историей и решениями сохранён: '+path)
            except Exception as e:self.fail(e)
    def closeEvent(self,event):
        self.timer.stop()
        if self.receiver:self.receiver.close(); self.receiver=None
        for child in self.demo_windows:child.close()
        if hasattr(self,'demo_temp'):self.demo_temp.cleanup()
        event.accept()
