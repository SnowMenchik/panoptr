"""Manual event GT review with original-frame boxes and separate CPU scoring."""
from pathlib import Path
from datetime import datetime
import copy,json,sys,uuid
import cv2
from PySide6.QtCore import Qt,Signal,QProcess
from PySide6.QtGui import QImage,QPixmap,QPainter,QPen
from PySide6.QtWidgets import (QDialog,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QPushButton,
    QLineEdit,QComboBox,QDoubleSpinBox,QSlider,QPlainTextEdit,QFileDialog,QMessageBox,QTabWidget,QCheckBox)
from final_core import save_json
from event_benchmark import new_project,validate,freeze

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]/'data/event_benchmark'; WORK.mkdir(parents=True,exist_ok=True)

class BoxCanvas(QWidget):
    changed=Signal()
    def __init__(self):
        super().__init__(); self.setMinimumSize(480,260); self.image=None; self.box=None; self.origin=None; self.rect=None
    def show_frame(self,image):
        self.image=image; self.box=None; self.update()
    def paintEvent(self,event):
        from PySide6.QtCore import QRect
        painter=QPainter(self); painter.fillRect(QRect(0,0,self.width(),self.height()),Qt.black)
        if self.image is None: return
        rgb=cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]
        q=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy()
        pix=QPixmap.fromImage(q).scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
        from PySide6.QtCore import QRect
        self.rect=QRect((self.width()-pix.width())//2,(self.height()-pix.height())//2,pix.width(),pix.height())
        painter.drawPixmap(self.rect,pix)
        if self.box:
            x1,y1,x2,y2=self.box; painter.setPen(QPen(Qt.green,2))
            painter.drawRect(int(self.rect.x()+x1/w*self.rect.width()),int(self.rect.y()+y1/h*self.rect.height()),
                int((x2-x1)/w*self.rect.width()),int((y2-y1)/h*self.rect.height()))
    def point(self,pos):
        if self.image is None or self.rect is None: return None
        h,w=self.image.shape[:2]
        return [max(0.,min(float(w),(pos.x()-self.rect.x())/self.rect.width()*w)),
                max(0.,min(float(h),(pos.y()-self.rect.y())/self.rect.height()*h))]
    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton: self.origin=self.point(event.position()); self.box=None
    def mouseMoveEvent(self,event):
        pos=self.point(event.position())
        if self.origin and pos:
            a,b=self.origin,pos; self.box=[min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])]; self.update()
    def mouseReleaseEvent(self,event): self.mouseMoveEvent(event); self.origin=None; self.changed.emit()

class EventBenchmarkDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent); self.setWindowTitle('Panoptr — независимая оценка событий'); self.resize(1150,880)
        self.project=None; self.path=None; self.cap=None; self.process=QProcess(self); self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_log); self.process.finished.connect(self.done_process)
        self.process.errorOccurred.connect(lambda *_:self.log.appendPlainText(self.process.errorString()))
        layout=QVBoxLayout(self); self.tabs=QTabWidget(); layout.addWidget(self.tabs); self.build_editor(); self.build_tools()
        self.status=QLabel('Начните с локального видео. Разметка модели и решение оператора не являются независимым GT.'); self.status.setWordWrap(True); layout.addWidget(self.status)
        self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(100); layout.addWidget(self.log)
        b=QPushButton('Остановить только инструмент оценки'); b.clicked.connect(self.process.terminate); layout.addWidget(b)
    def safe(self,fn):
        try: return fn()
        except Exception as exc: QMessageBox.warning(self,'Оценка событий',str(exc))
    def button(self,layout,text,fn):
        b=QPushButton(text); b.clicked.connect(lambda:self.safe(fn)); layout.addWidget(b); return b
    def field(self,form,label,directory=False):
        row=QWidget(); h=QHBoxLayout(row); h.setContentsMargins(0,0,0,0); edit=QLineEdit(); h.addWidget(edit)
        def choose():
            path=QFileDialog.getExistingDirectory(self,label) if directory else QFileDialog.getOpenFileName(self,label)[0]
            if path: edit.setText(path)
        self.button(h,'Выбрать',choose); form.addRow(label,row); return edit
    def spin(self,form,label):
        v=QDoubleSpinBox(); v.setRange(0,100000); v.setDecimals(3); v.setSingleStep(.1); form.addRow(label,v); return v
    def build_editor(self):
        page=QWidget(); layout=QVBoxLayout(page); row=QHBoxLayout(); layout.addLayout(row)
        self.button(row,'Новый проект из видео',self.create_project); self.button(row,'Открыть GT JSON',self.open_project)
        self.button(row,'Добавить видео-сцену',self.add_scene); self.button(row,'Сохранить черновик как…',self.save_project)
        self.button(row,'Зафиксировать копию набора',self.freeze_project)
        row=QHBoxLayout(); layout.addLayout(row); self.scene_combo=QComboBox(); self.scene_combo.currentIndexChanged.connect(self.select_scene); row.addWidget(self.scene_combo)
        self.event_combo=QComboBox(); self.event_combo.currentIndexChanged.connect(self.select_event); row.addWidget(self.event_combo)
        self.button(row,'Новый объект',self.new_event); self.button(row,'Удалить объект',self.delete_event)
        body=QHBoxLayout(); layout.addLayout(body,1); left=QVBoxLayout(); body.addLayout(left,3); self.canvas=BoxCanvas(); left.addWidget(self.canvas,1)
        self.slider=QSlider(Qt.Horizontal); self.slider.setRange(0,0); self.slider.valueChanged.connect(self.seek); left.addWidget(self.slider)
        self.clock=QLabel('0 с'); left.addWidget(self.clock); controls=QHBoxLayout(); left.addLayout(controls)
        self.button(controls,'← кадр',lambda:self.slider.setValue(self.slider.value()-1)); self.button(controls,'кадр →',lambda:self.slider.setValue(self.slider.value()+1))
        right=QVBoxLayout(); body.addLayout(right,2); form=QFormLayout(); right.addLayout(form)
        self.class_name=QLineEdit('person'); form.addRow('Класс как у модели',self.class_name)
        self.kind=QComboBox(); self.kind.addItems(['target','normal']); form.addRow('Цель / похожая норма',self.kind)
        self.location=QComboBox(); self.location.addItems(['inside','near','outside','unknown']); form.addRow('Положение',self.location)
        self.start=self.spin(form,'Начало объекта, с'); self.end=self.spin(form,'Конец объекта, с')
        row=QHBoxLayout(); right.addLayout(row); self.button(row,'Начало = этот кадр',lambda:self.start.setValue(self.current_time()))
        self.button(row,'Конец = этот кадр',lambda:self.end.setValue(self.current_time()))
        self.tags=QLineEdit(); form.addRow('Теги через запятую',self.tags)
        self.button(right,'Записать параметры объекта',self.save_event)
        self.button(right,'Сохранить нарисованный bbox на этом кадре',self.add_keyframe)
        self.button(right,'Удалить bbox этого кадра',self.delete_keyframe)
        self.group=QLineEdit(); form.addRow('Группа исходной съёмки',self.group)
        self.condition=QLineEdit(); form.addRow('Условия сцены',self.condition)
        self.reviewer=QLineEdit(); form.addRow('Проверяющий',self.reviewer)
        self.bad_start=self.spin(form,'Начало непригодности, с'); self.bad_end=self.spin(form,'Конец непригодности, с')
        self.button(right,'Добавить непригодный интервал',self.add_bad)
        self.reviewed=QCheckBox('Вся сцена просмотрена; все целевые объекты отмечены'); right.addWidget(self.reviewed)
        self.button(right,'Сохранить проверку сцены',self.save_scene)
        self.summary=QLabel(); self.summary.setWordWrap(True); right.addWidget(self.summary)
        self.raw=QPlainTextEdit(); self.raw.setMaximumHeight(135); layout.addWidget(self.raw)
        self.button(layout,'Применить JSON: интервалы, bbox, протокол и параметры системы',self.apply_json)
        info=QLabel('GT: один ID на физический объект; начало/конец — видимость в этой сцене. Рисуйте bbox на исходных кадрах. По умолчанию промежуток между bbox ≤0,5 с. normal — похожий нормальный элемент, не целевая тревога. До фиксации задайте пороги/размер/зону в JSON.'); info.setWordWrap(True); layout.addWidget(info)
        self.tabs.addTab(page,'Разметка времени и объектов')
    def build_tools(self):
        page=QWidget(); layout=QVBoxLayout(page); form=QFormLayout(); layout.addLayout(form)
        self.frozen=self.field(form,'Фиксированный набор с lock.json',True)
        self.model=self.field(form,'Завершённые локальные веса .pt')
        self.button(layout,'Прогнать фиксированные сцены на CPU и оценить события',lambda:self.worker('event_infer',{'frozen':self.frozen.text(),'model':self.model.text()}))
        self.pred=self.field(form,'JSON результатов с временной шкалой')
        self.button(layout,'Оценить импортированные результаты',lambda:self.worker('event_score',{'frozen':self.frozen.text(),'predictions':self.pred.text()}))
        self.first=self.field(form,'Отчёт A report.json'); self.second=self.field(form,'Отчёт B report.json')
        self.button(layout,'Сравнить A/B на одном протоколе',lambda:self.worker('event_compare',{'first':self.first.text(),'second':self.second.text()}))
        self.references=QPlainTextEdit(); self.references.setPlaceholderText('Абсолютные папки обучения, val и подбора порогов: одна папка на строку'); self.references.setMaximumHeight(90); layout.addWidget(self.references)
        self.button(layout,'Аудит точных совпадений (первые 1000 файлов)',self.audit)
        info=QLabel('CPU-прогон моделирует последовательное расписание с пропуском кадров на время predict. Задержка не включает экспозицию, транспорт, GUI и запись. Во время обучения это функциональная проверка. Независимость проверяйте по исходным сценам и истории подбора порогов. Заморозка и отсутствие точных копий сами по себе её не доказывают. Живой журнал нельзя импортировать без корректного перевода времени и исходных кадров.'); info.setWordWrap(True); layout.addWidget(info); layout.addStretch()
        self.tabs.addTab(page,'Оценка и сравнение')
    def scene(self):
        if self.project is None or self.scene_combo.currentIndex()<0: raise ValueError('Откройте проект.')
        return self.project['scenes'][self.scene_combo.currentIndex()]
    def selected_event(self):
        s=self.scene(); index=self.event_combo.currentIndex()
        if not 0<=index<len(s['events']): raise ValueError('Выберите объект.')
        return s['events'][index]
    def current_time(self): return self.slider.value()/self.scene()['fps']
    def sync(self):
        if self.project: self.raw.setPlainText(json.dumps(self.project,ensure_ascii=False,indent=2))
        if self.project and self.scene_combo.currentIndex()>=0:
            s=self.scene(); self.summary.setText(f"Объектов: {len(s['events'])}; bbox: {sum(len(e['keyframes']) for e in s['events'])}; непригодные интервалы: {s['unusable']}")
    def load(self,project,path=None):
        self.project=validate(project); self.path=Path(path) if path else None
        self.scene_combo.blockSignals(True); self.scene_combo.clear(); self.scene_combo.addItems([s['scene_id'] for s in self.project['scenes']]); self.scene_combo.blockSignals(False)
        self.select_scene(); self.sync()
    def create_project(self):
        path=QFileDialog.getOpenFileName(self,'Локальное видео')[0]
        if path: self.load(new_project(path))
    def add_scene(self):
        self.scene(); path=QFileDialog.getOpenFileName(self,'Добавить видео')[0]
        if path:
            self.project['scenes'].append(new_project(path,'scene_'+uuid.uuid4().hex[:8])['scenes'][0]); self.load(self.project,self.path)
            self.scene_combo.setCurrentIndex(len(self.project['scenes'])-1)
    def open_project(self):
        path=QFileDialog.getOpenFileName(self,'GT JSON')[0]
        if path: self.load(json.loads(Path(path).read_text(encoding='utf-8')),path)
    def select_scene(self,*_):
        if not self.project or self.scene_combo.currentIndex()<0: return
        if self.cap: self.cap.release()
        s=self.scene(); source=Path(s['source']); source=source if source.is_absolute() else self.path.parent/source if self.path else source
        self.cap=cv2.VideoCapture(str(source)); self.slider.blockSignals(True); self.slider.setRange(0,max(0,round(s['duration_s']*s['fps'])-1)); self.slider.setValue(0); self.slider.blockSignals(False)
        self.group.setText(s['source_group']); self.condition.setText(s.get('condition','original')); self.reviewer.setText(s.get('reviewer','')); self.reviewed.setChecked(s.get('reviewed')==[[0.,s['duration_s']]])
        self.refresh_objects(); self.seek(0)
    def refresh_objects(self,index=0):
        self.event_combo.blockSignals(True); self.event_combo.clear(); self.event_combo.addItems([e['gt_id']+' / '+e['class_name'] for e in self.scene()['events']]); self.event_combo.setCurrentIndex(index); self.event_combo.blockSignals(False)
        self.select_event(); self.sync()
    def select_event(self,*_):
        if not self.project or self.event_combo.currentIndex()<0: return
        e=self.selected_event(); self.class_name.setText(e['class_name']); self.kind.setCurrentText(e.get('kind','target')); self.location.setCurrentText(e['location'])
        self.start.setValue(e['start_s']); self.end.setValue(e['end_s']); self.tags.setText(','.join(e.get('tags',[]))); self.seek(self.slider.value())
    def seek(self,index):
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,index); ok,image=self.cap.read()
        if ok:
            self.canvas.show_frame(image); t=self.current_time(); self.clock.setText(f'Кадр {index} · {t:.3f} с · {image.shape[1]}×{image.shape[0]}')
            if self.event_combo.currentIndex()>=0:
                k=next((k for k in self.selected_event()['keyframes'] if abs(k['t']-t)<1e-6),None)
                if k: self.canvas.box=k['box']; self.canvas.update()
    def new_event(self):
        s=self.scene(); t=self.current_time(); duration=s['duration_s']
        s['events'].append({'gt_id':uuid.uuid4().hex[:10],'class_name':'person','kind':'target','location':'unknown',
            'start_s':t,'end_s':min(duration,t+1/s['fps']),'keyframes':[],'tags':[]})
        self.refresh_objects(len(s['events'])-1)
    def delete_event(self): self.selected_event(); del self.scene()['events'][self.event_combo.currentIndex()]; self.refresh_objects()
    def save_event(self):
        e=self.selected_event(); e.update({'class_name':self.class_name.text().strip(),'kind':self.kind.currentText(),'location':self.location.currentText(),
            'start_s':self.start.value(),'end_s':self.end.value(),'tags':[t.strip() for t in self.tags.text().split(',') if t.strip()]}); self.sync()
    def add_keyframe(self):
        e=self.selected_event(); self.save_event(); t=self.current_time(); box=self.canvas.box
        if not box or box[2]<=box[0] or box[3]<=box[1]: raise ValueError('Нарисуйте рамку мышью.')
        if not e['start_s']<=t<=e['end_s']: raise ValueError('Этот кадр вне времени объекта.')
        e['keyframes']=[k for k in e['keyframes'] if abs(k['t']-t)>1e-6]+[{'t':t,'box':box.copy()}]; e['keyframes'].sort(key=lambda k:k['t']); self.sync()
    def delete_keyframe(self):
        e=self.selected_event(); t=self.current_time(); e['keyframes']=[k for k in e['keyframes'] if abs(k['t']-t)>1e-6]; self.sync()
    def add_bad(self):
        self.scene()['unusable'].append([self.bad_start.value(),self.bad_end.value()]); self.sync()
    def save_scene(self):
        s=self.scene(); s.update({'source_group':self.group.text().strip(),'condition':self.condition.text().strip(),'reviewer':self.reviewer.text().strip(),
            'reviewed':[[0.,s['duration_s']]] if self.reviewed.isChecked() else []}); self.sync()
    def apply_json(self): self.load(json.loads(self.raw.toPlainText()),self.path)
    def save_project(self):
        if self.project is None: raise ValueError('Нет проекта.')
        if json.loads(self.raw.toPlainText())!=self.project: raise ValueError('Сначала примените изменённый JSON.')
        self.save_scene()
        if self.event_combo.currentIndex()>=0: self.save_event()
        # Draft can contain unfinished objects; strict validation applies at freeze.
        path=QFileDialog.getSaveFileName(self,'Новый черновик GT',str(WORK/'draft.json'),'JSON (*.json)')[0]
        if path:
            if (Path(path).parent/'lock.json').exists(): raise ValueError('Сохраните новую копию вне фиксированного набора.')
            p=copy.deepcopy(self.project)
            for s in p['scenes']:
                source=Path(s['source'])
                if not source.is_absolute() and self.path: s['source']=str((self.path.parent/source).resolve())
            save_json(path,p); self.path=Path(path); self.project=p; self.sync(); self.status.setText('Черновик: '+path)
    def freeze_project(self):
        if not self.path: raise ValueError('Сначала сохраните черновик.')
        if json.loads(self.raw.toPlainText())!=self.project: raise ValueError('Сначала примените изменённый JSON.')
        self.save_scene()
        if self.event_combo.currentIndex()>=0: self.save_event()
        # Freeze current in-memory revision, not stale last-saved editor content.
        validate(self.project,True); parent=QFileDialog.getExistingDirectory(self,'Папка для новой фиксированной копии',str(WORK))
        if parent:
            folder=Path(parent)/('frozen_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            draft=Path(parent)/('freeze_input_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
            p=copy.deepcopy(self.project)
            for s in p['scenes']:
                source=Path(s['source'])
                if not source.is_absolute(): s['source']=str((self.path.parent/source).resolve())
            save_json(draft,p); freeze(draft,folder); self.frozen.setText(str(folder)); self.status.setText('Зафиксировано: '+str(folder))
    def worker(self,action,extra):
        if self.process.state()!=QProcess.NotRunning: raise ValueError('Инструмент уже работает.')
        folder=WORK/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'); folder.mkdir(parents=True)
        cfg={'action':action,'output':str(folder/'audit.json' if action=='event_audit' else folder/'result'),**extra}
        save_json(folder/'config.json',cfg); self.output=Path(cfg['output']); self.log.clear(); self.status.setText('CPU-инструмент: '+action)
        self.process.start(sys.executable,[str(ROOT/'final_cli.py'),str(folder/'config.json')])
    def audit(self):
        if not self.path: raise ValueError('Сохраните черновик GT для аудита.')
        if self.project!=json.loads(self.path.read_text(encoding='utf-8')): raise ValueError('Аудит проверяет сохранённый черновик; сначала сохраните изменения.')
        self.worker('event_audit',{'project':str(self.path),'roots':[r.strip() for r in self.references.toPlainText().splitlines() if r.strip()],'limit':1000})
    def read_log(self): self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace'))
    def done_process(self,code,*_):
        self.read_log(); self.status.setText(('Готово: ' if code==0 else 'Ошибка: ')+str(self.output))
        if code==0:
            path=self.output/'evaluation'/'report.json' if (self.output/'evaluation'/'report.json').exists() else self.output/'report.json'
            if path.exists():
                r=json.loads(path.read_text(encoding='utf-8')); self.log.appendPlainText(json.dumps(r['summary'],ensure_ascii=False,indent=2))
    def closeEvent(self,event):
        if self.process.state()!=QProcess.NotRunning:
            QMessageBox.information(self,'Оценка событий','Сначала остановите или дождитесь инструмента оценки.'); event.ignore(); return
        if self.cap: self.cap.release()
        super().closeEvent(event)
