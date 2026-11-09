"""Offline path inspection and explicit live-channel configuration. CPU jobs only."""
from pathlib import Path
from datetime import datetime
import json,time
import cv2,numpy as np
from PySide6.QtCore import QProcess,Qt
from PySide6.QtGui import QImage,QPixmap
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QPushButton,QLineEdit,QComboBox,
    QCheckBox,QDoubleSpinBox,QSpinBox,QPlainTextEdit,QFileDialog,QMessageBox,QTabWidget,QWidget,QScrollArea)
from final_core import read_image,save_json
from path_geometry import RouteState,propose_geometry,frame_digest,serializable_geometry,GroundCalibration,object_context,source_key
from path_tracking import PathEventPipeline

ROOT=Path(__file__).resolve().parent; OUT=ROOT.parents[1]/'15_Понимание_пути'

def demo_geometry(branches=1):
    frame=np.full((360,640,3),(87,100,101),np.uint8); masks=[]; boxes=[]
    for j in range(branches):
        for cls,x in [(0,130+j*260),(1,300+j*260)]:
            mask=np.zeros((360,640),np.uint8); cv2.line(mask,(x,100),(x,359),1,7); masks.append(mask.astype(bool)); boxes.append([x-4,100,x+4,360,.92,cls]); frame[mask.astype(bool)]=(165,180,183)
    ballast=np.ones((360,640),bool); ballast[:100]=False; masks.append(ballast); boxes.append([0,100,640,360,.9,2])
    g=propose_geometry(masks,boxes,{0:'rail_left',1:'rail_right',2:'ballast'},frame.shape,1,1,0.,'synthetic_demonstration',frame_digest(frame)); g['scope']='synthetic_GT_not_model_inference'; return frame,g

def read_geometry(folder):
    folder=Path(folder); g=json.loads((folder/'geometry.json').read_text(encoding='utf-8'))
    with np.load(folder/'masks.npz',allow_pickle=False) as data:
        g['semantic_masks']={k[len('semantic_'):]:data[k].astype(bool) for k in data.files if k.startswith('semantic_')}
        for c in g['candidates']: c['mask']=data['candidate_'+c['id']].astype(bool)
    if any(m.shape!=tuple(g['shape']) for m in g['semantic_masks'].values()) or any(c['mask'].shape!=tuple(g['shape']) for c in g['candidates']): raise ValueError('Размер масок не совпадает с кадром.')
    return g

class PathUnderstandingDialog(QDialog):
    def __init__(self,panels=None,parent=None):
        super().__init__(parent); self.panels=panels or []; self.route=RouteState(); self.frame,self.raw=demo_geometry(); self.image_path=None; self.calibration=None; self.geometry=None
        self.setWindowTitle('Panoptr — понимание пути, балласт и сопровождение'); self.resize(1320,900)
        self.process=QProcess(self); self.process.setProcessChannelMode(QProcess.MergedChannels); self.process.readyReadStandardOutput.connect(self.read_log); self.process.finished.connect(self.finished); self.process.errorOccurred.connect(self.process_error)
        from ui_theme import create_competition_banner, apply_competition_theme
        outer=QVBoxLayout(self); outer.addWidget(create_competition_banner('ПОНИМАНИЕ ПУТИ, БАЛЛАСТ И СОПРОВОЖДЕНИЕ ВСМ', 'Сегментация рельсов и балласта • Стрелочные переводы • Сравнение IoU и ByteTrack')); label=QLabel('Коридор — по видимым рельсам; стрелка требует выбора. Эвристический показатель не является вероятностью безопасности. CPU; обучение не запускается.'); label.setWordWrap(True); outer.addWidget(label)
        tabs=QTabWidget(); outer.addWidget(tabs); page=QWidget(); layout=QHBoxLayout(page); left=QVBoxLayout(); layout.addLayout(left,3)
        self.geometry_status=QLabel(); self.geometry_status.setWordWrap(True); left.addWidget(self.geometry_status)
        self.preview=QLabel(); self.preview.setAlignment(Qt.AlignCenter); self.preview.setMinimumSize(640,400); self.preview.setStyleSheet('background:#142331'); left.addWidget(self.preview,1)
        self.details=QPlainTextEdit(); self.details.setReadOnly(True); self.details.setMaximumHeight(270); left.addWidget(self.details)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); body=QWidget(); controls=QVBoxLayout(body); scroll.setWidget(body); layout.addWidget(scroll,2); form=QFormLayout(); controls.addLayout(form)
        self.channel=QComboBox(); self.channel.addItems([f'Канал{i+1}' for i in range(len(self.panels))] or ['Автономный просмотр']); form.addRow('Источник применения',self.channel)
        self.model=QLineEdit(str(ROOT/'models/yolo26s_straight_track_seg_best.pt')); form.addRow('Локальные веса сегментации',self.model)
        self.conf=QDoubleSpinBox(); self.conf.setRange(.01,.99); self.conf.setSingleStep(.05); self.conf.setValue(.35); form.addRow('Порог масок',self.conf)
        self.size=QSpinBox(); self.size.setRange(32,1920); self.size.setSingleStep(32); self.size.setValue(640); form.addRow('Размер CPU-сегментации',self.size)
        self.switch=QCheckBox('Стрелка / неоднозначный выбор пути'); controls.addWidget(self.switch)
        self.candidates=QComboBox(); form.addRow('Кандидат',self.candidates); self.reviewer=QLineEdit('Оператор'); form.addRow('Кто подтверждает',self.reviewer)
        self.tracker=QComboBox(); self.tracker.addItems(['iou','bytetrack']); form.addRow('Сопровождение',self.tracker)
        self.calibration_path=QLineEdit(); form.addRow('JSON калибровки (необязательно)',self.calibration_path)
        self.box=QLineEdit('[180,180,240,300]'); form.addRow('Пробная рамка [x1,y1,x2,y2]',self.box)
        self.object_class=QComboBox(); self.object_class.addItems(['rock','person','ballast','missing']); form.addRow('Класс для проверки контекста',self.object_class)
        for text,fn in [('Открыть изображение / первый кадр видео',self.open_image),('Выбрать локальные веса',self.choose_weights),('Сегментировать этот кадр на CPU',self.segment),('Подтвердить кандидат',self.confirm),('Проверить рамку / балласт',self.probe),('Загрузить и проверить калибровку',self.load_calibration),('Экспорт геометрии и шаблона калибровки',self.export),('Демо: два пути и балласт',self.demo_switch)]: self.button(controls,text,fn)
        if self.panels:
            self.button(controls,'Снимок результата выбранного канала',self.snapshot)
            self.button(controls,'Применить настройки к каналу',self.apply_to_channel)
            self.button(controls,'Подтвердить этот путь в канале',self.confirm_live)
        controls.addStretch(); self.switch.toggled.connect(self.refresh); tabs.addTab(page,'Рельсы / коридор / объекты')
        bench=QWidget(); bl=QVBoxLayout(bench); self.compare_input=QLineEdit(str(OUT/'synthetic_tracking.json')); bl.addWidget(QLabel('JSON одинаковых детекций и проверенных GT с постояннымиID объектов:')); bl.addWidget(self.compare_input)
        self.button(bl,'Выбрать JSON сравнения',self.choose_benchmark); self.button(bl,'Сравнить IoU и ByteTrack на CPU',self.compare)
        info=QLabel('Измеряются пропуски видимых наблюдений, потери объектов, сменыID, повторные кандидаты событий, ложные треки и задержка первого сопровождения. Синтетическая проверка не измеряет точность на настоящем пути. ByteTrack сохраняет низкие оценки только если детектор передал их; высокий conf заранее отбрасывает эти находки.'); info.setWordWrap(True); bl.addWidget(info)
        self.comparison=QPlainTextEdit(); self.comparison.setReadOnly(True); bl.addWidget(self.comparison); tabs.addTab(bench,'Сравнение сопровождения')
        self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(90); outer.addWidget(self.log); self.button(outer,'Остановить CPU-инструмент',self.process.terminate); self.populate(); self.refresh()
    def button(self,layout,text,fn):
        b=QPushButton(text); b.clicked.connect(lambda:self.safe(fn)); layout.addWidget(b); return b
    def safe(self,fn):
        try: fn()
        except Exception as exc: QMessageBox.warning(self,'Понимание пути',str(exc))
    def populate(self):
        self.candidates.clear()
        for c in self.raw['candidates']: self.candidates.addItem(f"{c['id']} / Q={c['quality_score']:.2f} / {'пригоден' if c['eligible'] else 'непригоден'}",c['id'])
    def refresh(self,*args):
        self.geometry=self.route.evaluate(self.raw,self.raw['captured_at'],self.raw['sequence'],self.raw['generation'],self.raw['source_id'],switch=self.switch.isChecked())
        labels={'unknown':'Геометрия неизвестна','low_confidence':'Геометрия недостаточно надёжна','ambiguous':'Подтвердите выбранный путь','estimated':'Коридор оценён по видимым рельсам','confirmed':'Путь подтверждён оператором'}
        q=self.geometry.get('quality_score'); score='' if q is None else f' · показатель качества {q:.2f} (эвристика)'
        scope='ДЕМОНСТРАЦИЯ: синтетические маски. ' if self.raw.get('scope')=='synthetic_GT_not_model_inference' else ''
        reason='; '.join(self.geometry.get('reasons',[])); self.geometry_status.setText(scope+labels.get(self.geometry['status'],self.geometry['status'])+score+'. Метры не выдаются без проверенной калибровки.'+(' '+reason if reason else ''))
        self.geometry_status.setStyleSheet('font-weight:600; padding:6px; background:'+('#e0eee2' if self.geometry['corridor_mask'] is not None else '#fff0d2')+'; color:#172531')
        display=self.frame.copy(); semantics=self.raw.get('semantic_masks',{})
        for name,color in [('rail_left',[220,170,45]),('rail_right',[70,210,230]),('rail',[200,150,60])]:
            m=semantics.get(name)
            if m is not None: display[m]=(display[m]*.35+np.array(color)*.65).astype(np.uint8)
        for c in self.raw['candidates']: cv2.polylines(display,[np.asarray(c['polygon_px'],np.int32)],True,(0,175,255),1)
        mask=self.geometry['corridor_mask']
        if mask is not None: display[mask]=(display[mask]*.7+np.array([65,220,80])*.3).astype(np.uint8)
        rgb=cv2.cvtColor(display,cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]; image=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy(); self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        self.details.setPlainText(json.dumps(serializable_geometry(self.geometry),ensure_ascii=False,indent=2)+'\n\nИсточник: '+self.raw.get('scope','локальная сегментация; независимость не подтверждена'))
    def open_image(self):
        path,_=QFileDialog.getOpenFileName(self,'Изображение/видео',str(ROOT),'Изображения/видео (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mkv)')
        if not path: return
        if Path(path).suffix.lower() in ('.mp4','.avi','.mkv'):
            cap=cv2.VideoCapture(path); ok,frame=cap.read(); cap.release()
            if not ok: raise ValueError('Видео не читается.')
            self.frame=frame
        else: self.frame=read_image(path)
        from path_geometry import unknown_geometry
        self.raw=unknown_geometry(self.frame.shape,1,1,0.,str(Path(path).resolve()),'Сначала сегментируйте кадр'); self.raw['frame_sha256']=frame_digest(self.frame); self.image_path=path; self.route.reset(); self.calibration=None; self.populate(); self.refresh()
    def choose_weights(self):
        path,_=QFileDialog.getOpenFileName(self,'Локальные завершённые веса',str(ROOT/'models'),'Веса (*.pt)')
        if path: self.model.setText(path)
    def start_worker(self,cfg):
        if self.process.state()!=QProcess.NotRunning: raise ValueError('Инструмент ещё выполняется.')
        folder=OUT/'tool_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'); folder.mkdir(parents=True); self.action=cfg['action']; self.output=folder/'result'; cfg['output']=str(self.output)
        if cfg['action']=='segment':
            from PIL import Image
            image=folder/'input.png'; Image.fromarray(cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)).save(image); cfg['image']=str(image)
            self.pending_frame_hash=frame_digest(self.frame)
            self.pending_settings=(self.model.text(),self.conf.value(),self.size.value(),self.raw['source_id'])
        save_json(folder/'config.json',cfg); self.log.clear()
        import sys
        self.process.start(sys.executable,[str(ROOT/'path_cli.py'),str(folder/'config.json')])
    def segment(self): self.start_worker({'action':'segment','model':self.model.text(),'confidence':self.conf.value(),'imgsz':self.size.value(),'source_id':self.raw['source_id']})
    def read_log(self): self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace'))
    def process_error(self,error): self.log.appendPlainText(self.process.errorString())
    def finished(self,code,status):
        self.read_log()
        if code!=0: return
        try:
            if self.action=='segment':
                g=read_geometry(self.output)
                if frame_digest(self.frame)!=self.pending_frame_hash or g['frame_sha256']!=self.pending_frame_hash: raise ValueError('Кадр изменился: старый результат не применён.')
                if self.pending_settings!=(self.model.text(),self.conf.value(),self.size.value(),self.raw['source_id']): raise ValueError('Источник или настройки изменились: старый результат не применён.')
                self.raw=g; self.route.reset(); self.populate(); self.refresh()
            else: self.comparison.setPlainText((self.output/'comparison.md').read_text(encoding='utf-8'))
        except Exception as exc: self.log.appendPlainText(str(exc))
    def confirm(self): self.route.confirm(self.raw,self.candidates.currentData(),self.reviewer.text()); self.refresh()
    def probe(self):
        box=json.loads(self.box.text()); h,w=self.frame.shape[:2]
        if len(box)!=4 or not all(isinstance(x,(float,int)) and np.isfinite(x) for x in box) or not 0<=box[0]<box[2]<=w or not 0<=box[1]<box[3]<=h: raise ValueError('Рамка должна находиться в кадре.')
        name=self.object_class.currentText(); pipeline=PathEventPipeline(self.tracker.currentText()); health={'assessed':True,'synthetic_probe_only':True}
        updates=pipeline.update(0,1,1,0.,np.array([box+[.9,0]]),{0:name},self.frame.shape,{},health,geometry=self.geometry,calibration=self.calibration)
        self.details.setPlainText(json.dumps({'probe_uses_user_bbox_not_detector':True,'objects':pipeline.context,'event_candidates':len(updates)},ensure_ascii=False,indent=2))
    def load_calibration(self):
        path=self.calibration_path.text()
        if not path:
            path,_=QFileDialog.getOpenFileName(self,'Калибровка с проверочными точками',str(OUT),'JSON (*.json)')
            if not path: return
            self.calibration_path.setText(path)
        self.calibration=GroundCalibration(json.loads(Path(path).read_text(encoding='utf-8')))
        self.details.setPlainText(f'Калибровка: max={self.calibration.max_error:.3f}м; RMS={self.calibration.rms:.3f}м. Прошла допуск: {self.calibration.valid}.\nИсточник/разрешение и область проверяются для каждого контакта. Реальную погрешность вводит проверяющий.')
    def export(self):
        folder=OUT/'reviews'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'); folder.mkdir(parents=True)
        save_json(folder/'geometry.json',serializable_geometry(self.raw)); save_json(folder/'selected_route.json',serializable_geometry(self.geometry))
        np.savez_compressed(folder/'masks.npz',**{f'semantic_{k}':v for k,v in self.raw.get('semantic_masks',{}).items()},**{f"candidate_{c['id']}":c['mask'] for c in self.raw['candidates']})
        save_json(folder/'calibration_template.json',{'format':'panoptr_ground_calibration_v1','source_id':self.raw['source_id'],'shape':self.raw['shape'],
            'fit_points':[],'check_points':[],'reviewer':'','max_check_error_m':.1,'note':'Fill measured [u,v,X,Y] points. Check points must be independent. Template is not a calibration.'})
        self.preview.grab().save(str(folder/'preview.png')); self.log.appendPlainText('Сохранено: '+str(folder))
    def demo_switch(self): self.frame,self.raw=demo_geometry(2); self.route.reset(); self.switch.setChecked(True); self.populate(); self.refresh()
    def snapshot(self):
        panel=self.panels[self.channel.currentIndex()]; packet=panel.detections
        if packet is None or packet.geometry is None: raise ValueError('Нет сегментации текущего канала. Примените настройки, затем запустите источник самостоятельно.')
        self.frame=packet.source_frame.copy(); self.raw=packet.geometry; self.route.reset(); self.populate(); self.refresh()
    def apply_to_channel(self):
        panel=self.panels[self.channel.currentIndex()]
        if not Path(self.model.text()).is_file(): raise ValueError('Выберите существующие локальные веса.')
        from final_core import validate_policy
        panel.policy=validate_policy({**panel.policy,'path_enabled':True,'geometry_model':self.model.text(),'geometry_confidence':self.conf.value(),
            'geometry_image_size':self.size.value(),'geometry_switch':self.switch.isChecked(),'tracker':self.tracker.currentText(),'geometry_calibration_file':self.calibration_path.text()})
        panel._inference_settings_changed(); self.log.appendPlainText('Настройки применены. Источник/обучение не запускаются; сегментация пути используетCPU. На стрелке нужен свежий снимок и подтверждение.')
    def confirm_live(self):
        panel=self.panels[self.channel.currentIndex()]; raw=panel.path_raw
        if raw is None or raw.get('frame_sha256')!=self.raw.get('frame_sha256') or raw.get('model_sha256')!=self.raw.get('model_sha256') or raw['source_id']!=self.raw['source_id'] or raw['sequence']!=self.raw['sequence'] or raw['generation']!=panel.generation or not panel.health.get('assessed') or not 0<=(time.monotonic()-raw['captured_at'])*1000<=panel.policy['geometry_max_age_ms']: raise ValueError('Нужен свежий пригодный снимок этого канала и той же модели; старый выбор не переносится автоматически.')
        panel.route_state.confirm(raw,self.candidates.currentData(),self.reviewer.text()); self.log.appendPlainText('Выбор подтверждён с привязкой к источнику, поколению, кадру и маске. При изменении топологии потребуется повтор.')
    def choose_benchmark(self):
        path,_=QFileDialog.getOpenFileName(self,'Проверенные GT и одинаковые детекции',str(OUT),'JSON (*.json)')
        if path: self.compare_input.setText(path)
    def compare(self): self.start_worker({'action':'compare','input':self.compare_input.text()})
    def closeEvent(self,event):
        if self.process.state()!=QProcess.NotRunning: QMessageBox.information(self,'CPU-инструмент выполняется','Сначала завершите или остановите инструмент.'); event.ignore(); return
        event.accept()
