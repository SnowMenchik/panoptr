from pathlib import Path
from datetime import datetime
import json,sys,copy
import numpy as np
from PIL import Image,ImageDraw
from PySide6.QtCore import Qt,QPointF,QRectF,QProcess,Signal
from PySide6.QtGui import QPainter,QColor,QPen,QImage,QPixmap
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLineEdit,QPushButton,QComboBox,QCheckBox,
    QDoubleSpinBox,QSpinBox,QLabel,QFileDialog,QMessageBox,QListWidget,QPlainTextEdit,QSplitter,QScrollArea)
from core import read,dump
from annotation import project,frame,save_frame,queue,add_object,edit_mask,mask_bbox
from annotation_models import MODES

ROOT=Path(__file__).resolve().parent; WORK=ROOT.parents[2]/'10_Лаборатория_данных'

class Canvas(QWidget):
    changed=Signal(str,object)
    def __init__(self):
        super().__init__(); self.setMinimumSize(400,280); self.image=None; self.objects=[]; self.mode='view'; self.selected=None
        self.points=[]; self.polygon=[]; self.start=None; self.end=None; self.stroke=[]; self.setMouseTracking(True)
    def set_frame(self,root,data):
        self.root=Path(root); self.data=data
        with Image.open(data['image']) as im: self.image=np.array(im.convert('RGB'))
        self.objects=data['objects']; self.points=[]; self.polygon=[]; self.start=None; self.end=None; self.update()
    def geometry_values(self):
        if self.image is None: return 1,0,0
        h,w=self.image.shape[:2]; scale=min(self.width()/w,self.height()/h)
        return scale,(self.width()-w*scale)/2,(self.height()-h*scale)/2
    def location(self,event):
        scale,x,y=self.geometry_values(); point=event.position()
        if self.image is None: return None
        px=(point.x()-x)/scale; py=(point.y()-y)/scale; h,w=self.image.shape[:2]
        return [max(0,min(w-1,px)),max(0,min(h-1,py))]
    def paintEvent(self,event):
        painter=QPainter(self); painter.fillRect(self.rect(),QColor('#202530'))
        if self.image is None:
            painter.setPen(Qt.white); painter.drawText(self.rect(),Qt.AlignCenter,'Создайте или откройте проект разметки'); return
        arr=self.image.copy()
        for index,obj in enumerate(self.objects):
            if obj.get('mask'):
                with Image.open(self.root/obj['mask']) as im: mask=np.array(im)>0
                color=np.array([(index*71+80)%255,(index*131+140)%255,(index*53+200)%255])
                arr[mask]=(arr[mask]*.6+color*.4).astype(np.uint8)
        h,w=arr.shape[:2]; scale,x,y=self.geometry_values(); painter.translate(x,y); painter.scale(scale,scale)
        painter.drawImage(0,0,QImage(arr.data,w,h,arr.strides[0],QImage.Format_RGB888).copy())
        for obj in self.objects:
            painter.setPen(QPen(QColor('#ffcb45') if obj['id']==self.selected else QColor('#50d5fa'),2/scale))
            a,b,c,d=obj['bbox']; painter.drawRect(QRectF(a,b,c-a,d-b))
        painter.setPen(QPen(QColor('#ffcb45'),2/scale))
        if self.start and self.end:
            a,b=self.start; c,d=self.end; painter.drawRect(int(min(a,c)),int(min(b,d)),int(abs(c-a)),int(abs(d-b)))
        for a,b in zip(self.polygon,self.polygon[1:]): painter.drawLine(QPointF(*a),QPointF(*b))
        for point,label in self.points:
            painter.setBrush(QColor('#46e390') if label else QColor('#f25c70')); painter.drawEllipse(QPointF(*point),4/scale,4/scale)
    def mousePressEvent(self,event):
        if self.image is None: return
        p=self.location(event)
        if self.mode=='points': self.points.append((p,1 if event.button()==Qt.LeftButton else 0)); self.update()
        elif self.mode=='polygon' and event.button()==Qt.LeftButton: self.polygon.append(p); self.update()
        elif self.mode in ('bbox','brush','erase') and event.button()==Qt.LeftButton: self.start=p; self.end=p; self.stroke=[p]
    def mouseMoveEvent(self,event):
        if self.start and event.buttons()&Qt.LeftButton:
            self.end=self.location(event); self.stroke.append(self.end); self.update()
    def mouseReleaseEvent(self,event):
        if not self.start: return
        p=self.location(event)
        if self.mode=='bbox':
            x,y=self.start; a,b=p; box=[min(x,a),min(y,b),max(x,a),max(y,b)]
            if box[2]>box[0] and box[3]>box[1]: self.changed.emit('bbox',box)
        elif self.mode in ('brush','erase'): self.changed.emit(self.mode,self.stroke+[p])
        self.start=None; self.end=None; self.update()
    def mouseDoubleClickEvent(self,event):
        if self.mode=='polygon' and len(self.polygon)>=3:
            self.changed.emit('polygon',self.polygon); self.polygon=[]; self.update()

class AnnotationWidget(QWidget):
    def __init__(self):
        super().__init__(); self.root=None; self.current=None; self.dirty=False; self.rows=[]; self.undo_stack=[]
        self.process=QProcess(self); self.process.setProcessChannelMode(QProcess.MergedChannels); self.process.readyReadStandardOutput.connect(self.read_log); self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        layout=QVBoxLayout(self); settings=QWidget(); form=QFormLayout(settings); scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(settings); scroll.setMaximumHeight(280); layout.addWidget(scroll)
        self.images=self.path_field(form,'Изображения'); self.project_path=QLineEdit(str(WORK/'annotation_project_01')); form.addRow('Проект разметки (новая папка)',self.project_path)
        self.classes=QLineEdit('person'); form.addRow('Классы объектов, без фона',self.classes)
        self.groups=QLineEdit(); self.groups.setPlaceholderText('Regex сцены; либо вложенная папка, либо CSV при подготовке'); form.addRow('Группы сцен',self.groups)
        self.independent=QCheckBox('Это независимые снимки, не кадры видео/аугментации'); form.addRow(self.independent)
        self.mode=QComboBox()
        for k,v in MODES.items(): self.mode.addItem(v,k)
        form.addRow('Вариант модели',self.mode)
        self.model=self.path_field(form,'Основные веса',False); self.sam=self.path_field(form,'SAM для связки',False); self.sam.setText(str(WORK/'annotation_models/mobile_sam.pt'))
        self.texts=QLineEdit('person'); form.addRow('Текстовые понятия (обычно английские)',self.texts)
        self.mapping=QLineEdit('{}'); form.addRow('Класс модели → ID проекта (JSON)',self.mapping)
        self.keep_unknown=QCheckBox('Оставлять неизвестные классы для ручного назначения'); form.addRow(self.keep_unknown)
        self.reference=self.path_field(form,'Пример YOLOE / видео',False)
        self.visual=QLineEdit('[]'); form.addRow('Примеры YOLOE: [{"bbox":[…],"class_id":0}]',self.visual)
        controls=QHBoxLayout(); self.device=QComboBox(); self.device.addItems(['cpu','cuda:0','cuda:1']); controls.addWidget(QLabel('Устройство')); controls.addWidget(self.device)
        self.conf=QDoubleSpinBox(); self.conf.setRange(0,1); self.conf.setSingleStep(.05); self.conf.setValue(.25); controls.addWidget(QLabel('Порог')); controls.addWidget(self.conf)
        self.size=QSpinBox(); self.size.setRange(32,2048); self.size.setSingleStep(32); self.size.setValue(640); controls.addWidget(QLabel('Размер')); controls.addWidget(self.size)
        self.limit=QSpinBox(); self.limit.setRange(0,100000); self.limit.setValue(10); controls.addWidget(QLabel('Предел (0 = все)')); controls.addWidget(self.limit); layout.addLayout(controls)
        buttons=QHBoxLayout(); layout.addLayout(buttons); self.actions=[]
        operations=[('Создать проект',self.create),('Открыть проект',self.open),('Проверить модель',lambda:self.worker('check')),
                         ('Разметить партию',lambda:self.worker('batch')),('SAM для выбранного',self.prompt),('Видео → проект',self.video),('Скачать публичные веса',self.download)]
        for i,(title,fn) in enumerate(operations):
            if i==4: buttons=QHBoxLayout(); layout.addLayout(buttons)
            self.add_button(buttons,title,fn)
        self.status=QLabel('Предложения модели требуют проверки. Кадр без находок тоже не является подтверждённым фоном.'); self.status.setWordWrap(True); layout.addWidget(self.status)
        split=QSplitter(); self.frames=QListWidget(); self.frames.currentRowChanged.connect(self.select_frame); split.addWidget(self.frames)
        editor=QWidget(); el=QVBoxLayout(editor); tools=QHBoxLayout(); el.addLayout(tools)
        self.tool=QComboBox()
        for title,key in [('Просмотр','view'),('Рамка','bbox'),('Полигон: двойной клик завершает','polygon'),('Кисть','brush'),('Ластик','erase'),('Точки SAM: ЛКМ + / ПКМ −','points')]: self.tool.addItem(title,key)
        self.tool.currentIndexChanged.connect(lambda:self.set_tool()); tools.addWidget(self.tool)
        self.class_id=QComboBox(); tools.addWidget(self.class_id); self.radius=QSpinBox(); self.radius.setRange(1,100); self.radius.setValue(8); tools.addWidget(QLabel('Кисть')); tools.addWidget(self.radius)
        self.modify=QCheckBox('Исправить выбранную рамку'); tools.addWidget(self.modify)
        self.canvas=Canvas(); self.canvas.changed.connect(self.canvas_edit); el.addWidget(self.canvas,1)
        self.objects=QListWidget(); self.objects.setMaximumHeight(90); self.objects.currentRowChanged.connect(self.select_object); el.addWidget(self.objects)
        edits=QHBoxLayout(); el.addLayout(edits)
        for i,(title,fn) in enumerate([('Сменить класс',self.reclassify),('Удалить объект',self.remove),('Отменить правку',self.undo),('Сохранить правки',self.save),('Подтвердить кадр',self.confirm),('Отклонить кадр',self.reject),('Рамка как пример',self.make_example)]):
            if i==4: edits=QHBoxLayout(); el.addLayout(edits)
            self.add_button(edits,title,fn,disable_busy=False)
        split.addWidget(editor); split.setStretchFactor(1,3); layout.addWidget(split,1)
        exports=QHBoxLayout(); layout.addLayout(exports); self.export_kind=QComboBox()
        for title,key in [('YOLO bbox','yolo_bbox'),('YOLO polygons','yolo_segment'),('COCO + точные RLE-маски','coco'),('Семантические PNG','semantic')]: self.export_kind.addItem(title,key)
        exports.addWidget(self.export_kind); self.overlap=QComboBox()
        for title,key in [('Перекрытие: остановить','error'),('Перекрытие: последний объект','last'),('Перекрытие: ignore','ignore')]: self.overlap.addItem(title,key)
        exports.addWidget(self.overlap); self.export_path=QLineEdit(str(WORK/'annotation_export_01')); exports.addWidget(self.export_path)
        self.add_button(exports,'Экспорт подтверждённых',self.export); self.add_button(exports,'Остановить',self.stop,disable_busy=False)
        self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(100); layout.addWidget(self.log)
        self.mode.currentIndexChanged.connect(self.mode_preset); self.mode_preset()

    def mode_preset(self):
        mode=self.mode.currentData(); names={'detector_sam':'mobile_sam.pt','sam_prompt':'mobile_sam.pt','sam_everything':'mobile_sam.pt',
            'fastsam_prompt':'FastSAM-s.pt','fastsam_everything':'FastSAM-s.pt','fastsam_text':'FastSAM-s.pt','world_text':'yolov8s-worldv2.pt',
            'yoloe_text':'yoloe-26s-seg.pt','yoloe_visual':'yoloe-26s-seg.pt','yoloe_free':'yoloe-26s-seg-pf.pt','sam2_video':'sam2.1_t.pt',
            'sam3_text':'sam3.pt','sam3_exemplar':'sam3.pt','sam3_video':'sam3.pt','sam3_video_text':'sam3.pt'}
        if mode in ('yolo_bbox','detector_sam'): candidate=WORK.parent/'09_Дополнительные_данные_2026-10-11/models/yolo26s.pt'
        elif mode=='yolo_segment': candidate=ROOT.parent/'models/yolo26s_straight_track_seg_best.pt'
        elif mode=='semantic_model': candidate=None
        else: candidate=WORK/'annotation_models'/names[mode]
        self.model.setText(str(candidate) if candidate else '')
        self.status.setText('SAM 3: выберите свои официально полученные веса; в комплект они не входят.' if 'sam3' in mode else 'Выберите веса для своей задачи. Автоматические предложения нужно просмотреть и подтвердить.')

    def path_field(self,form,label,directory=True):
        edit=QLineEdit(); row=QWidget(); h=QHBoxLayout(row); h.setContentsMargins(0,0,0,0); h.addWidget(edit); b=QPushButton('Выбрать'); h.addWidget(b)
        def pick():
            p=QFileDialog.getExistingDirectory(self,label) if directory else QFileDialog.getOpenFileName(self,label)[0]
            if p: edit.setText(p)
        b.clicked.connect(pick); form.addRow(label,row); return edit
    def add_button(self,row,title,fn,disable_busy=True):
        b=QPushButton(title); b.clicked.connect(lambda:self.safe(fn)); row.addWidget(b)
        if disable_busy: self.actions.append(b)
    def safe(self,fn):
        try: return fn()
        except Exception as exc: QMessageBox.warning(self,'Разметка',str(exc))
    def config(self):
        return {'project':self.project_path.text(),'images':self.images.text(),'classes':[v.strip() for v in self.classes.text().split(',') if v.strip()],
          'mode':self.mode.currentData(),'model':self.model.text(),'sam_model':self.sam.text(),'texts':[v.strip() for v in self.texts.text().split(',') if v.strip()],
          'class_map':json.loads(self.mapping.text()),'keep_unknown':self.keep_unknown.isChecked(),'device':self.device.currentText(),'confidence':self.conf.value(),'image_size':self.size.value(),
          'limit':self.limit.value(),'group_regex':self.groups.text(),'independent_images':self.independent.isChecked(),
          'reference_image':self.reference.text(),'video':self.reference.text(),'visual_boxes':json.loads(self.visual.text()),
          'exemplar_boxes':[b['bbox'] for b in json.loads(self.visual.text())]}
    def worker(self,action,extra=None):
        if self.process.state()!=QProcess.NotRunning: raise ValueError('Разметка ещё выполняется.')
        if self.dirty: self.save()
        cfg={**self.config(),'action':action,**(extra or {})}
        session=WORK/'annotation_sessions'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'); session.mkdir(parents=True); dump(session/'config.json',cfg)
        self.operation=action; self.log.clear()
        for b in self.actions: b.setEnabled(False)
        self.process.start(sys.executable,[str(ROOT/'annotation_cli.py'),str(session/'config.json')])
    def create(self): self.worker('create')
    def open(self):
        if self.dirty: self.save()
        p=QFileDialog.getExistingDirectory(self,'Проект разметки',self.project_path.text())
        if p: self.project_path.setText(p); self.load_project()
    def load_project(self):
        selected_id=self.current['id'] if self.current and self.root==Path(self.project_path.text()).resolve() else None
        self.dirty=False; self.current=None
        self.root=Path(self.project_path.text()).resolve(); meta=project(self.root); self.classes.setText(', '.join(meta['classes'])); self.images.setText(meta['image_root'])
        self.class_id.clear(); self.class_id.addItems(meta['classes']); self.rows=queue(self.root)
        self.frames.blockSignals(True); self.frames.clear()
        for r in self.rows: self.frames.addItem(f"{r['status']} | {len(r['objects'])} | {r['relative_image']}")
        selected_index=next((i for i,r in enumerate(self.rows) if r['id']==selected_id),0)
        self.frames.blockSignals(False); self.frames.setCurrentRow(selected_index); self.select_frame(selected_index)
    def select_frame(self,index):
        if not self.root or not 0<=index<len(self.rows): return
        def action():
            if self.dirty: self.save()
            self.current=frame(self.root,self.rows[index]['id']); self.dirty=False; self.undo_stack=[]; self.canvas.set_frame(self.root,self.current); self.refresh_objects()
        self.safe(action)
    def refresh_objects(self):
        selected_id=self.canvas.selected
        self.objects.blockSignals(True); self.objects.clear(); names=project(self.root)['classes']
        for obj in self.current['objects']:
            c=obj['class_id']; self.objects.addItem(f"{names[c] if c is not None else 'Класс не назначен'} | {'mask' if obj.get('mask') else 'bbox'} | {obj.get('confidence')}")
        self.objects.blockSignals(False); self.canvas.objects=self.current['objects']
        index=next((i for i,o in enumerate(self.current['objects']) if o['id']==selected_id),-1)
        self.objects.setCurrentRow(index); self.canvas.update()
    def refresh_frame_label(self):
        if not self.current: return
        for i,row in enumerate(self.rows):
            if row['id']==self.current['id']:
                self.rows[i]=copy.deepcopy(self.current)
                self.frames.item(i).setText(f"{self.current['status']} | {len(self.current['objects'])} | {self.current['relative_image']}")
                break
    def selected(self):
        i=self.objects.currentRow(); return self.current['objects'][i] if self.current and 0<=i<len(self.current['objects']) else None
    def select_object(self,index):
        obj=self.selected(); self.canvas.selected=obj['id'] if obj else None
        if obj and obj['class_id'] is not None: self.class_id.setCurrentIndex(obj['class_id'])
        self.canvas.update()
    def set_tool(self): self.canvas.mode=self.tool.currentData()
    def before_edit(self):
        if not self.current: raise ValueError('Выберите кадр.')
        if self.process.state()!=QProcess.NotRunning: raise ValueError('Дождитесь завершения модели перед правкой.')
        self.undo_stack.append(copy.deepcopy(self.current)); self.current['status']='edited'; self.dirty=True
    def canvas_edit(self,kind,value):
        def action():
            obj=self.selected(); self.before_edit()
            if kind=='bbox':
                if self.modify.isChecked() and obj:
                    if obj.get('mask'): raise ValueError('Рамка маски вычисляется из пикселей; исправьте маску кистью/SAM.')
                    obj['bbox']=value; obj['source']='manual_corrected'
                else: add_object(self.root,self.current,self.class_id.currentIndex(),bbox=value)
            elif kind=='polygon':
                mask=Image.new('L',(self.current['width'],self.current['height'])); ImageDraw.Draw(mask).polygon([tuple(p) for p in value],fill=255)
                add_object(self.root,self.current,self.class_id.currentIndex(),mask=np.array(mask)>0)
            else:
                if not obj: raise ValueError('Выберите объект для кисти/ластика.')
                edit_mask(self.root,self.current,obj,value,self.radius.value(),erase=kind=='erase')
            self.refresh_objects()
        self.safe(action)
    def reclassify(self):
        obj=self.selected()
        if not obj: raise ValueError('Выберите объект.')
        self.before_edit(); obj['class_id']=self.class_id.currentIndex(); obj['source']='manual_corrected'; self.refresh_objects()
    def remove(self):
        obj=self.selected()
        if not obj: raise ValueError('Выберите объект.')
        self.before_edit(); self.current['objects'].remove(obj); self.refresh_objects()
    def undo(self):
        if self.undo_stack:
            revision=self.current['revision']; self.current=self.undo_stack.pop(); self.current['revision']=revision; self.dirty=True; self.canvas.data=self.current; self.refresh_objects()
    def save(self):
        if not self.current: return
        save_frame(self.root,self.current,'manual_edit'); self.dirty=False; self.refresh_frame_label(); self.status.setText('Правки сохранены. Для обучения кадр нужно подтвердить.')
    def confirm(self):
        if not self.current: raise ValueError('Выберите кадр.')
        if self.process.state()!=QProcess.NotRunning: raise ValueError('Дождитесь завершения модели.')
        updated=copy.deepcopy(self.current); updated['status']='confirmed'; save_frame(self.root,updated,'human_confirmation'); self.current=updated; self.dirty=False; self.status.setText('Кадр подтверждён, включая проверку пропущенных объектов и фона.')
        self.refresh_frame_label(); self.refresh_objects()
    def reject(self):
        if not self.current: return
        if self.process.state()!=QProcess.NotRunning: raise ValueError('Дождитесь завершения модели.')
        self.current['status']='rejected'; save_frame(self.root,self.current,'human_rejection'); self.dirty=False
        self.refresh_frame_label()
    def prompt(self):
        if not self.current: raise ValueError('Выберите кадр.')
        obj=self.selected(); points=self.canvas.points
        prompt={'class_id':self.class_id.currentIndex(),'points':[p for p,_ in points] or None,'labels':[v for _,v in points] or None,
                'boxes':[obj['bbox']] if obj else None,'replace_object':obj['id'] if obj else None}
        self.worker('batch',{'frame_id':self.current['id'],'prompt':prompt})
    def make_example(self):
        obj=self.selected()
        if not obj: raise ValueError('Выберите рамку объекта.')
        self.reference.setText(self.current['image']); self.visual.setText(json.dumps([{'bbox':obj['bbox'],'class_id':self.class_id.currentIndex()}]))
    def video(self):
        obj=self.selected(); points=self.canvas.points
        self.worker('video',{'prompt_class':self.class_id.currentIndex(),'points':[p for p,_ in points] or None,'point_labels':[v for _,v in points] or None,
                              'boxes':[obj['bbox']] if obj else [b['bbox'] for b in json.loads(self.visual.text())] or None,'limit':self.limit.value() or 300})
    def download(self): self.worker('download',{'output':str(WORK/'annotation_models')})
    def export(self): self.worker('export',{'output':self.export_path.text(),'export_kind':self.export_kind.currentData(),'overlap':self.overlap.currentData()})
    def read_log(self): self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace'))
    def process_error(self,error):
        self.log.appendPlainText(self.process.errorString())
        if error==QProcess.FailedToStart:
            for b in self.actions: b.setEnabled(True)
            self.status.setText('Не удалось запустить обработчик; подробности в журнале.')
    def finished(self,code,status):
        self.read_log()
        for b in self.actions: b.setEnabled(True)
        if code==0 and self.operation in ('create','batch','video'): self.safe(self.load_project)
        self.status.setText('Операция завершена; предложения модели требуют проверки.' if code==0 else 'Ошибка: подробности в журнале.')
    def stop(self):
        if self.process.state()!=QProcess.NotRunning: self.process.terminate()
