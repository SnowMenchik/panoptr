from bootstrap import setup
setup()
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QImage,QPixmap,QFont,QFontDatabase
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLineEdit,
    QPushButton,QComboBox,QCheckBox,QDoubleSpinBox,QSpinBox,QTabWidget,QPlainTextEdit,QLabel,QTableWidget,
    QTableWidgetItem,QFileDialog,QMessageBox,QSplitter)
from core import read,dump,decode_mask,boxes_for

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]/'10_Лаборатория_данных'

def configure_font(app):
    # Explicitly load a Cyrillic font, including for headless/offscreen QA.
    for path in [Path('C:/Windows/Fonts/segoeui.ttf'),Path('C:/Windows/Fonts/arial.ttf')]:
        if path.exists():
            identifier=QFontDatabase.addApplicationFont(str(path))
            families=QFontDatabase.applicationFontFamilies(identifier)
            if families:
                app.setFont(QFont(families[0],10)); return

class LabWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle('Panoptr — Лаборатория данных и моделей'); self.resize(1250,900)
        self.process=QProcess(self); self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_log); self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.current_report=None; self.session=None; self.current_operation=''; self.busy_buttons=[]
        central=QWidget(); layout=QVBoxLayout(central)
        note=QLabel('Изображения → проверка разметки → подготовка → обучение → оценка. Исходные данные сохраняются. Запуск обучения — отдельной кнопкой.')
        note.setWordWrap(True); layout.addWidget(note)
        self.tabs=QTabWidget(); layout.addWidget(self.tabs,1)
        self.build_import(); self.build_training(); self.build_evaluation()
        self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(170); layout.addWidget(self.log)
        self.state=QLabel('Готово к импорту. GPU для проверки данных не используется.'); layout.addWidget(self.state)
        stop=QPushButton('Остановить текущую операцию'); stop.clicked.connect(self.stop); layout.addWidget(stop)
        self.setCentralWidget(central)
        from annotation_ui import AnnotationWidget
        self.annotation_widget=AnnotationWidget(); self.tabs.addTab(self.annotation_widget,'4. Разметка')
        # Annotation has its own worker log; avoid two logs consuming the editor area.
        self.tabs.currentChanged.connect(lambda index:(self.log.setVisible(index!=3),self.state.setVisible(index!=3)))

    def path_field(self,form,label,directory=True):
        edit=QLineEdit(); row=QWidget(); box=QHBoxLayout(row); box.setContentsMargins(0,0,0,0); box.addWidget(edit)
        button=QPushButton('Выбрать'); box.addWidget(button)
        def pick():
            value=QFileDialog.getExistingDirectory(self,label,edit.text()) if directory else QFileDialog.getOpenFileName(self,label,edit.text())[0]
            if value: edit.setText(value)
        button.clicked.connect(pick); form.addRow(label,row); return edit

    def button(self,row,text,callback):
        b=QPushButton(text); b.clicked.connect(callback); row.addWidget(b); self.busy_buttons.append(b); return b

    def build_import(self):
        page=QWidget(); layout=QVBoxLayout(page); form=QFormLayout(); layout.addLayout(form)
        self.format=QComboBox(); self.format.addItem('Семантические PNG-маски','semantic_masks'); self.format.addItem('YOLO: прямоугольники','yolo_bbox'); self.format.addItem('COCO: прямоугольники','coco_bbox'); form.addRow('Формат',self.format)
        self.format.addItem('YOLO: полигоны объектов','yolo_segment')
        self.images=self.path_field(form,'Папка изображений'); self.annotations=self.path_field(form,'Папка масок / YOLO-меток')
        self.coco=self.path_field(form,'COCO JSON (для COCO)',False); self.pairs=self.path_field(form,'CSV пар и сцен (необязательно)',False)
        self.classes=QLineEdit('background, rock'); form.addRow('Классы в порядке ID, через запятую',self.classes)
        self.mapping=QLineEdit('{"0":0,"1":1}'); form.addRow('Значения масок → ID (-1 = ignore)',self.mapping)
        self.suffix=QLineEdit('_mask'); form.addRow('Удалять суффикс у масок/меток',self.suffix)
        self.groups=QLineEdit(); self.groups.setPlaceholderText('Например: ^(scene\\d+)_frame\\d+$'); form.addRow('Regex сцены: первая группа захвата',self.groups)
        self.independent=QCheckBox('Подтверждаю: изображения независимы, это не кадры одного видео/серии'); form.addRow(self.independent)
        self.ratio=QDoubleSpinBox(); self.ratio.setRange(.05,.8); self.ratio.setSingleStep(.05); self.ratio.setValue(.2); form.addRow('Доля групп в val',self.ratio)
        self.seed=QSpinBox(); self.seed.setRange(0,2147483647); self.seed.setValue(20261012); form.addRow('Seed разделения групп',self.seed)
        self.output=QLineEdit(str(WORK/'dataset_01')); form.addRow('Новая папка подготовленного набора',self.output)
        row=QHBoxLayout(); layout.addLayout(row)
        self.button(row,'Проверить данные',self.audit_data); self.button(row,'Подготовить набор и BAT',self.prepare_data)
        self.button(row,'Создать репетицию',self.demo); self.button(row,'Загрузить конфигурацию',self.load_config)
        self.button(row,'Видео → кадры',self.video_frames)
        self.summary=QLabel('PNG-маски сохраняются попиксельно. COCO в этой версии импортируется как bbox; crowd/ignore блокируют автоматическое преобразование.'); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        self.table=QTableWidget(0,3); self.table.setHorizontalHeaderLabels(['Изображение','Разметка','Сцена']); self.table.horizontalHeader().setStretchLastSection(True); self.table.itemSelectionChanged.connect(self.preview)
        self.table.setColumnWidth(0,400); self.table.setColumnWidth(1,400)
        layout.addWidget(self.table,1)
        previews=QHBoxLayout(); self.original=QLabel('Исходное изображение'); self.overlay=QLabel('Маска / bbox поверх изображения')
        for label in (self.original,self.overlay): label.setAlignment(Qt.AlignCenter); label.setMinimumHeight(180); previews.addWidget(label,1)
        layout.addLayout(previews); self.tabs.addTab(page,'1. Данные')

    def build_training(self):
        page=QWidget(); layout=QVBoxLayout(page); form=QFormLayout(); layout.addLayout(form)
        self.dataset=self.path_field(form,'Подготовленный набор'); self.weights=self.path_field(form,'Исходные веса для дообучения (необязательно)',False)
        self.device=QComboBox(); self.device.addItems(['cpu','cuda:0','cuda:1']); form.addRow('Устройство',self.device)
        self.epochs=QSpinBox(); self.epochs.setRange(1,10000); self.epochs.setValue(30); form.addRow('Эпохи',self.epochs)
        self.size=QSpinBox(); self.size.setRange(32,2048); self.size.setSingleStep(32); self.size.setValue(512); form.addRow('Размер входа',self.size)
        self.batch=QSpinBox(); self.batch.setRange(1,128); self.batch.setValue(2); form.addRow('Batch',self.batch)
        self.lr=QDoubleSpinBox(); self.lr.setDecimals(6); self.lr.setRange(.000001,.1); self.lr.setValue(.001); self.lr.setSingleStep(.0001); form.addRow('Learning rate',self.lr)
        text=QLabel('Семантические маски: небольшая U-Net как базовое решение; без checkpoint обучение начинается с нуля. Детекция: нужны локальные веса YOLO. CPU выбран по умолчанию. GPU использует общий замок с текущими обучениями Panoptr.'); text.setWordWrap(True); layout.addWidget(text)
        row=QHBoxLayout(); layout.addLayout(row)
        self.button(row,'Проверить готовность',lambda:self.training('check')); self.button(row,'Проба: 1 эпоха, до 16 кадров',lambda:self.training('pilot'))
        self.button(row,'Запустить обучение',lambda:self.training('train')); self.button(row,'Продолжить с last.pt',lambda:self.training('resume'))
        layout.addStretch(); self.tabs.addTab(page,'2. Обучение')

    def build_evaluation(self):
        page=QWidget(); layout=QVBoxLayout(page); form=QFormLayout(); layout.addLayout(form)
        self.eval_dataset=self.path_field(form,'Набор для сравнения'); self.first=self.path_field(form,'Первая модель',False); self.second=self.path_field(form,'Вторая модель',False)
        self.eval_images=self.path_field(form,'Изображения для предсказаний'); self.eval_output=QLineEdit(str(WORK/'predictions_01')); form.addRow('Новая папка экспорта',self.eval_output)
        self.eval_device=QComboBox(); self.eval_device.addItems(['cpu','cuda:0','cuda:1']); form.addRow('Устройство оценки',self.eval_device)
        row=QHBoxLayout(); layout.addLayout(row)
        self.button(row,'Сравнить на val',self.compare); self.button(row,'Экспорт предсказаний первой модели',self.predict)
        label=QLabel('Сегментация: IoU/Dice по классам, матрица ошибок; экспорт индексных PNG исходного размера, CSV/JSON и ZIP. Детекция: метрики YOLO и JSON bbox. Val используется повторно и не считается независимым тестом.'); label.setWordWrap(True); layout.addWidget(label); layout.addStretch(); self.tabs.addTab(page,'3. Оценка и экспорт')

    def cfg(self):
        kind=self.format.currentData(); names=[c.strip() for c in self.classes.text().split(',') if c.strip()]
        return {'format':kind,'images':self.images.text(),'annotations':self.coco.text() if kind=='coco_bbox' else self.annotations.text(),
            'classes':names,'mask_mapping':json.loads(self.mapping.text()) if kind=='semantic_masks' else {},
            'mask_suffix':self.suffix.text(),'pairs_csv':self.pairs.text(),'group_regex':self.groups.text(),
            'independent_images':self.independent.isChecked(),'val_fraction':self.ratio.value(),'seed':self.seed.value()}

    def snapshot(self):
        self.session=WORK/'sessions'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'); self.session.mkdir(parents=True)
        dump(self.session/'config.json',self.cfg()); return self.session/'config.json'

    def start(self,operation,arguments):
        if self.process.state()!=QProcess.NotRunning: raise ValueError('Операция ещё выполняется.')
        self.current_operation=operation; self.log.clear(); self.state.setText('Выполняется: '+operation)
        for b in self.busy_buttons: b.setEnabled(False)
        self.process.start(sys.executable,[str(ROOT/'cli.py'),*arguments])

    def safe(self,callback):
        try: callback()
        except Exception as exc: QMessageBox.warning(self,'Лаборатория',str(exc))

    def audit_data(self):
        self.safe(lambda:self.start('audit',['audit','--config',str(self.snapshot()),'--output',str(self.session/'audit.json')]))

    def prepare_data(self):
        def action():
            config=self.snapshot(); self.prepared_output=self.output.text()
            self.start('prepare',['prepare','--config',str(config),'--output',self.prepared_output])
        self.safe(action)

    def load_config(self):
        name=QFileDialog.getOpenFileName(self,'Конфигурация импорта',str(WORK),'JSON (*.json)')[0]
        if name: self.safe(lambda:self.set_cfg(read(name)))

    def set_cfg(self,cfg):
        self.format.setCurrentIndex(self.format.findData(cfg['format'])); self.images.setText(cfg['images'])
        (self.coco if cfg['format']=='coco_bbox' else self.annotations).setText(cfg['annotations'])
        self.classes.setText(', '.join(cfg['classes'])); self.mapping.setText(json.dumps(cfg.get('mask_mapping',{})))
        self.suffix.setText(cfg.get('mask_suffix','')); self.pairs.setText(cfg.get('pairs_csv','')); self.groups.setText(cfg.get('group_regex',''))
        self.independent.setChecked(cfg.get('independent_images',False)); self.ratio.setValue(cfg.get('val_fraction',.2))
        self.seed.setValue(cfg.get('seed',20261012))

    def demo(self):
        def action():
            self.demo_output=WORK/('rehearsal_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            self.start('demo',['demo','--output',str(self.demo_output)])
        self.safe(action)

    def video_frames(self):
        path=QFileDialog.getOpenFileName(self,'Видео для извлечения кадров','','Video (*.mp4 *.avi *.mov *.mkv);;All (*)')[0]
        if not path: return
        def action():
            target=WORK/('video_frames_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            self.start('frames',['frames','--video',path,'--output',str(target),'--every','30','--limit','300'])
        self.safe(action)

    def training(self,command):
        def action():
            root=Path(self.dataset.text()); meta=read(root/'dataset.json')
            settings={'epochs':self.epochs.value(),'image_size':self.size.value(),'batch':self.batch.value(),'lr':self.lr.value(),
                'device':self.device.currentText(),'workers':0,'seed':meta['seed'],'checkpoint':self.weights.text(),'task':meta['task']}
            # Resume reads the exact stored settings; form edits must not silently change an existing run.
            if command=='resume':
                stored=read(root/'training.json')
                if settings!=stored: raise ValueError('Для продолжения восстановите настройки из training.json; новые настройки означают новый эксперимент.')
            else: dump(root/'training.json',settings)
            self.start(command,[command,'--dataset',str(root)])
        self.safe(action)

    def compare(self):
        def action():
            WORK.mkdir(parents=True,exist_ok=True); target=WORK/('comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
            self.start('compare',['compare','--dataset',self.eval_dataset.text(),'--checkpoint',self.first.text(),'--second',self.second.text(),'--device',self.eval_device.currentText(),'--output',str(target)])
        self.safe(action)

    def predict(self):
        self.safe(lambda:self.start('predict',['predict','--checkpoint',self.first.text(),'--images',self.eval_images.text(),'--output',self.eval_output.text(),'--device',self.eval_device.currentText()]))

    def read_log(self):
        self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace'))

    def process_error(self,error):
        self.log.appendPlainText(self.process.errorString())
        if error==QProcess.FailedToStart:
            for b in self.busy_buttons: b.setEnabled(True)
            self.state.setText('Не удалось запустить процесс.')

    def finished(self,code,status):
        self.read_log()
        for b in self.busy_buttons: b.setEnabled(True)
        self.state.setText('Операция завершена.' if code==0 else 'Операция остановлена с ошибкой. Подробности в журнале.')
        def update():
            if code!=0: return
            if self.current_operation=='audit':
                report=read(self.session/'audit.json'); self.current_report=report
                self.summary.setText(f"Пар: {len(report['records'])}; групп: {report['group_count']}; ошибок: {len(report['issues'])}. Значения масок: {report['mask_values']}. " + ('Можно готовить набор.' if report['ready'] else 'Подготовка заблокирована.'))
                checked={r['image']:r for r in report['records']}
                self.preview_rows=[checked.get(r['image'],r) for r in report['pairs']]
                self.table.setRowCount(len(self.preview_rows))
                for i,r in enumerate(self.preview_rows):
                    for j,value in enumerate([r['image'],r['annotation'],r.get('group',r.get('explicit_group','Не проверено'))]):
                        item=QTableWidgetItem(Path(value).name if j<2 else value); item.setToolTip(value); self.table.setItem(i,j,item)
                self.log.appendPlainText('\n'.join(report['issues']))
                if self.preview_rows: self.table.selectRow(0)
            elif self.current_operation=='prepare':
                self.dataset.setText(self.prepared_output); self.eval_dataset.setText(self.prepared_output)
                settings=read(Path(self.prepared_output)/'training.json'); self.size.setValue(settings['image_size'])
                self.summary.setText('Набор подготовлен. BAT и training.json сохранены рядом с dataset.json. Обучение не запускалось.')
            elif self.current_operation=='demo':
                self.set_cfg(read(self.demo_output/'good_config.json')); self.output.setText(str(self.demo_output/'prepared'))
                self.log.appendPlainText('Для проверки ошибок загрузите broken_config.json из '+str(self.demo_output))
        self.safe(update)

    def preview(self):
        if not self.current_report or self.table.currentRow()<0: return
        self.safe(self._preview)

    def _preview(self):
        r=self.preview_rows[self.table.currentRow()]; cfg=self.current_report['configuration']
        with Image.open(r['image']) as im: image=im.convert('RGB')
        # Render thumbnails only; audit runs outside the UI process.
        scale=min(520/image.width,240/image.height,8)
        thumb=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.BILINEAR); overlay=thumb.copy()
        if cfg['format']=='semantic_masks':
            try: mask=decode_mask(r['annotation'],cfg['mask_mapping'],len(cfg['classes']))
            except ValueError:
                with Image.open(r['annotation']) as im: raw=im.convert('RGB'); raw.thumbnail((520,240))
                self.log.appendPlainText('Показана исходная маска: соответствие классов не подтверждено.')
                for label,im in [(self.original,thumb),(self.overlay,raw)]:
                    arr=np.array(im,dtype=np.uint8); label.setPixmap(QPixmap.fromImage(QImage(arr.data,im.width,im.height,arr.strides[0],QImage.Format_RGB888).copy()))
                return
            small=np.asarray(Image.fromarray(mask).resize(thumb.size,Image.Resampling.NEAREST)); base=np.array(thumb)
            colors=np.array([[int((i*71+70)%230+25),int((i*43+30)%230+25),int((i*113+10)%230+25)] for i in range(len(cfg['classes']))],dtype=np.uint8)
            keep=small!=255; color=np.zeros_like(base); color[keep]=colors[small[keep]]
            base[keep]=(base[keep]*.6+color[keep]*.4).astype(np.uint8); overlay=Image.fromarray(base)
        else:
            draw=ImageDraw.Draw(overlay)
            if cfg['format']=='yolo_segment':
                from core import segments_for
                for segment in r.get('segments',segments_for(r,cfg)):
                    points=np.array(segment[1:]).reshape(-1,2)*[thumb.width,thumb.height]
                    draw.polygon([tuple(p) for p in points],outline='red',width=2)
                rows=[]
            else: rows=r.get('boxes',boxes_for(r,cfg,image.width,image.height))
            for c,x,y,w,h in rows:
                box=((x-w/2)*thumb.width,(y-h/2)*thumb.height,(x+w/2)*thumb.width,(y+h/2)*thumb.height)
                draw.rectangle(box,outline='red',width=2); draw.text((box[0],box[1]),cfg['classes'][c],fill='red')
        for label,im in [(self.original,thumb),(self.overlay,overlay)]:
            arr=np.array(im,dtype=np.uint8); q=QImage(arr.data,im.width,im.height,arr.strides[0],QImage.Format_RGB888).copy(); label.setPixmap(QPixmap.fromImage(q))

    def stop(self):
        self.annotation_widget.stop()
        if self.process.state()!=QProcess.NotRunning or self.annotation_widget.process.state()!=QProcess.NotRunning:
            self.process.terminate(); self.state.setText('Остановка. Продолжение возможно с последней полностью сохранённой эпохи.')

    def closeEvent(self,event):
        if self.process.state()!=QProcess.NotRunning or self.annotation_widget.process.state()!=QProcess.NotRunning:
            # An active worker must not be killed silently when the window closes.
            QMessageBox.information(self,'Операция выполняется','Дождитесь завершения или нажмите «Остановить текущую операцию».'); event.ignore()
        else:
            try:
                if self.annotation_widget.dirty: self.annotation_widget.save()
                event.accept()
            except Exception as exc:
                QMessageBox.warning(self,'Правки не сохранены',str(exc)); event.ignore()

if __name__=='__main__':
    app=QApplication(sys.argv); app.setStyle('Fusion'); configure_font(app); window=LabWindow(); window.show(); sys.exit(app.exec())
