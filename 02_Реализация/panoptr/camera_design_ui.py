"""Camera geometry calculator; isolated from capture, training and GPU."""
from pathlib import Path
from datetime import datetime
import json
from PySide6.QtCore import Qt,QRectF
from PySide6.QtGui import QPainter,QPen,QColor,QPolygonF
from PySide6.QtWidgets import (QDialog,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QPushButton,
    QComboBox,QDoubleSpinBox,QSpinBox,QScrollArea,QTableWidget,QTableWidgetItem,QFileDialog,QMessageBox,QHeaderView,QSplitter)
from camera_design import preset,calculate,export_report,validate

ROOT=Path(__file__).resolve().parent; WORK=ROOT.parents[1]/'13_Расчёт_камеры'

class ProjectionCanvas(QWidget):
    def __init__(self): super().__init__(); self.report=None; self.setMinimumSize(360,220)
    def paintEvent(self,event):
        painter=QPainter(self); painter.fillRect(QRectF(0,0,self.width(),self.height()),QColor('#0F1E36'))
        if not self.report: return
        p=self.report['inputs']; target=self.report['target']; w,h=p['width_px'],p['height_px']
        scale=min((self.width()-40)/w,(self.height()-45)/h); ox=(self.width()-w*scale)/2; oy=25
        rect=QRectF(ox,oy,w*scale,h*scale); painter.setPen(QPen(QColor('#92aec9'),1)); painter.drawRect(rect)
        painter.setPen(QColor('white')); painter.drawText(15,17,'Проекция объекта и движение углов за выдержку')
        from PySide6.QtCore import QPointF
        painter.save(); painter.setClipRect(rect)
        if target.get('projectable'):
            poly=QPolygonF([QPointF(ox+x*scale,oy+y*scale) for x,y in target['sensor_polygon_px']])
            painter.setPen(QPen(QColor('#5ad6a5') if target['fully_visible'] else QColor('#fdb960'),2)); painter.drawPolygon(poly)
            trails=target.get('exposure_trails_sensor_px')
            if trails:
                for i in range(4):
                    painter.setPen(QPen(QColor('#ffca68'),2)); trail=QPolygonF([QPointF(ox+t[i][0]*scale,oy+t[i][1]*scale) for t in trails]); painter.drawPolyline(trail)
        painter.restore(); painter.setPen(QColor('white'))
        painter.drawText(15,self.height()-8,'Схема в пикселях сенсора; не изображение с камеры')

FIELDS=[
 ('speed_kmh','Скорость, км/ч',0,1000),('camera_height_m','Высота камеры, м',.001,1000),
 ('pitch_deg','Наклон вниз от горизонта, °',0,90),('distance_m','Расчётная точка вдоль пути, м',-100000,100000),
 ('lateral_offset_m','Смещение объекта вправо, м',-10000,10000),
 ('object_width_m','Ширина объекта, м',.000001,10000),('object_extent_m','Длина / высота объекта, м',.000001,10000),
 ('width_px','Ширина исходного кадра, px',1,32768),('height_px','Высота исходного кадра, px',1,32768),
 ('model_size_px','Длинная сторона входа модели, px',1,32768),('hfov_deg','Горизонтальный угол обзора, °',.1,160),
 ('sensor_width_mm','Ширина сенсора, мм (0 = неизвестно)',0,100),
 ('exposure_ms','Выдержка, мкс',0,1000000),('capture_fps','Номинальная частота камеры, кадров/с',.01,100000),
 ('analysis_fps','Запрошенный темп анализа, кадров/с',0,100000),('predict_ms','Время predict одного кадра, мс',0,100000),
 ('pipeline_ms','Остальная задержка тракта, мс',0,100000),('pitch_rate_deg_s','Угловая скорость pitch, °/с',-1000,1000),
 ('readout_ms','Полное считывание rolling shutter, мс',0,10000),('min_pixels','Минимум по короткой стороне, px модели',.01,10000),
 ('max_blur_px','Допустимый смаз, px модели',0,10000),('max_readout_motion_px','Допустимое движение за readout, px модели',0,10000),('required_observations','Требуемое число наблюдений',1,32768),
 ('required_lead_s','Требуемое время до положения объекта, с',0,10000),
 ('range_min_m','Ближняя граница поиска впереди, м',0,100000),('range_max_m','Предел поиска вдоль пути, м',.001,100000),
 ('fx_px','Калибровка fx, px',.01,1000000),('fy_px','Калибровка fy, px',.01,1000000),
 ('cx_px','Калибровка cx, px',0,32768),('cy_px','Калибровка cy, px',0,32768)]
INTS={'width_px','height_px','model_size_px','required_observations'}

class CameraDesignDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent); self.setWindowTitle('Panoptr — инженерный расчёт камеры'); self.resize(1260,920)
        self.profiles={'near':preset('near'),'forward':preset('forward')}; self.active='near'; self.report=None
        from ui_theme import apply_competition_theme, create_competition_banner
        layout=QVBoxLayout(self); layout.addWidget(create_competition_banner('ИНЖЕНЕРНЫЙ РАСЧЁТ КАМЕРЫ И ПОКРЫТИЯ ВСМ', 'Расчёт смаза при 400 км/ч, выдержки, разрешения и числа наблюдений в зоне видимости')); row=QHBoxLayout(); layout.addLayout(row)
        self.scenario=QComboBox(); self.scenario.addItem('Детали пути вблизи: плоский дефект','near'); self.scenario.addItem('Препятствия впереди: вертикальный объект','forward'); row.addWidget(self.scenario,1)
        self.button(row,'Исходный пример',self.reset); self.button(row,'Загрузить параметры',self.load_inputs); self.button(row,'Сохранить параметры',self.save_inputs)
        split=QSplitter(); layout.addWidget(split,1); scroll=QScrollArea(); scroll.setWidgetResizable(True); inputs=QWidget(); form=QFormLayout(inputs); scroll.setWidget(inputs); split.addWidget(scroll)
        self.fields={}; self.labels={}
        self.optics=QComboBox(); self.optics.addItem('По углу обзора: идеальная камера','fov'); self.optics.addItem('Из fx/fy/cx/cy: введённая калибровка','calibrated'); form.addRow('Модель оптики',self.optics)
        self.shutter=QComboBox(); self.shutter.addItem('Global shutter','global'); self.shutter.addItem('Rolling shutter','rolling'); form.addRow('Затвор',self.shutter)
        for key,label,lo,hi in FIELDS:
            widget=QSpinBox() if key in INTS else QDoubleSpinBox()
            widget.setRange(lo,hi)
            if key not in INTS: widget.setDecimals(6); widget.setSingleStep(.1)
            widget.setMinimumWidth(140); widget.setMaximumWidth(200)
            q=QLabel(label); q.setWordWrap(True); q.setMinimumWidth(230); form.addRow(q,widget); self.fields[key]=widget; self.labels[key]=q
        results=QWidget(); right=QVBoxLayout(results); split.addWidget(results); split.setSizes([560,700])
        self.canvas=ProjectionCanvas(); right.addWidget(self.canvas)
        self.summary=QLabel(); self.summary.setWordWrap(True); right.addWidget(self.summary)
        self.table=QTableWidget(0,2); self.table.setHorizontalHeaderLabels(['Показатель / требование','Расчёт']); self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.Stretch); self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeToContents); right.addWidget(self.table,1)
        self.notes=QLabel(); self.notes.setWordWrap(True); right.addWidget(self.notes)
        row=QHBoxLayout(); layout.addLayout(row); self.button(row,'Рассчитать',self.recalculate); self.button(row,'Экспорт отчёта и графика',self.export)
        info=QLabel('Пороги — ваши проектные цели, не норматив ВСМ и не гарантия Recall. Идеальная pinhole-геометрия, неподвижный объект, равномерное движение. Увеличение изображения не восстанавливает детали. Параметры не применяются к работающим камерам или обучению.'); info.setWordWrap(True); layout.addWidget(info)
        apply_competition_theme(self)
        self.scenario.currentIndexChanged.connect(self.switch); self.fill(self.profiles['near']); self.recalculate()
    def safe(self,fn):
        try: return fn()
        except Exception as exc: QMessageBox.warning(self,'Расчёт камеры',str(exc))
    def button(self,row,text,fn):
        b=QPushButton(text); b.clicked.connect(lambda:self.safe(fn)); row.addWidget(b)
    def collect(self):
        p={k:v.value() for k,v in self.fields.items()}; p['exposure_ms']/=1000
        p.update({'scenario':self.active,'intrinsics_mode':self.optics.currentData(),'shutter':self.shutter.currentData()}); return validate(p)
    def fill(self,p):
        for key,widget in self.fields.items(): widget.setValue(p[key]*1000 if key=='exposure_ms' else p[key])
        self.optics.setCurrentIndex(self.optics.findData(p['intrinsics_mode'])); self.shutter.setCurrentIndex(self.shutter.findData(p['shutter']))
        self.labels['object_extent_m'].setText('Длина дефекта вдоль пути, м' if p['scenario']=='near' else 'Высота препятствия, м')
    def switch(self,*_):
        try: self.profiles[self.active]=self.collect()
        except ValueError: pass
        self.active=self.scenario.currentData(); self.fill(self.profiles[self.active]); self.safe(self.recalculate)
    def reset(self): self.profiles[self.active]=preset(self.active); self.fill(self.profiles[self.active]); self.recalculate()
    def recalculate(self):
        self.report=calculate(self.collect()); self.profiles[self.active]=self.report['inputs']; r=self.report; t=r['target']; c=r['coverage']; m=r['motion']; q=r['requirements']
        def num(x,unit=''): return 'N/A' if x is None else f'{x:.4g} {unit}'.strip()
        def dims(a,b): return num(a)+' × '+num(b)+' px'
        rows=[('Объект на исходном сенсоре',dims(t.get('sensor_width_px'),t.get('sensor_height_px'))),
            ('Объект после уменьшения для модели',dims(t.get('model_width_px'),t.get('model_height_px'))),
            ('Смаз: сенсор / вход модели',num(t.get('sensor_blur_px'))+' / '+num(t.get('model_blur_px'))+' px'),
            ('Фокус / вертикальный угол обзора',num(r['intrinsics']['lens_focal_mm'],'мм')+' / '+num(r['intrinsics']['actual_vfov_deg'],'°')),
            ('Читаемая длина / время прохода',num(c['readable_length_m'],'м')+' / '+num(c['readable_duration_s'],'с')),
            ('Эффективный темп анализа (расчёт)',num(c['effective_analysis_fps'],'кадров/с')),
            ('Ожидаемое число наблюдений N',num(c['expected_observations'])),
            ('Границы числа отсчётов при равномерной фазе',str(c['uniform_phase_min_observations'])+' … '+str(c['uniform_phase_max_observations'])),
            ('Шаг между анализами',num(c['sample_spacing_m'],'м')),
            ('Перемещение за выдержку',num(m['distance_during_exposure_m'],'м')),
            ('Перемещение за predict',num(m['distance_during_predict_m'],'м')),
            ('Перемещение за predict + остальной тракт',num(m['distance_during_pipeline_and_predict_m'],'м')),
            ('Первый ответ с ожиданием отсчёта (оценка)',num(m['worst_uniform_first_result_s_estimate'],'с')),
            ('Путь до первого ответа с ожиданием (оценка)',num(m['distance_before_first_result_m_estimate'],'м')),
            ('Время до положения объекта от начала читаемости',num(m['lead_at_first_readable_range_s_estimate'],'с')),
            ('Межстрочное движение за полное считывание',num(m['rolling_full_readout_motion_envelope_model_px'],'px модели')),
            ('Предельная выдержка у расчётной точки',num(q['max_exposure_ms_at_design_target']*1000 if q['max_exposure_ms_at_design_target'] is not None else None,'мкс')),
            ('Минимальная ширина сенсора: текущие FOV/пропорции',num(q['sensor_width_px_at_same_aspect_and_fov'],'px')),
            ('Нужная длинная сторона модели: текущая геометрия',num(q['model_size_px_for_current_angular_geometry'],'px')),
            ('Верхний угол обзора для нужных px модели',num(q['max_hfov_deg_for_current_model_pixels'],'°')),
            ('Минимальный темп по текущему читаемому окну',num(q['min_analysis_fps_for_required_observations'],'кадров/с')),
            ('Нижняя граница темпа только по геометрии',num(q['min_analysis_fps_geometry_only_lower_bound'],'кадров/с')),
            ('Predict при таком темпе, без остальной задержки',num(q['max_predict_ms_if_serial_and_zero_extra_latency'],'мс')),
            ('Наклон для центрирования объекта',num(q['suggested_pitch_to_target_centre_deg'],'°')),
            ('Поток RGB8 / mono8 без сжатия',num(r['data_rate']['raw_RGB8_MB_s'])+' / '+num(r['data_rate']['raw_mono8_MB_s'])+' МБ/с')]
        checks={'target_inside_frame':'Объект целиком в кадре за выдержку','target_resolution':'Достаточно пикселей у расчётной точки',
            'target_exposure_blur':'Смаз ниже выбранного порога','observation_count':'Минимальное число отсчётов при равномерной фазе',
            'response_within_contiguous_window':'Ответ успевает в непрерывное читаемое окно','requested_lead_time':'Выполнено введённое время предупреждения',
            'rolling_readout_motion_budget':'Движение за readout известно и укладывается в бюджет',
            'nonzero_exposure':'Положительная физическая выдержка','nonzero_predict_time':'Положительное время обработки'}
        rows += [(checks.get(k,k),'Выполнено' if v else 'Не выполнено') for k,v in r['checks'].items()]
        self.table.setRowCount(len(rows))
        for i,(label,value) in enumerate(rows):
            self.table.setItem(i,0,QTableWidgetItem(label)); self.table.setItem(i,1,QTableWidgetItem(value))
        self.summary.setText('Все рассчитанные проектные условия выполнены.' if r['all_design_checks_met'] else 'Есть невыполненные проектные условия: проверьте таблицу. Это расчёт, а не измерение текущего оборудования.')
        self.notes.setText('\n'.join(r['warnings']) if r['warnings'] else 'Проверьте освещение, реальную калибровку, затвор и измеренную скорость обработки. Эти параметры не оценены по одной геометрии.')
        self.canvas.report=r; self.canvas.update()
    def save_inputs(self):
        p=self.collect(); path=QFileDialog.getSaveFileName(self,'Параметры расчёта',str(WORK/(self.active+'.json')),'JSON (*.json)')[0]
        if path:
            from final_core import save_json
            save_json(path,p)
    def load_inputs(self):
        path=QFileDialog.getOpenFileName(self,'Параметры расчёта',str(WORK),'JSON (*.json)')[0]
        if path:
            p=validate(json.loads(Path(path).read_text(encoding='utf-8'))); self.active=p['scenario']; self.profiles[self.active]=p
            self.scenario.blockSignals(True); self.scenario.setCurrentIndex(self.scenario.findData(self.active)); self.scenario.blockSignals(False); self.fill(p); self.recalculate()
    def export(self):
        self.recalculate(); base=QFileDialog.getExistingDirectory(self,'Папка для нового отчёта',str(WORK))
        if base:
            path=Path(base)/(self.active+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')); export_report(self.report,path)
            QMessageBox.information(self,'Расчёт сохранён',str(path))
