from pathlib import Path
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QLineEdit,QComboBox,QPlainTextEdit,QPushButton,QListWidget,QTabWidget,QWidget,QFormLayout,QMessageBox
from rail_reference import catalogue,advice,valid_codes

class RailReferenceDialog(QDialog):
    def __init__(self,store=None,event=None,parent=None):
        super().__init__(parent);self.store=store;self.event_record=event;self.setWindowTitle('Panoptr — визуальный признак и справочник рельсов');self.resize(1050,800)
        from ui_theme import create_competition_banner
        outer=QVBoxLayout(self);outer.addWidget(create_competition_banner('СПРАВОЧНИК ДЕФЕКТОВ РЕЛЬСОВ И ЗАПИСЬ ОБСЛЕДОВАНИЯ', 'Классификатор ОАО «РЖД» № 2499р • Разделение визуального признака ИИ и заключения дефектоскописта'));info=QLabel('№ 2499р, предоставленная редакция 13.11.2024. Класс модели и confidence не устанавливают нормативный код, ДР/ОДР или допустимую скорость.');info.setWordWrap(True);outer.addWidget(info)
        self.tabs=QTabWidget();outer.addWidget(self.tabs);page=QWidget();layout=QVBoxLayout(page)
        self.query=QLineEdit();self.query.setPlaceholderText('Найти код или способ обследования');layout.addWidget(self.query)
        self.entries=QListWidget();layout.addWidget(self.entries);self.methods=QPlainTextEdit();self.methods.setReadOnly(True);layout.addWidget(self.methods)
        self.catalog=catalogue();self.query.textChanged.connect(self.filter);self.entries.currentRowChanged.connect(self.select);self.tabs.addTab(page,'Справочник способов выявления');self.filter()
        page=QWidget();layout=QVBoxLayout(page);self.guidance=QLabel(advice(event['class_name'] if event else '')['text'] if event else 'Откройте событие для отдельной записи обследования.');self.guidance.setWordWrap(True);layout.addWidget(self.guidance)
        form=QFormLayout();layout.addLayout(form);self.status=QComboBox()
        for title,key in [('Визуальный признак, код не установлен','visual_only'),('Требуется измерение / дополнительное обследование','needs_measurement'),('Код записан специалистом после обследования','expert_recorded')]:self.status.addItem(title,key)
        form.addRow('Статус',self.status);self.code=QComboBox();self.code.addItem('Не установлен','')
        for code in sorted(valid_codes()):self.code.addItem(code,code)
        form.addRow('Код специалиста',self.code);self.reviewer=QLineEdit();form.addRow('Кто провёл обследование',self.reviewer);self.method=QLineEdit();form.addRow('Каким способом проверено',self.method);self.note=QPlainTextEdit();form.addRow('Измерения и обоснование',self.note)
        self.save_button=QPushButton('Сохранить отдельную запись обследования');self.save_button.setEnabled(store is not None and event is not None);self.save_button.clicked.connect(self.save);layout.addWidget(self.save_button)
        layout.addWidget(QLabel('Это не автоматическое заключение. Решение оператора по карточке и запись нормативного обследования хранятся отдельно.'))
        self.tabs.addTab(page,'Запись к событию')
        from ui_theme import apply_competition_theme
        apply_competition_theme(self)
        if event:
            values=event.get('technical_review',{});self.status.setCurrentIndex(max(0,self.status.findData(values.get('status','visual_only'))));self.code.setCurrentIndex(max(0,self.code.findData(values.get('code') or '')));self.reviewer.setText(values.get('reviewer',''));self.method.setText(values.get('method',''));self.note.setPlainText(values.get('note',''));self.tabs.setCurrentIndex(1)
    def filter(self):
        q=self.query.text().lower();self.filtered=[r for r in self.catalog['entries'] if q in ((r['code_context'] or '')+' '+r['methods']).lower()];self.entries.clear();self.entries.addItems([r['code_context'] or 'Раздел каталога' for r in self.filtered]);self.entries.setCurrentRow(0)
    def select(self,index):
        self.methods.setPlainText(self.filtered[index]['methods']+'\n\nРаздел 4.1: код зависит от места, типа/причины и положения по длине. Фотография не заменяет необходимые измерения или дефектоскопирование.' if 0<=index<len(self.filtered) else '')
    def save(self):
        try:
            self.store.save_review(self.event_record['event_id'],{'status':self.status.currentData(),'code':self.code.currentData(),'reviewer':self.reviewer.text(),'method':self.method.text(),'note':self.note.toPlainText()});self.accept()
        except (ValueError,OSError) as e:QMessageBox.warning(self,'Запись обследования',str(e))
