from dataset_lab.bootstrap import setup
setup()
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont,QFontDatabase
from camera_design_ui import CameraDesignDialog

if __name__=='__main__':
    if '--check' in sys.argv:
        import json
        from camera_design import calculate,preset
        report=calculate(preset('near')); print(json.dumps({'ready':True,'training_used':False,'design_checks':report['checks']},ensure_ascii=False)); sys.exit(0)
    app=QApplication(sys.argv); app.setStyle('Fusion')
    if Path('C:/Windows/Fonts/segoeui.ttf').exists(): QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf'); app.setFont(QFont('Segoe UI',9))
    window=CameraDesignDialog(); window.show(); sys.exit(app.exec())
