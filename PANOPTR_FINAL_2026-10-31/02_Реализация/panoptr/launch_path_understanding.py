from dataset_lab.bootstrap import setup
setup()
import sys,json
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont,QFontDatabase
from path_ui import PathUnderstandingDialog
if __name__=='__main__':
    if '--check' in sys.argv:
        from path_benchmark import compare,synthetic_tracking_fixture
        print(json.dumps({'CPU_only':True,'metrics':compare(synthetic_tracking_fixture())['metrics']},ensure_ascii=False)); sys.exit(0)
    app=QApplication(sys.argv); app.setStyle('Fusion'); QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf'); app.setFont(QFont('Segoe UI',9)); window=PathUnderstandingDialog(); window.show(); sys.exit(app.exec())
