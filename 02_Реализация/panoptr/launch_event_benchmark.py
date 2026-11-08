from dataset_lab.bootstrap import setup
setup()
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase,QFont
from event_benchmark_ui import EventBenchmarkDialog

if __name__=='__main__':
    app=QApplication(sys.argv); app.setStyle('Fusion')
    font=Path('C:/Windows/Fonts/segoeui.ttf')
    if font.exists(): QFontDatabase.addApplicationFont(str(font)); app.setFont(QFont('Segoe UI',9))
    window=EventBenchmarkDialog()
    if len(sys.argv)>1:
        import json
        path=Path(sys.argv[1]); window.load(json.loads(path.read_text(encoding='utf-8')),path)
    window.show(); sys.exit(app.exec())
