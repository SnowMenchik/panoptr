from dataset_lab.bootstrap import setup
setup()
import sys,json
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont,QFontDatabase
from virtual_stand import VirtualStandDialog
if __name__=='__main__':
    if '--check' in sys.argv:
        from stand_physics import engineering
        d=engineering(); print(json.dumps({'offline':True,'GPU_used':False,'line_rate_hz':d['line_rate_required_hz'],'line_rate_pass':d['line_rate_pass'],'deployment_verified':False})); sys.exit(0)
    app=QApplication(sys.argv); app.setStyle('Fusion'); QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf'); app.setFont(QFont('Segoe UI',9))
    window=VirtualStandDialog(); window.show(); sys.exit(app.exec())
