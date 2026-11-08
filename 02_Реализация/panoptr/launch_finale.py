"""Start the normal offline app and open the field workflow; never starts acquisition."""
from dataset_lab.bootstrap import setup
setup()
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from main import MainWindow
from trained_models import drone10_config
from launch_final import configure_font
def launch():
    app=QApplication(sys.argv);app.setStyle('Fusion');configure_font(app);config=drone10_config(Path(__file__).resolve().parent)
    window=MainWindow(config);window.show();QTimer.singleShot(0,window._open_finale);return app.exec()
if __name__=='__main__':sys.exit(launch())
