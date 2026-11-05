"""Offline CPU-only entry point. No model load, camera start or API listener by default."""
from dataset_lab.bootstrap import setup
setup()
import sys
from PySide6.QtWidgets import QApplication
from route_memory_ui import RouteMemoryWindow

if __name__=='__main__':
    app=QApplication(sys.argv); app.setStyle('Fusion')
    window=RouteMemoryWindow(); window.show(); sys.exit(app.exec())
