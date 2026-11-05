"""Local startup, CPU by default; no installation, downloads or automatic camera start."""
from dataset_lab.bootstrap import setup
setup()
import multiprocessing as mp
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont,QFontDatabase
from main import MainWindow,load_config

def configure_font(app):
    path=Path('C:/Windows/Fonts/segoeui.ttf')
    if path.exists():
        identifier=QFontDatabase.addApplicationFont(str(path)); families=QFontDatabase.applicationFontFamilies(identifier)
        if families: app.setFont(QFont(families[0],10))

def launch():
    mp.freeze_support(); config=load_config()
    if sum(flag in sys.argv for flag in ('--trained-models','--person07','--person08','--rock09','--drone10'))>1:
        raise SystemExit('Choose one profile: --trained-models, --person07, --person08, --rock09 or --drone10.')
    if '--trained-models' in sys.argv:
        from trained_models import trained_config
        config=trained_config(Path(__file__).resolve().parent)
    if '--person07' in sys.argv:
        from trained_models import person07_config
        config=person07_config(Path(__file__).resolve().parent)
    if '--person08' in sys.argv:
        from trained_models import person08_config
        config=person08_config(Path(__file__).resolve().parent)
    if '--rock09' in sys.argv:
        from trained_models import rock09_config
        config=rock09_config(Path(__file__).resolve().parent)
    if '--drone10' in sys.argv:
        from trained_models import drone10_config
        config=drone10_config(Path(__file__).resolve().parent)
    if '--check' in sys.argv:
        import json
        from final_tools import readiness
        report=readiness(Path(__file__).resolve().parent,[p['model'] for p in config['cameras']]); print(json.dumps(report,ensure_ascii=False,indent=2)); return 0 if report['ready'] else 1
    for channel in config['cameras']: channel['device']='GPU' if '--gpu' in sys.argv else 'CPU'
    app=QApplication(sys.argv); app.setStyle('Fusion'); configure_font(app)
    window=MainWindow(config); window.show(); return app.exec()

if __name__=='__main__': sys.exit(launch())
