"""Portable dispatcher; child tools inherit only the included runtime."""
from pathlib import Path
import os,subprocess,sys,runpy
ROOT=Path(__file__).resolve().parent
APP=ROOT/'02_Реализация/panoptr'
os.environ['PYTHONHOME']=str(ROOT/'runtime')
os.environ['PYTHONNOUSERSITE']='1'
os.environ.pop('PYTHONPATH',None)
os.environ['QT_PLUGIN_PATH']=str(ROOT/'runtime/Lib/site-packages/PySide6/plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=str(ROOT/'runtime/Lib/site-packages/PySide6/plugins/platforms')
sys.path.insert(0,str(APP))
from dataset_lab.bootstrap import setup
setup()
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True

def _ensure_dirs():
    for d in ['data/sessions','data/route_memory','data/event_benchmark','data/vsm_results','10_Лаборатория_данных']:
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    if os.name=='nt' and not (ROOT/'08_Обучение/runs').exists() and (ROOT/'link_runtime.ps1').is_file():
        subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'link_runtime.ps1')],check=False)

def main():
    _ensure_dirs()
    command=sys.argv[1] if len(sys.argv)>1 else 'main'
    args=sys.argv[2:]
    entries={'finale':'launch_finale.py','main':'launch_final.py','lab':'dataset_lab/main.py','routes':'launch_route_memory.py',
     'stand':'launch_virtual_stand.py','simulator':'vsm_simulator/main.py','camera':'launch_camera_design.py',
     'path':'launch_path_understanding.py','events':'launch_event_benchmark.py','stress_matrix':'run_vsm_stress_matrix.py','check':'portable_check.py'}
    if command not in entries:raise SystemExit('Unknown module')
    entry=APP/entries[command]
    sys.path.insert(0,str(entry.parent));os.chdir(entry.parent);sys.argv=[str(entry),*args]
    if command=='main':sys.argv.insert(1,'--drone10')
    runpy.run_path(str(entry),run_name='__main__')

if __name__=='__main__':main()
