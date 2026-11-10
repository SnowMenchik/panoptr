"""Portable local dependencies and writable caches; no host environment fallback."""
import os,sys,json
from pathlib import Path
def setup():
    root=Path(__file__).resolve().parents[3]
    os.environ['PYTHONNOUSERSITE']='1'
    os.environ['PYTHONIOENCODING']='utf-8'
    os.environ['PYTHONUTF8']='1'
    os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['YOLO_OFFLINE']='true'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    for key,sub in [('YOLO_CONFIG_DIR','yolo'),('MPLCONFIGDIR','matplotlib'),('TORCH_HOME','torch'),('HF_HOME','huggingface')]:
        folder=root/'data/cache'/sub;folder.mkdir(parents=True,exist_ok=True);os.environ[key]=str(folder)
    settings=root/'data/cache/yolo/Ultralytics/settings.json'
    if settings.exists():
        values=json.loads(settings.read_text(encoding='utf-8'))
        values.update({key:str(root/'data'/sub) for key,sub in [('datasets_dir','datasets'),('weights_dir','weights'),('runs_dir','runs')]})
        settings.write_text(json.dumps(values,ensure_ascii=False,indent=2),encoding='utf-8')
    candidates=[Path(__file__).parents[1]]
    for folder in candidates:
        if str(folder) not in sys.path:sys.path.append(str(folder))
