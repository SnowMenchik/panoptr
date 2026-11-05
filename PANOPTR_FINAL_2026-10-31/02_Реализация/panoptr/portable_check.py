"""Offline self-check of the runtime, models and spawned worker beside the app."""
from pathlib import Path
import sys,os,json,time,hashlib,multiprocessing as mp
ROOT=Path(__file__).resolve().parents[2]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def spawn_echo(queue):queue.put({'prefix':sys.prefix,'executable':sys.executable})

def main():
    mp.freeze_support()
    from dataset_lab.bootstrap import setup
    setup()
    import importlib,numpy as np,cv2,torch
    names=['numpy','PIL','cv2','torch','torchvision','ultralytics','PySide6','PySide6.QtWidgets','PySide6.QtSvg',
           'psutil','yaml','matplotlib','scipy','pandas','polars','pygrabber','clip','einops','lap']
    modules={}
    for name in names:
        module=importlib.import_module(name);path=Path(module.__file__).resolve()
        assert path.is_relative_to(ROOT),f'External dependency: {name}: {path}'
        modules[name]=str(path.relative_to(ROOT))
    assert Path(sys.prefix).resolve()==ROOT/'runtime'
    app=Path(__file__).resolve().parent
    from trained_models import drone10_config
    from final_tools import readiness,training_registry
    config=drone10_config(app);check=readiness(app,[r['model'] for r in config['cameras']]);assert check['ready']
    queue=mp.get_context('spawn').Queue();process=mp.get_context('spawn').Process(target=spawn_echo,args=(queue,))
    process.start();worker=queue.get(timeout=20);process.join(20)
    if process.is_alive():process.terminate();process.join();raise RuntimeError('Worker did not close')
    assert process.exitcode==0 and Path(worker['prefix']).resolve()==ROOT/'runtime'
    cuda={'available':torch.cuda.is_available()}
    if cuda['available']:
        x=torch.ones((8,8),device='cuda');assert float((x@x)[0,0])==8
        cuda['device']=torch.cuda.get_device_name(0);del x
    manifest=ROOT/'manifest.json';integrity=None
    if '--integrity' in sys.argv:
        data=json.loads(manifest.read_text(encoding='utf-8'))
        for row in data['files']:
            path=(ROOT/row['path']).resolve();assert path.is_relative_to(ROOT)
            assert path.is_file() and path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],row['path']
        integrity={'verified_files':len(data['files']),'passed':True}
    if '--models' in sys.argv:
        from ultralytics import YOLO
        video=cv2.VideoCapture(str(ROOT/'examples/drone.avi'));ok,frame=video.read();video.release();assert ok
        model=YOLO(config['cameras'][1]['model']);result=model.predict(frame,device='cpu',imgsz=320,verbose=False)[0]
        assert model.names=={0:'drone'}
        if cuda['available']:result=model.predict(frame,device='0',imgsz=320,verbose=False)[0];assert result.boxes.data.is_cuda
    registry=training_registry(ROOT);assert registry['runs'] and all(r['best_checkpoint_exists'] for r in registry['runs'])
    report={'passed':True,'python':sys.version.split()[0],'prefix':str(Path(sys.prefix).relative_to(ROOT)),
        'dependencies_internal':True,'modules':modules,'worker':{'passed':True,'bundled_python':True},
        'CUDA':cuda,'deployed_drone_best_epoch':53,'registry_models':len(registry['runs']),'integrity':integrity,
        'scope':'Checks this computer; industrial cameras, another Windows installation and hardware qualification are not certified.'}
    (ROOT/'data').mkdir(exist_ok=True)
    (ROOT/'data/last_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
