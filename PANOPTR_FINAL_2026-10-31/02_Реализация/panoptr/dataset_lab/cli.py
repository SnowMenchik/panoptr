"""CLI workers used by the Qt interface and generated BAT launchers."""
from bootstrap import setup
setup()
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
from core import read,dump,audit,prepare,verify_prepared

@contextlib.contextmanager
def training_lock(device,lock_path=None):
    # Same physical GPU and same lock as the existing 08/09 training launchers.
    if device=='cpu':
        yield
        return
    path=Path(lock_path) if lock_path else Path(__file__).resolve().parents[3]/'08_Обучение/runs/.training.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as handle:
        if path.stat().st_size==0: handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            try: msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError: raise ValueError('GPU занят другим обучением. Дождитесь завершения предыдущего запуска.')
        else:
            import fcntl
            try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError: raise ValueError('Другое обучение уже запущено.')
        try: yield
        finally:
            handle.seek(0)
            if os.name=='nt': msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(handle,fcntl.LOCK_UN)

def check_training(root,settings):
    report=verify_prepared(root); meta=read(Path(root)/'dataset.json')
    if settings.get('task')!=meta['task']: raise ValueError('Задача настроек не совпадает с датасетом.')
    if settings['device'] not in ('cpu','cuda:0','cuda:1'): raise ValueError('Неизвестное устройство.')
    if not 1<=int(settings['epochs'])<=10000 or not 1<=int(settings['batch'])<=128 or not 0<float(settings['lr'])<=.1:
        raise ValueError('Недопустимые epochs/batch/lr.')
    if meta['task'] in ('detect','segment'):
        checkpoint=settings.get('checkpoint','')
        if not checkpoint or not Path(checkpoint).is_file(): raise ValueError('Для детекции выберите локальные исходные веса YOLO; автоскачивание отключено.')
        from ultralytics import YOLO
        model=YOLO(checkpoint)
        if model.task!=meta['task']: raise ValueError('Нужен YOLO нужной задачи (detect/segment); семантика имеет отдельный backend.')
    elif settings.get('checkpoint'):
        from semantic import load_checkpoint
        _,ck=load_checkpoint(settings['checkpoint'])
        if ck['classes']!=meta['classes']: raise ValueError('Классы семантических весов отличаются.')
    missing={split:[meta['classes'][i] for i in range(len(meta['classes'])) if not meta['class_image_coverage'][split][str(i)]] for split in ('train','val')}
    report.update(training_started=False,missing_classes=missing,device_requested=settings['device'])
    if missing['train'] or missing['val']: raise ValueError('Классы отсутствуют в train/val. Измените группы/seed или состав данных: '+str(missing))
    return report

def train_detection(root,settings,run,pilot=False,resume=False):
    from ultralytics import YOLO
    from core import IMAGE_EXTS
    import shutil
    root=Path(root).resolve(); run=Path(run).resolve()
    source=run/'weights/last.pt' if resume else Path(settings['checkpoint'])
    if resume and not source.is_file(): raise ValueError('Нет last.pt для продолжения.')
    if not resume and run.exists(): raise ValueError('Запуск уже существует: используйте RESUME или новую папку.')
    model=YOLO(str(source)); meta=read(root/'dataset.json')
    if resume:
        info=read(run/'lab_run.json')
        if info['dataset_signature']!=signature(root) or info['settings']!=settings: raise ValueError('Настройки/датасет изменены с прошлого запуска.')
    else:
        model.add_callback('on_train_start',lambda trainer:dump(Path(trainer.save_dir)/'lab_run.json',
            {'dataset_signature':signature(root),'settings':settings,'pilot':pilot}))
    if pilot:
        # Separate genuine small dataset rather than fraction=... on arbitrary source ordering.
        data_root=run.parent/(run.name+'_data')
        if data_root.exists(): raise ValueError('Папка пробного набора уже существует.')
        data_root.mkdir(parents=True)
        rows=read(root/'manifest.json')
        for split in ('train','val'):
            for r in [r for r in rows if r['split']==split][:16]:
                for field,kind in [('prepared_image','images'),('prepared_annotation','labels')]:
                    src=root/r[field]; dst=data_root/kind/split/src.name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
        import yaml
        (data_root/'data.yaml').write_text(yaml.safe_dump({'path':str(data_root),'train':'images/train','val':'images/val','nc':len(meta['classes']),'names':dict(enumerate(meta['classes']))},allow_unicode=True),encoding='utf-8')
    else: data_root=root
    if resume: model.train(resume=True,device=settings['device'].replace('cuda:',''),workers=0)
    else:
        model.train(data=str(data_root/'data.yaml'),epochs=1 if pilot else int(settings['epochs']),imgsz=int(settings['image_size']),
            batch=int(settings['batch']),lr0=float(settings['lr']),optimizer='AdamW',device=settings['device'].replace('cuda:',''),
            workers=0,seed=int(settings['seed']),project=str(run.parent),name=run.name,exist_ok=False,amp=False,plots=bool(settings.get('plots',True)))
    # For crash-safe resume this file is also written by callback below (before the first epoch).
    dump(run/'lab_run.json',{'dataset_signature':signature(root),'settings':settings,'pilot':pilot})
    best=YOLO(str(model.trainer.best))
    if best.names!=dict(enumerate(meta['classes'])) or best.model.model[-1].nc!=len(meta['classes']):
        raise ValueError('Итоговые классы YOLO не совпадают с подготовленным набором.')
    dump(run/'completed.json',{'best_checkpoint':str(Path(model.trainer.best).resolve()),'pilot':pilot,'application_weights_replaced':False})
    return read(run/'completed.json')

def signature(root):
    import hashlib
    root=Path(root)
    return hashlib.sha256((root/'manifest.json').read_bytes()+(root/'dataset.json').read_bytes()).hexdigest()

def compare_detection(root,first,second,device,output):
    import tempfile
    from ultralytics import YOLO
    verify_prepared(root); meta=read(Path(root)/'dataset.json')
    if meta['task'] not in ('detect','segment'): raise ValueError('Нужен YOLO-датасет detect/segment.')
    if output and Path(output).exists(): raise ValueError('Отчёт уже существует.')
    results=[]
    temp_root=(Path(output).resolve().parent if output else Path(root).resolve()/'evaluation_temp')
    temp_root.mkdir(parents=True,exist_ok=True)
    for checkpoint in (first,second):
        model=YOLO(checkpoint)
        if model.task!=meta['task'] or model.names!=dict(enumerate(meta['classes'])): raise ValueError('YOLO-классы и задача должны совпадать с датасетом.')
        with tempfile.TemporaryDirectory(prefix='panoptr_eval_',dir=temp_root) as folder:
            scores=model.val(data=str(Path(root)/'data.yaml'),split='val',device=device.replace('cuda:',''),imgsz=960,
                batch=1,workers=0,project=folder,name='val',plots=False)
            results.append({'checkpoint':str(checkpoint),'image_size':960,'metrics':{k:float(v) for k,v in scores.results_dict.items()}})
    result={'task':meta['task'],'split':'val','independent_test':False,'models':results}
    if output: dump(output,result)
    return result

def predict_detection(checkpoint,images,output,device):
    from ultralytics import YOLO
    from core import files,IMAGE_EXTS
    import zipfile
    root=Path(images).resolve(); output=Path(output).resolve(); archive=Path(str(output)+'.zip')
    if output.exists() or archive.exists() or output.is_relative_to(root): raise ValueError('Нужна новая папка вне исходных изображений; ZIP также не должен существовать.')
    paths=files(root,IMAGE_EXTS)
    if not paths: raise ValueError('Нет изображений.')
    model=YOLO(checkpoint)
    if model.task not in ('detect','segment'): raise ValueError('Нужен YOLO detect/segment.')
    output.mkdir(parents=True); dump(output/'status.json',{'ready':False}); rows=[]
    for i,path in enumerate(paths):
        result=model.predict(source=str(path),device=device.replace('cuda:',''),imgsz=960,verbose=False)[0]
        boxes=result.boxes
        masks=[]
        if model.task=='segment':
            from annotation_models import original_masks
            from annotation import rle_encode
            masks=[rle_encode(mask) for mask in original_masks(result)]
        rows.append({'image':path.relative_to(root).as_posix(),'height':int(result.orig_shape[0]),'width':int(result.orig_shape[1]),
                     'detections':[{'class_id':int(c),'confidence':float(s),'xyxy':list(map(float,b))}
                        for c,s,b in zip(boxes.cls.cpu().tolist(),boxes.conf.cpu().tolist(),boxes.xyxy.cpu().tolist())]})
        if model.task=='segment':
            for j,obj in enumerate(rows[-1]['detections']): obj['segmentation']=masks[j] if j<len(masks) else None
        print('Экспорт',i+1,'/',len(paths),flush=True)
    dump(output/'predictions.json',rows); dump(output/'classes.json',model.names); dump(output/'status.json',{'ready':True,'images':len(rows)})
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in output.iterdir(): z.write(p,p.name)
    return {'images':len(rows),'output':str(output),'archive':str(archive)}

def extract_frames(video,output,every=30,limit=300):
    import cv2
    from core import digest
    from PIL import Image
    cv2.setNumThreads(1); output=Path(output).resolve(); video=Path(video).resolve()
    if output.exists() or not video.is_file(): raise ValueError('Нужно существующее видео и новая выходная папка.')
    if every<1 or limit<1: raise ValueError('Шаг и предел кадров должны быть положительными.')
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise ValueError('Не удалось открыть видео.')
    group='video_'+digest(video)[:16]; fps=float(cap.get(cv2.CAP_PROP_FPS)); rows=[]; index=0
    try:
        folder=output/'images'/group; folder.mkdir(parents=True)
        while len(rows)<limit:
            ok,frame=cap.read()
            if not ok: break
            if index%every==0:
                target=folder/f'frame_{index:08d}.png'; Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(target)
                rows.append({'image':target.relative_to(output/'images').as_posix(),'group':group,'frame_index':index,
                             'seconds':index/fps if fps>0 else None})
                print('Кадр',index,flush=True)
            index+=1
    finally: cap.release()
    if not rows: raise ValueError('Видео не содержит доступных кадров.')
    dump(output/'frames.json',{'source_video':str(video),'fps':fps,'frames':rows,'masks_provided':False,
        'warning':'Все кадры одного видео — одна группа. Для независимой val нужны другие сцены/видео или обоснованная ручная политика.'})
    return {'output':str(output),'frames':len(rows),'group':group,'labels_created':False}

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['audit','prepare','check','pilot','train','resume','compare','predict','demo','frames'])
    parser.add_argument('--config'); parser.add_argument('--dataset'); parser.add_argument('--output'); parser.add_argument('--checkpoint')
    parser.add_argument('--second'); parser.add_argument('--images'); parser.add_argument('--device',default='cpu'); parser.add_argument('--settings')
    parser.add_argument('--video'); parser.add_argument('--every',type=int,default=30); parser.add_argument('--limit',type=int,default=300)
    args=parser.parse_args(argv)
    if args.command=='frames': result=extract_frames(args.video,args.output,args.every,args.limit)
    elif args.command=='demo':
        from demo import create_demo
        result=create_demo(args.output)
    elif args.command in ('audit','prepare'):
        cfg=read(args.config)
        if args.command=='audit': result=audit(cfg); dump(args.output,result)
        else: result=prepare(cfg,args.output)
    elif args.command in ('compare','predict'):
        with training_lock(args.device):
            if args.command=='compare':
                task=read(Path(args.dataset)/'dataset.json')['task']
                if task in ('detect','segment'): result=compare_detection(args.dataset,args.checkpoint,args.second,args.device,args.output)
                else:
                    from semantic import compare
                    result=compare(args.dataset,args.checkpoint,args.second,args.device,args.output)
            else:
                # Prefer format-tagged weights-only semantic files. YOLO loading is only the fallback.
                from semantic import load_checkpoint,export_predictions
                try: load_checkpoint(args.checkpoint)
                except Exception: result=predict_detection(args.checkpoint,args.images,args.output,args.device)
                else: result=export_predictions(args.checkpoint,args.images,args.output,args.device)
    else:
        if not args.dataset: raise ValueError('Не указан датасет.')
        root=Path(args.dataset).resolve(); settings=read(args.settings or root/'training.json')
        result=check_training(root,settings)
        if args.command!='check':
            run=root/'runs'/('pilot_'+__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S') if args.command=='pilot' else 'train')
            if args.command!='resume' and run.exists(): raise ValueError('Запуск уже существует. Используйте продолжение.')
            run.parent.mkdir(parents=True,exist_ok=True)
            with training_lock(settings['device']):
                if settings['task']=='semantic':
                    from semantic import train
                    result=train(root,settings,run,pilot=args.command=='pilot',resume=args.command=='resume')
                else: result=train_detection(root,settings,run,pilot=args.command=='pilot',resume=args.command=='resume')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 0

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as exc:
        print('ОШИБКА: '+str(exc),file=sys.stderr,flush=True); sys.exit(1)
