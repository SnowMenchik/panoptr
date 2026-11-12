"""Two YOLO checkpoints on exactly the same copied local split, CPU only."""
from pathlib import Path
from unittest.mock import patch
import hashlib,json,math,shutil
import numpy as np
import yaml
from PIL import Image
from final_core import save_json,sha256
from final_tools import IMAGE_EXTS

def prepare_validation(yaml_path,split,output,task,names,limit=0):
    yaml_path=Path(yaml_path).resolve(); cfg=yaml.safe_load(yaml_path.read_text(encoding='utf-8-sig'))
    if not isinstance(cfg,dict) or split not in ('val','test') or split not in cfg: raise ValueError('В YAML нет выбранного val/test.')
    original_names=cfg.get('names'); original_names=dict(enumerate(original_names)) if isinstance(original_names,list) else {int(k):str(v) for k,v in original_names.items()} if isinstance(original_names,dict) else {}
    if original_names!=names or len(set(names.values()))!=len(names): raise ValueError('ID и названия классов YAML должны точно совпадать с обеими моделями.')
    root=Path(cfg.get('path') or yaml_path.parent).expanduser(); root=root if root.is_absolute() else yaml_path.parent/root
    values=cfg[split] if isinstance(cfg[split],list) else [cfg[split]]; paths=[]
    for value in values:
        if not isinstance(value,str) or '://' in value: raise ValueError('Разрешены только локальные источники, без загрузки датасетов.')
        source=Path(value); source=source if source.is_absolute() else root/source
        if source.is_dir(): paths.extend(p for p in source.rglob('*') if p.suffix.lower() in IMAGE_EXTS)
        elif source.is_file() and source.suffix.lower()=='.txt':
            for line in source.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip(): continue
                if '://' in line: raise ValueError('Список изображений должен содержать локальные пути.')
                p=Path(line.strip()); paths.append(p if p.is_absolute() else source.parent/p)
        else: raise ValueError('Источник выборки не найден: '+str(source))
    paths=sorted({p.resolve() for p in paths}); total=len(paths)
    if limit<0 or not paths: raise ValueError('Нет изображений или неверный предел.')
    if limit: paths=paths[:limit]
    rows=[]; gt_counts=[0]*len(names)
    for path in paths:
        if not path.is_file(): raise ValueError('Изображение не найдено: '+str(path))
        with Image.open(path) as im:
            if im.getexif().get(274,1)!=1: raise ValueError('Сначала согласуйте EXIF-поворот и разметку: '+str(path))
        with Image.open(path) as im: im.verify()
        parts=list(path.parts); indices=[i for i,v in enumerate(parts) if v=='images']
        if not indices: raise ValueError('Нужен YOLO layout images/... и labels/... . Используйте подготовку в лаборатории.')
        parts[indices[-1]]='labels'; label=Path(*parts).with_suffix('.txt')
        if not label.is_file(): raise ValueError('Нет явного GT-файла: '+str(label)+'. Для проверенного фона нужен пустой TXT.')
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip(): continue
            fields=line.split(); c=int(fields[0]); coords=list(map(float,fields[1:]))
            if c not in names or not all(math.isfinite(v) and 0<=v<=1 for v in coords): raise ValueError('Некорректная разметка: '+str(label))
            if task=='detect':
                if len(coords)!=4 or min(coords[2:])<=0: raise ValueError('Для detect нужны нормализованные bbox.')
                x,y,w,h=coords
                if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6: raise ValueError('BBox вне кадра.')
            else:
                if len(coords)<6 or len(coords)%2: raise ValueError('Для segment нужны полигоны, не bbox.')
                points=np.array(coords).reshape(-1,2)
                area=abs(np.dot(points[:,0],np.roll(points[:,1],1))-np.dot(points[:,1],np.roll(points[:,0],1)))/2
                if area<=1e-12: raise ValueError('Вырожденный полигон.')
            gt_counts[c]+=1
        rows.append({'source_image':str(path),'source_label':str(label),'image_sha256':sha256(path),'label_sha256':sha256(label)})
    if not sum(gt_counts): raise ValueError('Нет ни одного размеченного объекта; AP нельзя оценить на одном фоне.')
    output=Path(output)
    if output.exists(): raise ValueError('Папка проверки уже существует.')
    output.mkdir(parents=True); images=output/'validation_input/images/val'; labels=output/'validation_input/labels/val'; images.mkdir(parents=True); labels.mkdir(parents=True)
    for i,row in enumerate(rows):
        name=f'{i:08d}'; target=images/(name+Path(row['source_image']).suffix.lower()); annotation=labels/(name+'.txt')
        shutil.copy2(row['source_image'],target); shutil.copy2(row['source_label'],annotation); row.update(copied_image=str(target),copied_label=str(annotation))
    data={'path':str(output/'validation_input'),'train':'images/val','val':'images/val','test':'images/val','names':names,
        'purpose':'Validation-only copy. The train alias satisfies the validator schema and must not be used to train a model.'}
    local_yaml=output/'validation_only.yaml'; local_yaml.write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False),encoding='utf-8')
    fingerprint=hashlib.sha256(json.dumps({'task':task,'names':names,'files':[(r['image_sha256'],r['label_sha256']) for r in rows]},sort_keys=True).encode()).hexdigest()
    metadata={'source_yaml':str(yaml_path),'source_yaml_sha256':sha256(yaml_path),'selected_split':split,'total_source_images':total,
        'evaluated_images':len(rows),'complete_split':len(rows)==total,'gt_instances_per_class':gt_counts,'dataset_signature':fingerprint,'source_manifest':rows,
        'download_directive_ignored':True,'independent_test_verified':False,'warning':'The word test does not prove independence from training or model selection.'}
    save_json(output/'input_manifest.json',metadata); return local_yaml,metadata

def compare(cfg,output):
    import torch
    torch.set_num_threads(2)
    from ultralytics import YOLO
    output=Path(output).resolve()
    if output.exists(): raise ValueError('Выберите новую папку сравнения.')
    paths=[Path(cfg['first']).resolve(),Path(cfg['second']).resolve()]
    if not all(p.is_file() for p in paths): raise ValueError('Нужны два локальных checkpoint.')
    models=[YOLO(str(p)) for p in paths]; task=models[0].task; names=models[0].names
    if task not in ('detect','segment') or models[1].task!=task or models[1].names!=names: raise ValueError('Обе модели должны иметь одинаковые классы и задачу detect/segment.')
    size=int(cfg.get('image_size',640))
    if not 32<=size<=1280: raise ValueError('Размер проверки должен быть 32–1280.')
    local_yaml,metadata=prepare_validation(cfg['data_yaml'],cfg.get('split','val'),output,task,names,int(cfg.get('limit',0)))
    save_json(output/'status.json',{'ready':False}); rows=[]
    for number,(path,model) in enumerate(zip(paths,models)):
        before=sha256(path)
        # No font downloads are needed with plots=False. The dataset's download program is never passed to Ultralytics.
        with patch('ultralytics.data.utils.check_font',return_value=None):
            scores=model.val(data=str(local_yaml),split='val',device='cpu',imgsz=size,batch=1,workers=0,plots=False,save_json=False,
                project=str(output/'validation_runs'),name=f'model_{number+1}',conf=.001,iou=.7)
        classes=[]
        for c,name in names.items():
            entry={'class_id':int(c),'class_name':name,'gt_instances':metadata['gt_instances_per_class'][c],'box':None,'mask':None}
            for field,metric in [('box',scores.box),('mask',getattr(scores,'seg',None))]:
                if metric is not None and entry['gt_instances']>0 and c in metric.ap_class_index:
                    values=metric.class_result(list(metric.ap_class_index).index(c)); entry[field]={key:float(v) for key,v in zip(['precision','recall','ap50','ap50_95'],values)}
            classes.append(entry)
        if sha256(path)!=before: raise ValueError('Checkpoint изменился во время проверки. Выберите завершённые веса.')
        for r in metadata['source_manifest']:
            if sha256(r['copied_image'])!=r['image_sha256'] or sha256(r['copied_label'])!=r['label_sha256']: raise ValueError('Валидатор изменил входные файлы; результаты не считаем сопоставимыми.')
        rows.append({'checkpoint':str(path),'sha256':before,'metrics':{k:float(v) for k,v in scores.results_dict.items()},'per_class':classes,
            'validator_mean_stage_ms':{k:float(v) for k,v in scores.speed.items()}})
        print('Проверена модель',number+1,'/ 2',flush=True)
    report={'ready':True,'task':task,'device':'cpu','image_size':size,'input':metadata,'models':rows,'training_started':False,'source_data_modified':False,
        'definitions':{'precision_recall':'Ultralytics P/R at its selected F1 operating point, not a common production confidence threshold.',
            'ap':'Same copied ground truth, confidence cutoff .001, NMS IoU .7, same image size for both models.',
            'speed':'Validator mean stage timings, not P95 camera-to-card latency. Concurrent training can affect them.'}}
    save_json(output/'comparison.json',report); save_json(output/'status.json',{'ready':True})
    import csv
    with (output/'per_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['model','class_id','class_name','gt_instances','kind','precision','recall','ap50','ap50_95']; writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for i,m in enumerate(rows):
            for c in m['per_class']:
                for kind in ('box','mask'):
                    if kind=='mask' and task!='segment': continue
                    writer.writerow({'model':i+1,**{k:c[k] for k in ('class_id','class_name','gt_instances')},'kind':kind,**(c[kind] or {})})
    return report
