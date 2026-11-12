"""Reviewed annotation projects. Inference proposals never become ground truth silently."""
from pathlib import Path
from datetime import datetime,timezone
import collections,csv,hashlib,json,math,re,shutil,uuid,zipfile
import numpy as np
from PIL import Image,ImageDraw
from core import files,IMAGE_EXTS,read,dump,digest,under

STATUSES={'unreviewed','edited','confirmed','rejected'}

def atomic(path,data):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp'); dump(temporary,data); temporary.replace(path)

def project(path):
    data=read(Path(path)/'project.json')
    if data.get('format')!='panoptr_annotation_v1': raise ValueError('Это не проект разметки лаборатории.')
    return data

def create_project(images,output,classes,group_regex='',independent=False):
    root=Path(images).resolve(); output=Path(output).resolve()
    if not root.is_dir() or output.exists() or output.is_relative_to(root): raise ValueError('Нужна папка изображений и новая папка проекта вне исходников.')
    if not classes or len(classes)>253 or len(set(classes))!=len(classes) or any(not str(c).strip() for c in classes): raise ValueError('Задайте 1–253 уникальных класса объектов, без фонового класса.')
    paths=files(root,IMAGE_EXTS)
    if not paths: raise ValueError('Нет изображений.')
    pattern=re.compile(group_regex) if group_regex else None
    rows=[]
    for i,p in enumerate(paths):
        with Image.open(p) as im:
            im.load(); w,h=im.size
            if im.getexif().get(274,1)!=1: raise ValueError('EXIF-поворот требует согласования до разметки: '+str(p))
        rel=p.relative_to(root); sha=digest(p); group=''
        if pattern:
            match=pattern.search(rel.with_suffix('').as_posix())
            if not match or not match.groups(): raise ValueError('Regex сцены не сопоставился: '+str(rel))
            group=match.group(1)
        elif independent: group='image:'+sha
        elif rel.parent!=Path('.'): group='folder:'+rel.parent.as_posix()
        identifier=hashlib.sha256(rel.as_posix().encode()).hexdigest()[:24]
        rows.append({'id':identifier,'image':str(p),'relative_image':rel.as_posix(),'image_sha256':sha,'width':w,'height':h,
                     'group':group,'status':'unreviewed','objects':[],'revision':0,'inference_complete':False})
        print('Импорт',i+1,'/',len(paths),flush=True)
    output.mkdir(parents=True); (output/'frames').mkdir(); (output/'masks').mkdir()
    dump(output/'project.json',{'format':'panoptr_annotation_v1','classes':classes,'image_root':str(root),
         'frame_ids':[r['id'] for r in rows],'created_utc':datetime.now(timezone.utc).isoformat(),
         'annotation_is_ground_truth':False,'review_required':True})
    for r in rows: dump(output/'frames'/f"{r['id']}.json",r)
    return {'project':str(output),'images':len(rows),'classes':classes}

def frame(root,identifier):
    if not re.fullmatch('[a-f0-9]{24}',identifier): raise ValueError('Недопустимый ID кадра.')
    data=read(Path(root)/'frames'/f'{identifier}.json')
    if data['id']!=identifier: raise ValueError('ID кадра не совпадает.')
    return data

def all_frames(root): return [frame(root,i) for i in project(root)['frame_ids']]

def validate_frame(root,data):
    meta=project(root); w,h=data['width'],data['height']
    if data['status'] not in STATUSES: raise ValueError('Неизвестный статус.')
    if digest(data['image'])!=data['image_sha256']: raise ValueError('Исходное изображение изменено.')
    with Image.open(data['image']) as im:
        if im.size!=(w,h): raise ValueError('Размер изображения изменён.')
    seen=set()
    for obj in data['objects']:
        if obj['id'] in seen: raise ValueError('Повтор ID объекта.')
        seen.add(obj['id']); c=obj.get('class_id')
        score=obj.get('confidence')
        if score is not None and (not math.isfinite(score) or not 0<=score<=1): raise ValueError('Достоверность должна быть в диапазоне 0–1.')
        if c is not None and (not isinstance(c,int) or not 0<=c<len(meta['classes'])): raise ValueError('Класс вне диапазона.')
        if data['status']=='confirmed' and c is None: raise ValueError('Нельзя подтвердить объект без выбранного класса.')
        box=obj.get('bbox'); x1,y1,x2,y2=box
        if not all(math.isfinite(v) for v in box) or not 0<=x1<x2<=w or not 0<=y1<y2<=h: raise ValueError('Неверная рамка объекта.')
        if obj.get('mask'):
            with Image.open(under(root,obj['mask'])) as im: arr=np.array(im)
            if arr.shape!=(h,w) or not set(map(int,np.unique(arr)))<={0,255} or not arr.any(): raise ValueError('Маска должна быть непустой бинарной PNG исходного размера.')
            expected=mask_bbox(arr>0)
            if any(abs(a-b)>1e-5 for a,b in zip(expected,box)): raise ValueError('Рамка не соответствует маске.')
    return True

def save_frame(root,data,action='edit'):
    old=frame(root,data['id'])
    if data['revision']!=old['revision']: raise ValueError('Кадр изменён другим процессом. Перезагрузите проект.')
    for name in ['id','image','image_sha256','relative_image','width','height']:
        if data[name]!=old[name]: raise ValueError('Нельзя изменять происхождение кадра.')
    validate_frame(root,data); data['revision']+=1; data['updated_utc']=datetime.now(timezone.utc).isoformat()
    atomic(Path(root)/'frames'/f"{data['id']}.json",data)
    with (Path(root)/'history.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'frame':data['id'],'revision':data['revision'],'action':action,'status':data['status'],
                                'time':data['updated_utc']},ensure_ascii=False)+'\n')
    return data

def mask_bbox(mask):
    ys,xs=np.nonzero(mask)
    if not len(xs): raise ValueError('Пустая маска.')
    return [int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]

def add_object(root,data,class_id,bbox=None,mask=None,confidence=None,source='manual',extra=None):
    identifier=uuid.uuid4().hex
    obj={'id':identifier,'class_id':class_id,'bbox':list(bbox) if bbox is not None else None,'mask':None,
         'confidence':float(confidence) if confidence is not None else None,'source':source,'provenance':extra or {}}
    if mask is not None:
        mask=np.asarray(mask,dtype=bool)
        if mask.shape!=(data['height'],data['width']): raise ValueError('Маска не в исходных координатах.')
        obj['bbox']=mask_bbox(mask); obj['mask']=f"masks/{data['id']}_{identifier}.png"
        Image.fromarray(mask.astype(np.uint8)*255).save(Path(root)/obj['mask'])
    data['objects'].append(obj); data['status']='edited' if source=='manual' else 'unreviewed'
    return obj

def edit_mask(root,data,obj,points,radius,erase=False):
    if not obj.get('mask'): raise ValueError('Кисть требует маски; для bbox сначала создайте маску полигоном или SAM.')
    with Image.open(under(root,obj['mask'])) as im: mask=im.copy()
    draw=ImageDraw.Draw(mask); fill=0 if erase else 255
    for x,y in points: draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=fill)
    arr=np.array(mask)
    if not arr.any(): raise ValueError('Маска стала пустой; удалите объект кнопкой удаления.')
    # A revision stores a new mask, preserving previous model proposals and earlier edits.
    obj['mask']=f"masks/{data['id']}_{uuid.uuid4().hex}.png"; mask.save(Path(root)/obj['mask']); obj['bbox']=mask_bbox(arr>0)
    obj['source']='manual_corrected'; data['status']='edited'

def queue(root):
    rows=all_frames(root)
    def priority(r):
        scores=[o['confidence'] for o in r['objects'] if o.get('confidence') is not None]
        return (r['status'] in ('confirmed','rejected'),bool(r['objects']),min(scores) if scores else -1,r['relative_image'])
    return sorted(rows,key=priority)

def rle_encode(mask):
    values=np.asarray(mask,dtype=np.uint8).reshape(-1,order='F'); counts=[]; current=0; run=0
    for value in values:
        value=int(bool(value))
        if value==current: run+=1
        else: counts.append(run); run=1; current=value
    counts.append(run)
    return {'size':list(mask.shape),'counts':counts}

def rle_decode(rle):
    counts=rle['counts']; out=np.zeros(sum(counts),dtype=bool); pos=0; value=False
    for n in counts:
        out[pos:pos+n]=value; pos+=n; value=not value
    return out.reshape(rle['size'],order='F')

def polygons(mask):
    import cv2
    contours,hierarchy=cv2.findContours(np.asarray(mask,dtype=np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None: return []
    # Standard YOLO polygon format cannot retain holes or disconnected components as one instance.
    if len(contours)!=1 or hierarchy[0][0][2]!=-1: raise ValueError('YOLO polygon потеряет отверстия/части объекта. Используйте COCO RLE или семантические PNG.')
    points=contours[0].reshape(-1,2)
    if len(points)<3: raise ValueError('Маска слишком тонкая для YOLO polygon.')
    return points

def export_project(root,output,kind,overlap='error'):
    root=Path(root).resolve(); output=Path(output).resolve(); meta=project(root)
    if output.exists() or Path(str(output)+'.zip').exists() or output.is_relative_to(Path(meta['image_root'])):
        raise ValueError('Выберите новую папку вне исходных изображений; ZIP тоже не должен существовать.')
    if kind not in ('yolo_bbox','yolo_segment','coco','semantic'): raise ValueError('Неизвестный экспорт.')
    if kind=='semantic' and 'background' in meta['classes']: raise ValueError('background зарезервирован для фона семантической маски; переименуйте класс объекта.')
    if overlap not in ('error','last','ignore'): raise ValueError('Неизвестная политика перекрытий.')
    rows=[r for r in all_frames(root) if r['status']=='confirmed']
    if not rows: raise ValueError('Нет подтверждённых кадров. Пустые кадры тоже надо проверить вручную.')
    rendered=[]; coco={'images':[],'annotations':[],'categories':[{'id':i,'name':n} for i,n in enumerate(meta['classes'])]}
    for number,r in enumerate(rows):
        validate_frame(root,r); labels=[]; semantic=np.zeros((r['height'],r['width']),dtype=np.uint8); owned=np.zeros(semantic.shape,dtype=bool)
        for obj in r['objects']:
            c=obj['class_id']; x1,y1,x2,y2=obj['bbox']; mask=None
            if obj.get('mask'):
                with Image.open(under(root,obj['mask'])) as im: mask=np.array(im)>0
            if kind in ('semantic','yolo_segment') and mask is None: raise ValueError('Маска отсутствует: bbox не превращается в сегментацию автоматически.')
            if kind=='semantic':
                conflict=owned & mask
                if conflict.any() and overlap=='error': raise ValueError('Объекты перекрываются. Явно выберите last или ignore, либо исправьте маски.')
                semantic[mask]=c+1; owned|=mask
                if overlap=='ignore': semantic[conflict]=255
            elif kind=='yolo_segment':
                points=polygons(mask); labels.append(str(c)+' '+' '.join(f'{v:.10f}' for v in (points/np.array([r['width'],r['height']])).reshape(-1)))
            elif kind=='yolo_bbox':
                labels.append(f"{c} {(x1+x2)/2/r['width']:.10f} {(y1+y2)/2/r['height']:.10f} {(x2-x1)/r['width']:.10f} {(y2-y1)/r['height']:.10f}")
            else:
                ann={'id':len(coco['annotations'])+1,'image_id':number,'category_id':c,'bbox':[x1,y1,x2-x1,y2-y1],
                     'area':int(mask.sum()) if mask is not None else (x2-x1)*(y2-y1),'iscrowd':0}
                if mask is not None: ann['segmentation']=rle_encode(mask)
                coco['annotations'].append(ann)
        coco['images'].append({'id':number,'file_name':r['relative_image'],'width':r['width'],'height':r['height']})
        coco['images'][-1]['group']=r['group']
        rendered.append((r,labels,semantic))
    # All export feasibility checks completed before creating the output directory.
    output.mkdir(parents=True); pairs=[]
    for r,labels,semantic in rendered:
        rel=Path(r['relative_image']); target=under(output/'images',rel); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(r['image'],target)
        # Append extension, so a.jpg and a.png remain distinct annotations.
        lab_rel=rel.parent/(rel.name+('.png' if kind=='semantic' else '.txt'))
        if kind!='coco':
            lab=under(output/('masks' if kind=='semantic' else 'labels'),lab_rel); lab.parent.mkdir(parents=True,exist_ok=True)
            if kind=='semantic': Image.fromarray(semantic).save(lab)
            else: lab.write_text('\n'.join(labels)+'\n' if labels else '',encoding='utf-8')
        pairs.append({'image':rel.as_posix(),'annotation':lab_rel.as_posix(),'group':r['group']})
    if kind=='coco': dump(output/'annotations.json',coco)
    with (output/'pairs.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['image','annotation','group']); writer.writeheader(); writer.writerows(pairs)
    names=['background',*meta['classes']] if kind=='semantic' else meta['classes']; dump(output/'classes.json',names)
    config={'format':'semantic_masks' if kind=='semantic' else 'coco_bbox' if kind=='coco' else kind,
            'images':str(output/'images'),'annotations':str(output/'masks' if kind=='semantic' else output/'annotations.json' if kind=='coco' else output/'labels'),
            'classes':names,'pairs_csv':str(output/'pairs.csv') if kind!='coco' else '',
            'mask_suffix':'','mask_mapping':{str(i):i for i in range(len(names))} if kind=='semantic' else {},
            'independent_images':False,'group_regex':'','val_fraction':.2,'seed':20261012}
    if kind=='semantic': config['mask_mapping']['255']=-1
    dump(output/'import_config.json',config); dump(output/'provenance.json',{'source_project':str(root),'export_kind':kind,'confirmed_frames':len(rows),
         'excluded_unreviewed_edited_rejected':len(meta['frame_ids'])-len(rows),'overlap_policy':overlap,
         'source_frames':[r for r,_,_ in rendered],'training_started':False,
         'warning':'Confirmed means human-reviewed, not independently proven ground truth. Keep val/test manually checked and separate.'})
    with zipfile.ZipFile(str(output)+'.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(output).as_posix())
    return {'output':str(output),'images':len(rows),'kind':kind,'archive':str(output)+'.zip',
            'lab_import_supported':True}
