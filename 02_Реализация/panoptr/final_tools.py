"""Read-only readiness/registry/data review and CPU rehearsal commands."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,math,time,importlib.util,platform
import cv2
import numpy as np
from final_core import save_json,sha256,frame_health,validate_policy,EventAssociator,iou

IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}

def readiness(app_root,model_paths=None):
    app_root=Path(app_root); requirements=['cv2','numpy','PySide6','torch','ultralytics','psutil']; dependencies={name:importlib.util.find_spec(name) is not None for name in requirements}
    weights=[]
    for value in model_paths or [str(p) for p in (app_root/'models').glob('*.pt')]:
        path=Path(value); path=path if path.is_absolute() else app_root/path
        weights.append({'path':str(path),'exists':path.is_file(),'sha256':sha256(path) if path.is_file() else None})
    return {'ready':all(dependencies.values()) and bool(weights) and all(r['exists'] for r in weights),'dependencies':dependencies,'weights':weights,
        'python':platform.python_version(),'scope':'Local modules and files; cameras, codecs, CUDA and another computer are not validated by this check.',
        'network_required_for_check':False,'gpu_used':False}

def training_registry(final_root):
    root=Path(final_root); rows=[]
    portable=root/'assets/model_registry.json'
    if (root/'portable.json').is_file() and portable.is_file():
        data=json.loads(portable.read_text(encoding='utf-8'))
        for row in data['runs']:
            path=root/row['checkpoint'];row['checkpoint']=str(path);row['best_checkpoint_exists']=path.is_file()
        return data
    for folder in ['08_Обучение','09_Дополнительные_данные_2026-10-11']:
        for path in (root/folder/'runs').rglob('results.csv'):
            try:
                with path.open(encoding='utf-8-sig',newline='') as f: records=[{k.strip():v.strip() for k,v in row.items() if k is not None} for row in csv.DictReader(f)]
                if not records: continue
                latest=records[-1]; epochs=[]
                for row in records:
                    clean={}
                    for key,value in row.items():
                        try:
                            n=float(value); clean[key]=n if math.isfinite(n) else None
                        except (ValueError,TypeError): clean[key]=value
                    epochs.append(clean)
                weights=path.parent/'weights/best.pt'
                rows.append({'run':str(path.parent),'epochs_recorded':len(records),'last_epoch':latest.get('epoch'),
                    'metrics':epochs[-1],'metrics_origin':'Last CSV row, not necessarily the epoch of best.pt','best_checkpoint_exists':weights.is_file(),'checkpoint':str(weights) if weights.is_file() else None,
                    'state':'snapshot; may still be training','per_class_metrics':None,'all_epochs':epochs})
            except Exception as exc: rows.append({'run':str(path.parent),'error':str(exc)})
    return {'runs':rows,'ranking_performed':False,'warning':'Metrics from different datasets/splits/settings cannot determine a common winner. Per-class metrics require explicit validation reports.','training_started':False}

def duplicate_review(train_root,val_root,limit=500,threshold=6):
    from PIL import Image
    roots=[Path(train_root).resolve(),Path(val_root).resolve()]
    if not all(r.is_dir() for r in roots) or roots[0]==roots[1]: raise ValueError('Нужны две разные папки изображений.')
    if not 1<=int(limit)<=5000 or not 0<=int(threshold)<=64: raise ValueError('Предел 1–5000; расстояние 0–64.')
    groups=[]; errors=[]; counts=[]
    for root in roots:
        paths=sorted(p for p in root.rglob('*') if p.suffix.lower() in IMAGE_EXTS); counts.append(len(paths)); rows=[]
        for p in paths[:limit]:
            try:
                with Image.open(p) as im:
                    gray=np.array(im.convert('L').resize((32,32)),np.float32)
                    coeff=cv2.dct(gray)[:8,:8].reshape(-1); bits=coeff>np.median(coeff[1:]); bits[0]=False
                rows.append({'image':str(p),'sha256':sha256(p),'phash':bits})
            except Exception as exc: errors.append({'image':str(p),'error':str(exc)})
        groups.append(rows)
    candidates=[]
    for left in groups[0]:
        for right in groups[1]:
            exact=left['sha256']==right['sha256']; distance=int((left['phash']!=right['phash']).sum())
            if exact or distance<=threshold: candidates.append({'train':left['image'],'val':right['image'],'exact_duplicate':exact,'phash_distance':distance,'action':'review source scene before splitting; no automatic deletion'})
    return {'candidates':sorted(candidates,key=lambda r:(not r['exact_duplicate'],r['phash_distance'])),'errors':errors,
        'total_images':counts,'checked_images':[len(g) for g in groups],'complete_scan':all(n<=limit for n in counts),
        'limitations':'pHash can miss crops/rotations and can match unrelated flat images. Scene grouping and human review remain necessary.','sources_modified':False}

def review_predictions(predictions,ground_truth=None,second=None):
    first=json.loads(Path(predictions).read_text(encoding='utf-8')); first=first.get('frames',first) if isinstance(first,dict) else first
    if not isinstance(first,list): raise ValueError('Нужен список кадров с image и detections.')
    def index(path):
        if not path: return None
        rows=json.loads(Path(path).read_text(encoding='utf-8')); rows=rows.get('frames',rows) if isinstance(rows,dict) else rows
        result={r['image']:r.get('detections',r.get('objects',[])) for r in rows}
        if len(result)!=len(rows): raise ValueError('Повторяющиеся имена кадров.')
        return result
    truth=index(ground_truth); other=index(second); queue=[]
    for r in first:
        reasons=[]; objects=r.get('detections',[]); width=r.get('width'); height=r.get('height')
        for o in objects:
            if o.get('confidence',1)<.35: reasons.append('low_confidence')
            box=o['xyxy']
            if width and height and (box[2]-box[0])*(box[3]-box[1])/(width*height)<.002: reasons.append('small_object')
        def compare(reference):
            used=set(); missing=0
            for target in reference:
                candidates=[(iou(o['xyxy'],target['xyxy']),j) for j,o in enumerate(objects) if j not in used and o['class_id']==target['class_id']]
                score,j=max(candidates,default=(0,-1))
                if score>=.5: used.add(j)
                else: missing+=1
            return missing,len(objects)-len(used)
        if truth is not None:
            if r['image'] not in truth: reasons.append('ground_truth_missing')
            else:
                missed,extra=compare(truth[r['image']]); reasons+=['missed_gt_object']*missed+['unmatched_prediction']*extra
        if other is not None:
            if r['image'] not in other: reasons.append('second_prediction_missing')
            elif any(compare(other[r['image']])): reasons.append('model_disagreement')
        if not objects: reasons.append('empty_prediction_check_background')
        if reasons: queue.append({'image':r['image'],'reasons':sorted(set(reasons)),'priority':len(reasons),'detections':objects})
    return {'queue':sorted(queue,key=lambda r:-r['priority']),'ground_truth_provided':truth is not None,'second_model_provided':other is not None,
        'thresholds':{'low_confidence':.35,'small_area_fraction':.002,'matching_iou':.5},'training_started':False,
        'warning':'Heuristics rank review candidates, not calibrated uncertainty. Class IDs and image coordinates must agree across files. Missing GT is never a negative image.'}

def rehearse(cfg,output):
    output=Path(output).resolve()
    if output.exists() or Path(str(output)+'.zip').exists(): raise ValueError('Для репетиции нужна новая папка; ZIP тоже не должен существовать.')
    video=Path(cfg['video']); model_path=Path(cfg['model'])
    if not video.is_file() or not model_path.is_file(): raise ValueError('Выберите локальное видео и локальные веса.')
    policy=validate_policy(cfg.get('policy',{})); cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise ValueError('Не удалось открыть видео.')
    frames=[]; limit=int(cfg.get('limit',12)); every=int(cfg.get('every',1))
    if not 1<=limit<=300 or every<1: cap.release(); raise ValueError('Предел 1–300, шаг >=1.')
    fps=float(cap.get(cv2.CAP_PROP_FPS)) or 25.; n=0
    try:
        while len(frames)<limit:
            ok,frame=cap.read()
            if not ok: break
            if n%every==0: frames.append((n,frame))
            n+=1
    finally: cap.release()
    if not frames: raise ValueError('Видео не содержит кадров.')
    output.mkdir(parents=True); save_json(output/'status.json',{'ready':False}); print('Загрузка CPU-модели',flush=True)
    import torch
    torch.set_num_threads(2)
    from ultralytics import YOLO
    model=YOLO(str(model_path)); size=int(cfg.get('image_size',320)); conf=float(cfg.get('confidence',.25))
    model.predict(frames[0][1],device='cpu',imgsz=size,conf=conf,verbose=False) # explicit warmup
    scenarios=['baseline','blur','darkness','camera_gap','model_failure','recovery']; report=[]
    from main import InferenceScheduler
    for scenario in scenarios:
        rows=[]; tracker=EventAssociator(); latencies=[]; folder=output/scenario; folder.mkdir(); preview_saved=False
        for index,(number,original) in enumerate(frames):
            gap=scenario in ('camera_gap','recovery') and len(frames)//3<=index<2*len(frames)//3
            failure=scenario=='model_failure' and index==len(frames)//2
            if gap:
                rows.append({'source_frame':number,'source_time_s':number/fps,'assessed':False,'reason':'injected_stream_gap','detections':[]}); continue
            image=cv2.GaussianBlur(original,(21,21),8) if scenario=='blur' else np.zeros_like(original) if scenario=='darkness' else original
            started=time.monotonic()
            try:
                if failure: raise RuntimeError('Injected predict failure')
                result=model.predict(image,device='cpu',imgsz=size,conf=conf,verbose=False)[0]
            except Exception as exc:
                rows.append({'source_frame':number,'source_time_s':number/fps,'assessed':False,'reason':'injected_predict_error' if failure else 'actual_predict_error','error':str(exc),'detections':[]}); continue
            duration=(time.monotonic()-started)*1000; latencies.append(duration)
            packet=InferenceScheduler._make_packet(result,image.shape); health=frame_health(image,started,time.monotonic(),min(fps/every,1000/max(duration,.001)),policy)
            if not preview_saved:
                from main import CameraPanel
                from PIL import Image
                visual=CameraPanel._draw_detections(image,packet); Image.fromarray(cv2.cvtColor(visual,cv2.COLOR_BGR2RGB)).save(folder/'preview.png'); preview_saved=True
            events=tracker.update(0,0,number,number/fps,packet.boxes,packet.names,image.shape,policy,health,packet.masks)
            rows.append({'source_frame':number,'source_time_s':number/fps,'inference_ms':duration,'health':health,
                'detections':[{'xyxy':list(map(float,b[:4])),'class_id':int(b[5]),'confidence':float(b[4])} for b in packet.boxes],'event_ids':[e['event_id'] for e in events]})
        save_json(folder/'frames.json',rows)
        report.append({'scenario':scenario,'processed':len(latencies),'source_frames':len(frames),'injected_unavailable':sum('reason' in r for r in rows),
            'p95_inference_ms':float(np.percentile(latencies,95)) if latencies else None,'events':len(tracker.events),
            'recall':None,'fp_per_minute':None,'recovered_after_gap':scenario=='recovery' and bool(rows[-1].get('detections') is not None),
            'actual_predict_errors':sum(r.get('reason')=='actual_predict_error' for r in rows),
            'failure_kind':'software injection, not physical camera/GPU/temperature testing'})
        print('Сценарий',scenario,'завершён',flush=True)
    result={'ready':True,'scenarios':report,'device':'cpu','model':str(model_path),'model_sha256':sha256(model_path),'video_sha256':sha256(video),
        'execution_pass':all(r['actual_predict_errors']==0 for r in report) and report[0]['processed']>0,
        'policy':policy,'parameters':{'image_size':size,'confidence':conf,'limit':limit,'every':every},
        'quality_metrics_available':False,'reason':'No independently annotated ground truth supplied. Measurements are CPU predict timings after warmup, not camera-to-card latency.',
        'warning':'Video repetition and effects do not demonstrate real 300–400 km/h or temperature resistance.'}
    save_json(output/'report.json',result); save_json(output/'status.json',{'ready':True})
    with (output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(report[0])); writer.writeheader(); writer.writerows(report)
    lines=['# Репетиция финала','CPU-инференс после прогрева. Независимого GT нет; Recall и FP/мин не рассчитаны.','',
        '| Сценарий | Обработано | Имитаций отказа | P95 predict, мс | Событий |','|---|---:|---:|---:|---:|']
    lines += [f"| {r['scenario']} | {r['processed']} | {r['injected_unavailable']} | {round(r['p95_inference_ms'],1) if r['p95_inference_ms'] is not None else 'N/A'} | {r['events']} |" for r in report]
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import zipfile
    with zipfile.ZipFile(str(output)+'.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in output.rglob('*'):
            if path.is_file(): z.write(path,path.relative_to(output).as_posix())
    return result
