"""Frozen, independently reviewed temporal GT and operational event metrics.

No training, network, or GPU. Missing annotation never means background.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy, csv, hashlib, json, math, shutil, zipfile
from final_core import save_json, sha256, iou, DEFAULT_POLICY, validate_policy

PROTOCOL = {'spatial_iou': .3, 'max_keyframe_gap_s': .5,
            'min_valid_hits': 1, 'deadline_s': 1., 'small_area_fraction': .002}

def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Нужно конечное число.')
    return float(value)

def intervals(rows, duration):
    out=[]
    for row in rows:
        a,b=map(number,row)
        if not 0 <= a < b <= duration + 1e-6: raise ValueError('Интервал должен лежать внутри сцены.')
        out.append([a,b])
    merged=[]
    for a,b in sorted(out):
        if merged and a <= merged[-1][1]: merged[-1][1]=max(b,merged[-1][1])
        else: merged.append([a,b])
    return merged

def length(rows): return sum(b-a for a,b in rows)
def contains(rows,t): return any(a <= t < b for a,b in rows)
def overlap(a,b):
    return sum(max(0.,min(y,v)-max(x,u)) for x,y in a for u,v in b)

def validate(project, complete=False):
    p=copy.deepcopy(project)
    if p.get('format') != 'panoptr_event_gt_v1': raise ValueError('Неверный формат проекта событий.')
    protocol={**PROTOCOL, **p.get('protocol',{})}
    for key in PROTOCOL: protocol[key]=number(protocol[key])
    if not 0 < protocol['spatial_iou'] <= 1 or not 0 < protocol['small_area_fraction'] < 1:
        raise ValueError('Неверные доли протокола.')
    if protocol['max_keyframe_gap_s'] <= 0 or protocol['deadline_s'] < 0 or protocol['min_valid_hits'] < 1 or not protocol['min_valid_hits'].is_integer():
        raise ValueError('Неверные параметры временного сопоставления.')
    p['protocol']=protocol; seen=set()
    system=p.setdefault('system',{'image_size':640,'confidence':.25,'every':1,'policy':copy.deepcopy(DEFAULT_POLICY),'class_map':{}})
    if int(system['image_size'])<32 or int(system['every'])<1 or not 0<=number(system['confidence'])<=1:
        raise ValueError('Неверные параметры CPU-прогона.')
    system['policy']=validate_policy(system['policy'])
    if not isinstance(system.get('class_map',{}),dict): raise ValueError('Нужна карта имён классов.')
    if not p.get('scenes'): raise ValueError('Нет контрольных сцен.')
    for s in p['scenes']:
        sid=s['scene_id']
        if not isinstance(sid,str) or not sid or sid in seen: raise ValueError('Повторяющийся/пустой ID сцены.')
        seen.add(sid); duration=number(s['duration_s'])
        if duration <= 0: raise ValueError('Пустая сцена.')
        if number(s['fps'])<=0: raise ValueError('Неверная частота временной шкалы.')
        if not s.get('source_group'): raise ValueError('Нужна группа исходной съёмки, а не только имя файла.')
        if s.get('timing_kind') not in ('video_nominal','measured','synthetic'):
            raise ValueError('Укажите происхождение времени сцены.')
        s['reviewed']=intervals(s.get('reviewed',[]),duration)
        s['unusable']=intervals(s.get('unusable',[]),duration)
        if abs(overlap(s['unusable'],s['reviewed'])-length(s['unusable']))>1e-6:
            raise ValueError('Непригодные интервалы должны быть просмотрены.')
        if complete and (abs(length(s['reviewed'])-duration)>1e-6 or not s.get('reviewer')):
            raise ValueError('Для фиксации нужен полный просмотр сцены и имя проверяющего.')
        for w in ('width','height'):
            if int(s[w]) <= 0: raise ValueError('Неверный размер изображения.')
        ids=set()
        for e in s.get('events',[]):
            if e['gt_id'] in ids or not e['gt_id']: raise ValueError('Повторяющийся/пустой ID объекта.')
            ids.add(e['gt_id']); a,b=number(e['start_s']),number(e['end_s'])
            if not 0 <= a < b <= duration: raise ValueError('Событие вне сцены.')
            if e.get('kind','target') not in ('target','normal'): raise ValueError('Неизвестный тип GT.')
            if not e.get('class_name'): raise ValueError('Нужен класс GT.')
            if e.get('location') not in ('inside','near','outside','unknown'): raise ValueError('Неверное положение GT.')
            times=[]
            for k in e.get('keyframes',[]):
                t=number(k['t']); box=list(map(number,k['box']))
                if len(box)!=4 or not 0 <= box[0] < box[2] <= s['width'] or not 0 <= box[1] < box[3] <= s['height']:
                    raise ValueError('GT bbox должен быть в пикселях исходного изображения.')
                if not a <= t <= b: raise ValueError('Ключевой кадр вне интервала объекта.')
                times.append(t)
            if (complete and not times) or len(times)!=len(set(times)): raise ValueError('Нужны уникальные ключевые кадры объекта.')
            if complete:
                # Never extrapolate identity through long unannotated motion.
                times=sorted(times); gap=protocol['max_keyframe_gap_s']
                if times[0]-a>gap or b-times[-1]>gap or any(y-x>gap for x,y in zip(times,times[1:])):
                    raise ValueError('Слишком редкие ключевые bbox: добавьте кадры или уточните протокол до фиксации.')
    return p

def new_project(video, scene_id='scene_001', group=None, timing_kind='video_nominal'):
    import cv2
    path=Path(video).resolve()
    if not path.is_file(): raise ValueError('Нужен локальный видеоролик.')
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened(): raise ValueError('Видео не открывается.')
        fps=cap.get(cv2.CAP_PROP_FPS); count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps<=0 or count<=0: raise ValueError('Нет достоверного номинального времени видео.')
        w,h=int(cap.get(3)),int(cap.get(4))
    finally: cap.release()
    return {'format':'panoptr_event_gt_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'protocol':copy.deepcopy(PROTOCOL),'independence':{'status':'unverified','note':'Проверить обучение, подбор порогов, источники и прошлые оценки.'},
        'scenes':[{'scene_id':scene_id,'source_group':group or path.stem,'source':str(path),
            'source_sha256':sha256(path),'duration_s':count/fps,'fps':fps,'width':w,'height':h,
            'timing_kind':timing_kind,'condition':'original','reviewer':'','reviewed':[],
            'unusable':[],'events':[],'scene_tags':[]}]}

def freeze(project_path, output):
    src=Path(project_path).resolve(); p=validate(json.loads(src.read_text(encoding='utf-8')),True)
    out=Path(output).resolve()
    if out.exists(): raise ValueError('Фиксация создаёт новую папку.')
    sources=[]
    for s in p['scenes']:
        path=Path(s['source']); path=path if path.is_absolute() else src.parent/path
        path=path.resolve()
        if not path.is_file() or sha256(path)!=s['source_sha256']: raise ValueError('Источник изменился: '+s['scene_id'])
        metadata=new_project(path)['scenes'][0]
        if any(metadata[k]!=s[k] for k in ('width','height')) or abs(metadata['fps']-s['fps'])>.001 or abs(metadata['duration_s']-s['duration_s'])>1e-5:
            raise ValueError('Размер/номинальное время GT не совпадают с видео: '+s['scene_id'])
        sources.append(path)
    out.mkdir(parents=True); assets=out/'assets'; assets.mkdir()
    for index,(s,path) in enumerate(zip(p['scenes'],sources)):
        dest=assets/(str(index)+path.suffix.lower()); shutil.copy2(path,dest)
        if sha256(dest)!=s['source_sha256']: raise ValueError('Ошибка копирования источника.')
        s['source']=dest.relative_to(out).as_posix()
    save_json(out/'ground_truth.json',p)
    lock={'format':'panoptr_event_lock_v1','frozen_utc':datetime.now(timezone.utc).isoformat(),
          'gt_sha256':sha256(out/'ground_truth.json'),'dataset_signature':signature(p),
          'assets':{s['source']:s['source_sha256'] for s in p['scenes']},
          'independence':p.get('independence',{}),'note':'Freeze prevents unnoticed changes; independence requires documented source audit.'}
    save_json(out/'lock.json',lock); return lock

def load_frozen(folder):
    folder=Path(folder).resolve(); lock=json.loads((folder/'lock.json').read_text(encoding='utf-8'))
    if sha256(folder/'ground_truth.json')!=lock['gt_sha256']: raise ValueError('GT изменён после фиксации.')
    p=validate(json.loads((folder/'ground_truth.json').read_text(encoding='utf-8')),True)
    if signature(p)!=lock['dataset_signature']: raise ValueError('Подпись набора не совпала.')
    for s in p['scenes']:
        asset=(folder/s['source']).resolve()
        if not asset.is_relative_to(folder/'assets') or lock['assets'].get(s['source'])!=s['source_sha256'] or sha256(asset)!=s['source_sha256']:
            raise ValueError('Контрольная сцена изменена.')
    return p,lock

def gt_box(e,t,gap):
    ks=sorted(e['keyframes'],key=lambda k:k['t'])
    before=[k for k in ks if k['t']<=t]; after=[k for k in ks if k['t']>=t]
    if before and after:
        a,b=before[-1],after[0]
        if b['t']-a['t']>gap: return None
        f=(t-a['t'])/(b['t']-a['t']) if b['t']!=a['t'] else 0.
        return [x+(y-x)*f for x,y in zip(a['box'],b['box'])]
    nearest=min(ks,key=lambda k:abs(k['t']-t))
    return nearest['box'] if abs(nearest['t']-t)<=gap else None

def validate_predictions(pred,p,lock):
    if pred.get('format')!='panoptr_event_predictions_v1' or pred.get('dataset_signature')!=lock['dataset_signature']:
        raise ValueError('Предсказания должны относиться к этому фиксированному набору.')
    if pred.get('system')!=p['system']: raise ValueError('Параметры системы отличаются от зафиксированного протокола.')
    scenes={s['scene_id']:s for s in p['scenes']}; found=set()
    for r in pred.get('scenes',[]):
        sid=r['scene_id']
        if sid not in scenes or sid in found: raise ValueError('Неизвестная/повторяющаяся сцена предсказаний.')
        found.add(sid); s=scenes[sid]
        if r.get('source_sha256')!=s['source_sha256']: raise ValueError('Неверный источник предсказаний.')
        r['assessed']=intervals(r.get('assessed',[]),s['duration_s'])
        ids=set()
        for e in r.get('events',[]):
            if not e.get('event_id') or e['event_id'] in ids: raise ValueError('Повторяющийся ID карточки.')
            ids.add(e['event_id']); ts=[]
            for o in e.get('observations',[]):
                t=number(o['t']); alarm=number(o['alarm_t']); box=list(map(number,o['box']))
                if not 0<=t<s['duration_s'] or alarm<t or len(box)!=4 or not 0<=box[0]<box[2]<=s['width'] or not 0<=box[1]<box[3]<=s['height']: raise ValueError('Неверное время/геометрия наблюдения.')
                if not isinstance(o.get('valid'),bool): raise ValueError('Каждое наблюдение должно явно указывать пригодность.')
                ts.append(t)
            if not ts or len(ts)!=len(set(ts)): raise ValueError('Пустая карточка или повтор кадра в карточке.')
    if found!=set(scenes): raise ValueError('Нет результатов всех сцен. Отсутствующая сцена не считается пустой.')

def alarm_observations(e,min_hits):
    hits=0; rows=[]
    for o in sorted(e['observations'],key=lambda o:(o['t'],o['alarm_t'])):
        hits+=int(o['valid'])
        if hits>=min_hits and o['valid']: rows.append(o)
    return rows

def edge(e,g,protocol):
    candidates=[]
    # A false card must not become a true first alarm merely because its track
    # later wanders onto a real object. Evaluate the first qualifying card alarm.
    for o in alarm_observations(e,protocol['min_valid_hits'])[:1]:
        if not g['start_s']<=o['t']<=g['end_s']: continue
        box=gt_box(g,o['t'],protocol['max_keyframe_gap_s'])
        score=iou(o['box'],box) if box else 0.
        if score>=protocol['spatial_iou']: candidates.append((o['alarm_t'],score,o))
    return min(candidates,key=lambda x:x[0]) if candidates else None

def max_matching(edges):
    # Maximum-cardinality bipartite assignment, deterministic IoU tie order.
    by={}
    for pi,gi,value in edges: by.setdefault(pi,[]).append((gi,value))
    matched={}
    def visit(pi,seen):
        for gi,v in sorted(by.get(pi,[]),key=lambda x:(-x[1][1],x[0])):
            if gi in seen: continue
            seen.add(gi)
            if gi not in matched or visit(matched[gi],seen): matched[gi]=pi; return True
        return False
    for pi in sorted(by): visit(pi,set())
    return {pi:gi for gi,pi in matched.items()}

def evaluate(project,pred,lock):
    p=validate(project,True); pred=copy.deepcopy(pred); validate_predictions(pred,p,lock)
    protocol=p['protocol']; pr={s['scene_id']:s for s in pred['scenes']}; details=[]; scenes=[]; gt_rows=[]
    for s in p['scenes']:
        r=pr[s['scene_id']]; targets=[g for g in s['events'] if g.get('kind','target')=='target']
        alarms=sorted([e for e in r.get('events',[]) if alarm_observations(e,protocol['min_valid_hits'])],
            key=lambda e:alarm_observations(e,protocol['min_valid_hits'])[0]['alarm_t'])
        edges=[]
        for pi,e in enumerate(alarms):
            for gi,g in enumerate(targets):
                value=edge(e,g,protocol) if e['class_name']==g['class_name'] else None
                if value: edges.append((pi,gi,value))
        assigned=max_matching(edges); edge_map={(pi,gi):v for pi,gi,v in edges}; found=set(assigned.values())
        for gi,g in enumerate(targets):
            pi=next((pi for pi,v in assigned.items() if v==gi),None)
            delay=max(0.,edge_map[pi,gi][0]-g['start_s']) if pi is not None else None
            usable=(g['end_s']-g['start_s'])-overlap([[g['start_s'],g['end_s']]],s['unusable'])>1e-6
            area=min((k['box'][2]-k['box'][0])*(k['box'][3]-k['box'][1])/(s['width']*s['height']) for k in g['keyframes'])
            row={'scene_id':s['scene_id'],'gt_id':g['gt_id'],'class_name':g['class_name'],'location':g['location'],
                'source_group':s['source_group'],'condition':s.get('condition','unknown'),'small':area<protocol['small_area_fraction'],
                'detected':pi is not None,'has_usable_interval':usable,'delay_s':delay,
                'within_deadline':delay is not None and delay<=protocol['deadline_s'],
                'predicted_event_id':alarms[pi]['event_id'] if pi is not None else None,
                'tags':g.get('tags',[]),'start_s':g['start_s'],'end_s':g['end_s'],
            'predicted_location':edge_map[pi,gi][2].get('location') if pi is not None else None}
            gt_rows.append(row)
        fp=duplicates=0
        for pi,e in enumerate(alarms):
            gi=assigned.get(pi); alternatives=[j for i,j,_ in edges if i==pi]
            duplicate=gi is None and any(j in found for j in alternatives)
            kind='matched' if gi is not None else 'duplicate' if duplicate else 'false_event'
            fp+=int(kind=='false_event'); duplicates+=int(duplicate)
            normal=[g['gt_id'] for g in s['events'] if g.get('kind')=='normal' and edge(e,g,protocol)]
            details.append({'scene_id':s['scene_id'],'event_id':e['event_id'],'class_name':e['class_name'],'result':kind,
                'source_group':s['source_group'],'condition':s.get('condition','unknown'),
                'gt_id':targets[gi]['gt_id'] if gi is not None else None,
                'possible_duplicate_of':[targets[j]['gt_id'] for j in alternatives] if duplicate else [],
                'normal_lookalike_ids':normal,'first_alarm_t':alarm_observations(e,protocol['min_valid_hits'])[0]['alarm_t']})
            details[-1]['first_source_t']=alarm_observations(e,protocol['min_valid_hits'])[0]['t']
            details[-1]['first_box']=alarm_observations(e,protocol['min_valid_hits'])[0]['box']
        reviewed=length(s['reviewed']); available=overlap(s['reviewed'],r['assessed'])
        scenes.append({'scene_id':s['scene_id'],'source_group':s['source_group'],'condition':s.get('condition','unknown'),
            'duration_s':reviewed,'gt_events':len(targets),'detected':len(found),'false_events':fp,'duplicates':duplicates,
            'false_events_per_minute':fp/(reviewed/60) if reviewed else None,'duplicate_cards_per_minute':duplicates/(reviewed/60) if reviewed else None,
            'unassessed_s':reviewed-available,'unassessed_fraction':1-available/reviewed if reviewed else None,
            'gt_unusable_s':length(s['unusable']),
            'incorrectly_assessed_bad_s':overlap(s['unusable'],r['assessed']),
            'bad_observation_fraction':length(s['unusable'])/reviewed if reviewed else None})
        scenes[-1]['nonqualifying_cards']=len(r.get('events',[]))-len(alarms)
        scenes[-1]['unassessed_good_s']=max(0.,reviewed-length(s['unusable'])-(available-overlap(s['unusable'],r['assessed'])))
    def aggregate(rows):
        detected=[r for r in rows if r['detected']]; usable=[r for r in rows if r['has_usable_interval']]
        delays=sorted(r['delay_s'] for r in detected)
        import numpy as np
        return {'gt_events':len(rows),'detected':len(detected),'missed':len(rows)-len(detected),
            'recall':len(detected)/len(rows) if rows else None,
            'recall_with_usable_observation':sum(r['detected'] for r in usable)/len(usable) if usable else None,
            'deadline_recall':sum(r['within_deadline'] for r in rows)/len(rows) if rows else None,
            'delay_median_s':float(np.median(delays)) if delays else None,
            'delay_p95_s':float(np.percentile(delays,95)) if delays else None}
    summary=aggregate(gt_rows); total=sum(s['duration_s'] for s in scenes)
    summary.update({'duration_s':total,'false_events':sum(s['false_events'] for s in scenes),
        'duplicates':sum(s['duplicates'] for s in scenes),'unassessed_s':sum(s['unassessed_s'] for s in scenes)})
    summary['false_events_per_minute']=summary['false_events']/(total/60)
    summary['duplicate_cards_per_minute']=summary['duplicates']/(total/60)
    summary['unassessed_fraction']=summary['unassessed_s']/total
    summary['incorrectly_assessed_bad_s']=sum(s['incorrectly_assessed_bad_s'] for s in scenes)
    summary['nonqualifying_cards']=sum(s['nonqualifying_cards'] for s in scenes)
    summary['unassessed_good_s']=sum(s['unassessed_good_s'] for s in scenes)
    loc=[r for r in gt_rows if r['detected'] and r['location']!='unknown' and r['predicted_location'] in ('inside','near','outside')]
    summary['matched_events_with_location']=len(loc)
    summary['location_accuracy_on_matched']=sum(r['location']==r['predicted_location'] for r in loc)/len(loc) if loc else None
    breakdown={}
    for key in ('class_name','location','source_group','condition','small'):
        breakdown[key]={str(v):aggregate([r for r in gt_rows if r[key]==v]) for v in sorted(set(r[key] for r in gt_rows),key=str)}
    alert_breakdown={}
    for key in ('class_name','source_group','condition'):
        values=set(d[key] for d in details)
        if key!='class_name': values.update(s[key] for s in scenes)
        groups={}
        for value in sorted(values):
            ds=[d for d in details if d[key]==value]
            duration=total if key=='class_name' else sum(s['duration_s'] for s in scenes if s[key]==value)
            counts={k:sum(d['result']==k for d in ds) for k in ('matched','false_event','duplicate')}
            groups[str(value)]={**counts,'duration_s':duration,'false_events_per_minute':counts['false_event']/(duration/60),
                'duplicates_per_minute':counts['duplicate']/(duration/60)}
        alert_breakdown[key]=groups
    return {'format':'panoptr_event_report_v1','dataset_signature':lock['dataset_signature'],'protocol':protocol,
        'model':pred.get('model'), 'system':p['system'],'independence':p.get('independence',{}),'summary':summary,
        'scenes':scenes,'ground_truth_results':gt_rows,'predicted_event_results':details,'breakdown':breakdown,'alert_breakdown':alert_breakdown,
        'definitions':{'recall':'One-to-one physical-object event matching of FIRST qualifying alarm, same class, source-time and bbox IoU; all annotated target events. A later overlapping observation cannot redeem a false first card.',
            'false_events':'Unmatched qualifying cards, excluding duplicate cards; wrong classes are false events and missed GT.',
            'denominator':'Full manually reviewed duration including bad observation intervals; no division by inference duration.',
            'duplicates':'Extra qualifying card compatible with an already matched GT object; reported separately from false events.',
            'delay':'First matching qualifying alarm_t minus GT start; misses excluded from delay statistics and included in Recall.',
            'availability':'Duration of explicit system assessed intervals; missing intervals mean unassessed.',
            'usable_recall':'GT events with at least some independently labelled usable time; does not prove model analysed that time.',
            'timing':pred.get('timing_definition','Imported timeline; provenance must state whether measured or simulated.')},
        'limitations':['Frozen dataset is not automatically independent of training or threshold selection.',
            'Small stratum means minimum annotated bbox area fraction, not a physical size.',
            'No scene confidence intervals yet; short clips do not establish deployment accuracy.']}

def write_report(report,output):
    out=Path(output)
    if out.exists(): raise ValueError('Отчёт записывается в новую папку.')
    out.mkdir(parents=True); save_json(out/'report.json',report)
    for name in ('scenes','ground_truth_results','predicted_event_results'):
        rows=report[name]
        if rows:
            with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    s=report['summary']; text=['# Проверка событий Panoptr',
        'Независимость: '+str(report['independence'].get('status','unverified')),
        'Подпись набора: '+report['dataset_signature'],'', '| Показатель | Значение |','|---|---:|']
    text += [f'| {k} | {v if v is not None else "N/A"} |' for k,v in s.items()]
    text += ['', '## Определения']+[f'- {k}: {v}' for k,v in report['definitions'].items()]
    (out/'report.md').write_text('\n'.join(text),encoding='utf-8')
    with zipfile.ZipFile(str(out)+'.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in out.iterdir(): z.write(path,path.name)
    return report

def score(frozen,predictions,output):
    if Path(output).resolve().is_relative_to(Path(frozen).resolve()): raise ValueError('Отчёт должен быть вне фиксированного набора.')
    p,lock=load_frozen(frozen); pred=json.loads(Path(predictions).read_text(encoding='utf-8'))
    result=evaluate(p,pred,lock); result['prediction_sha256']=sha256(predictions)
    return write_report(result,output)

def infer(frozen, model_path, output):
    """CPU replay of a serial latest-frame schedule; model time advances media clock.

    This is explicitly a schedule simulation, not measured physical camera/card latency.
    All thresholds and schedule settings come from frozen GT, not this invocation.
    """
    import time, cv2, numpy as np
    from final_core import EventAssociator, frame_health
    p,lock=load_frozen(frozen); path=Path(model_path).resolve(); out=Path(output).resolve()
    if out.is_relative_to(Path(frozen).resolve()): raise ValueError('Прогон должен быть вне фиксированного набора.')
    if not path.is_file() or path.suffix.lower()!='.pt': raise ValueError('Нужны локальные YOLO .pt веса.')
    if out.exists(): raise ValueError('Нужна новая папка CPU-прогона.')
    model_hash=sha256(path); import torch
    torch.set_num_threads(2)
    from ultralytics import YOLO
    from main import InferenceScheduler
    started=time.monotonic(); model=YOLO(str(path)); load_ms=(time.monotonic()-started)*1000
    if model.task not in ('detect','segment'): raise ValueError('Прогон поддерживает YOLO detect/segment.')
    out.mkdir(parents=True); save_json(out/'status.json',{'complete':False})
    result={'format':'panoptr_event_predictions_v1','dataset_signature':lock['dataset_signature'],
        'model':{'path':str(path),'sha256':model_hash,'load_ms':load_ms,'device':'cpu'},'system':p['system'],
        'timing_definition':'Serial latest-frame CPU schedule simulation. alarm_t=nominal video t+measured predict duration; no physical exposure, transport, GUI or I/O. Warmup excluded. Concurrent training affects duration.',
        'scenes':[]}; system=p['system']; mapping=system.get('class_map',{})
    for s in p['scenes']:
        print('CPU: '+s['scene_id'],flush=True); cap=cv2.VideoCapture(str(Path(frozen)/s['source']))
        if not cap.isOpened(): raise ValueError('Контрольное видео не открывается.')
        rows={}; trace=[]; assessed=[]; tracker=EventAssociator(); n=0; next_n=0; rates=[]; warmed=False
        try:
            while True:
                ok,image=cap.read()
                if not ok: break
                index=n; n+=1
                if index<next_n: continue
                t=index/s['fps']; start=time.monotonic(); error=None
                try:
                    if not warmed:
                        model.predict(image,device='cpu',imgsz=system['image_size'],conf=system['confidence'],verbose=False); warmed=True; start=time.monotonic()
                    detection=model.predict(image,device='cpu',imgsz=system['image_size'],conf=system['confidence'],verbose=False)[0]
                    packet=InferenceScheduler._make_packet(detection,image.shape)
                except Exception as exc: error=str(exc)
                dt=time.monotonic()-start; alarm=t+dt
                next_n=max(index+int(system['every']),math.ceil(alarm*s['fps']))
                end=min(s['duration_s'],next_n/s['fps'],t+system['policy']['max_age_ms']/1000)
                rates.append(t); rates=[v for v in rates if t-v<=3]
                rate=(len(rates)-1)/(rates[-1]-rates[0]) if len(rates)>1 and rates[-1]>rates[0] else 0.
                health=frame_health(image,0,dt,rate,system['policy'])
                if error: health['assessed']=False; health['reasons'].append('inference_error')
                if health['assessed'] and end>alarm: assessed.append([alarm,end])
                updates=[]
                if not error:
                    names={k:mapping.get(v,v) for k,v in packet.names.items()}
                    updates=tracker.update(0,0,index,t,packet.boxes,names,image.shape,system['policy'],health,packet.masks)
                for e in updates:
                    item=rows.setdefault(e['event_id'],{'event_id':e['event_id'],'class_name':e['class_name'],'observations':[]})
                    item['observations'].append({'t':t,'alarm_t':alarm,'box':e['box'],'valid':health['assessed'],
                        'location':e['zone']['location'],'confidence':e['confidence'],'source_frame':index})
                trace.append({'source_frame':index,'t':t,'alarm_t':alarm,'health':health,'error':error,
                    'event_ids':[e['event_id'] for e in updates]})
        finally: cap.release()
        # Truncated decoding must not be mistaken for a completed negative scene.
        expected=int(round(s['duration_s']*s['fps']))
        if n<expected-1: raise ValueError('Видео закончилось раньше зафиксированной длительности.')
        result['scenes'].append({'scene_id':s['scene_id'],'source_sha256':s['source_sha256'],
            'assessed':intervals(assessed,s['duration_s']),'events':list(rows.values()),'trace':trace,
            'decoded_frames':n,'processed_frames':len(trace),'failed_frames':sum(bool(r['error']) for r in trace)})
    if sha256(path)!=model_hash: raise ValueError('Веса изменились во время прогона; используйте завершённое обучение.')
    load_frozen(frozen) # also reject any mid-run source/GT mutation
    save_json(out/'predictions.json',result)
    report=evaluate(p,result,lock); write_report(report,out/'evaluation')
    save_json(out/'status.json',{'complete':True,'dataset_signature':lock['dataset_signature']}); return report

def compare_reports(first,second,output):
    a=json.loads(Path(first).read_text(encoding='utf-8')); b=json.loads(Path(second).read_text(encoding='utf-8'))
    for key in ('dataset_signature','protocol','system','definitions'):
        if a.get(key)!=b.get(key): raise ValueError('Сравнение требует одинакового набора, протокола и определения времени.')
    rows=[]
    for k,av in a['summary'].items():
        bv=b['summary'].get(k); rows.append({'metric':k,'A':av,'B':bv,'B_minus_A':bv-av if av is not None and bv is not None else None})
    out=Path(output)
    if out.exists(): raise ValueError('Нужна новая папка сравнения.')
    out.mkdir(parents=True); result={'dataset_signature':a['dataset_signature'],'models':[a.get('model'),b.get('model')],
        'independence':a.get('independence'),'metrics':rows,'report_sha256':[sha256(first),sha256(second)],
        'note':'No overall winner is selected automatically. CPU latency comparison while training is not a final benchmark.'}
    save_json(out/'comparison.json',result)
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return result

def audit_sources(project_path,roots,output,limit=1000):
    """Read-only exact-byte and decoded-pixel leakage check; explicit partial scope.

    Does not establish scene independence, near-duplicate or threshold-selection history.
    """
    import cv2, numpy as np
    from PIL import Image
    p=validate(json.loads(Path(project_path).read_text(encoding='utf-8')))
    if not roots: raise ValueError('Укажите папки обучения и подбора порогов.')
    paths=[]
    for root in roots:
        folder=Path(root).resolve()
        if not folder.is_dir(): raise ValueError('Нет папки: '+str(folder))
        paths.extend(x for x in folder.rglob('*') if x.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.webp','.avi','.mp4','.mov'))
    paths=sorted(set(paths)); selected=paths[:int(limit)] if limit else paths
    byte_index={}; pixel_index={}; errors=[]
    def pixels(image):
        return hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
    for path in selected:
        try:
            byte_index.setdefault(sha256(path),[]).append(str(path))
            if path.suffix.lower() not in ('.avi','.mp4','.mov'):
                with Image.open(path) as im: code=pixels(np.array(im.convert('RGB')))
                pixel_index.setdefault(code,[]).append(str(path))
        except Exception as exc: errors.append({'path':str(path),'error':str(exc)})
    matches=[]; base=Path(project_path).resolve().parent
    for s in p['scenes']:
        source=Path(s['source']); source=source if source.is_absolute() else base/source
        for target in byte_index.get(s['source_sha256'],[]): matches.append({'scene_id':s['scene_id'],'kind':'exact_file','reference':target})
        cap=cv2.VideoCapture(str(source)); n=0
        try:
            while True:
                ok,image=cap.read()
                if not ok: break
                code=pixels(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                for target in pixel_index.get(code,[]): matches.append({'scene_id':s['scene_id'],'kind':'decoded_pixel_match','source_frame':n,'reference':target})
                n+=1
        finally: cap.release()
    result={'checked_reference_files':len(selected),'total_reference_files':len(paths),'complete_file_scan':len(selected)==len(paths),
        'roots':[str(Path(r).resolve()) for r in roots],'errors':errors,'matches':matches,
        'independence_status':'overlap_found' if matches else 'unverified',
        'limitations':'Exact file/pixel checks can miss re-encoding, resizing and adjacent frames. Reference videos only byte-compared. Scene groups, threshold selection and previous evaluation require manual history review. No automatic certification.',
        'source_modified':False}
    if Path(output).exists(): raise ValueError('Нужен новый файл аудита.')
    save_json(output,result); return result
