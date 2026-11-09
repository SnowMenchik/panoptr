import os
"""Field demonstration profiles and auditable single-target trials; no training/control."""
from pathlib import Path
from datetime import datetime,timezone
import copy,csv,hashlib,html,json,math,re,shutil,time,uuid,zipfile

PROFILES={'field750':{'title':'Полигон 750 мм','gauge_mm':750.},
          'track1520':{'title':'Путь 1520 мм','gauge_mm':1520.},
          'model':{'title':'Учебный макет','gauge_mm':750.}}
TASKS={
 'fasteners':{'title':'Скрепления пути','model':'yolo26s_rfdd_fasteners_best.pt','classes':['missing','displaced','inverted','deformed','fractured'],'size':640},
 'rail':{'title':'Видимые повреждения рельса','model':'rail9_finetuned_2026.pt','classes':['rail_crack','rail_break','rail_surface_damage','rail_corrosion'],'size':640},
 'surface':{'title':'Маски поверхностных повреждений','model':'rail_surface_neu_seg_2026.pt','classes':['rail_surface_defect'],'size':640},
 'person':{'title':'Человек у пути','model':'person_railbench07_2026.pt','classes':['person'],'size':960}}

def now(): return datetime.now(timezone.utc).isoformat()
def summary_text(s):
    def number(value):return 'нет данных' if value is None else f'{value:.3f}'
    return '\n'.join([
      f"Проверено опытов с неизменными настройками: {s['reviewed_valid_trials']}; ждут проверки: {s['pending_review_trials']}",
      f"Целей: {s['target_events']}; обнаружено: {s['detected_events']}; пропущено: {s['missed_events']}",
      'Доля обнаруженных событий: '+number(s['event_recall']),
      f"Ложных событий: {s['false_events']}; в минуту: {number(s['false_events_per_minute'])}; повторных карточек: {s['duplicate_cards']}",
      'Доля времени без пригодного наблюдения (оценка по опросу): '+number(s['unassessed_fraction']),
      'Задержки первой карточки от ручной отметки, с: '+(', '.join(number(x) for x in s['first_card_delay_s']) or 'нет данных'),
      f"Вмешательств в ходе опытов: {s['assistance_count']}; всего действий человека: {s['all_assistance_actions']}",
      f"Исключено из метрик: смена настроек — {s['changed_configuration_trials']}; прервано — {s['interrupted_trials']}",
      'Это оценка карточек в данной серии; не измерение задержки камеры и не подтверждение работы на ВСМ.'])
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def profile(key,gauge_mm=None):
    if key not in PROFILES:raise ValueError('Неизвестный профиль.')
    p=copy.deepcopy(PROFILES[key]);p['id']=key
    if key=='model':
        if isinstance(gauge_mm,bool) or not isinstance(gauge_mm,(int,float)) or not math.isfinite(gauge_mm) or not 10<=gauge_mm<=2000:raise ValueError('Измерьте колею макета: 10–2000 мм.')
        p['gauge_mm']=float(gauge_mm)
    elif gauge_mm is not None and float(gauge_mm)!=p['gauge_mm']:raise ValueError('Для нестандартной колеи выберите макет.')
    p.update(metric_calibration_valid=False,industrial_speed_verified=False)
    return p
def configuration(app_root,task,site,source_kind,source,device='CPU',size=None,confidence=.25):
    if task not in TASKS or source_kind not in ('camera','file') or device not in ('CPU','GPU','Auto'):raise ValueError('Неверная задача, источник или устройство.')
    site=profile(site['id'],site.get('gauge_mm'));spec=TASKS[task];model=Path(app_root)/'models'/spec['model']
    if not model.is_file():raise ValueError('Нет локальной модели: '+str(model))
    if source_kind=='file' and not Path(source).is_file():raise ValueError('Выберите существующий локальный видеофайл.')
    if source_kind=='camera' and (isinstance(source,bool) or not isinstance(source,int) or source<0):raise ValueError('Выберите подключённую камеру.')
    size=size or spec['size']
    if size not in (320,416,512,640,960,1024,1280,1536) or not .05<=confidence<=.95:raise ValueError('Неверный размер/порог.')
    return {'profile':site,'task':task,'model':os.path.abspath(model),'model_sha256':digest(model),'source_kind':source_kind,'source':str(source),
            'device':device,'image_size':size,'confidence':confidence,'automatic_train_control':False}

class TrialSeries:
    """One physical target per positive trial. Ground-truth association is manual."""
    def __init__(self,root,config,operator,store_root=None,clock=time.monotonic):
        if not operator.strip():raise ValueError('Укажите испытателя.')
        self.root=Path(root).resolve()
        if self.root.exists():raise ValueError('Для серии нужна новая папка.')
        self.root.mkdir(parents=True);self.clock=clock;self.active=None;self.last_health=None
        self.data={'format':'panoptr.trials.v1','created_utc':now(),'operator':operator.strip(),'configuration':copy.deepcopy(config),
                   'event_store':str(Path(store_root).resolve()) if store_root else None,'trials':[],'actions':[],
                   'definitions':{'truth':'Manual trial declaration and explicit event judgments; not automatic GT.',
                   'delay':'First qualifying evidence card observed by host poll minus operator start marker; includes polling/disk delay, not camera exposure latency.',
                   'health':'Host poll estimate; gaps longer than one second are conservatively unassessed.',
                   'scope':'One target per positive trial; selected examples do not certify railway operation.'}}
        self.save()
    def save(self):write(self.root/'protocol.json',self.data)
    def stamp(self):
        value=float(self.clock())
        if not math.isfinite(value):raise ValueError('Нужно конечное монотонное время.')
        return value
    @classmethod
    def open(cls,root,clock=time.monotonic):
        obj=cls.__new__(cls);obj.root=Path(root).resolve();obj.clock=clock;obj.active=None;obj.last_health=None
        obj.data=json.loads((obj.root/'protocol.json').read_text(encoding='utf-8'))
        if obj.data.get('format')!='panoptr.trials.v1' or not isinstance(obj.data.get('trials'),list) or not isinstance(obj.data.get('actions'),list):raise ValueError('Неверный формат серии.')
        for t in obj.data['trials']:
            if not re.fullmatch(r'[0-9a-f]{32}',t.get('id','')):raise ValueError('Неверный ID опыта.')
            if any(not re.fullmatch(r'[0-9a-f]{32}',key) for key in t.get('events',{})):raise ValueError('Неверный ID карточки.')
            if t.get('status')=='running':t.update(status='interrupted',reviewed=False,end_utc=now(),interruption_reason='Серия открыта после завершения или сбоя процесса; длительность не восстановлена.')
        obj.action('open_series','Серия открыта; незавершённые опыты не включаются в метрики.',assistance=False);return obj
    def action(self,kind,reason,assistance=True):
        if not reason.strip():raise ValueError('Укажите причину вмешательства.')
        row={'utc':now(),'monotonic':self.stamp(),'kind':kind,'reason':reason.strip(),'assistance':bool(assistance),'trial_id':self.active['id'] if self.active else None}
        self.data['actions'].append(row)
        with (self.root/'actions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        if self.active and kind=='configuration_changed':self.active['configuration_changed']=True
        self.save()
    def begin(self,expected,class_name,conditions,events=()):
        if self.active:raise ValueError('Сначала завершите текущий опыт.')
        if expected not in ('target','negative') or (expected=='target' and not class_name.strip()):raise ValueError('Укажите тип опыта и класс одного целевого объекта.')
        self.active={'id':uuid.uuid4().hex,'expected':expected,'class_name':class_name.strip(),'conditions':conditions.strip(),
          'start_utc':now(),'start_monotonic':self.stamp(),'end_monotonic':None,'baseline_ids':[r['event_id'] for r in events],
          'events':{},'judgments':{},'health':[],'reviewed':False,'configuration_changed':False,'status':'running'}
        self.data['trials'].append(self.active);self.last_health=None;self.save();return self.active['id']
    def observe(self,events,assessed):
        if not self.active:return
        stamp=self.stamp();t=self.active
        if stamp<t['start_monotonic']:raise ValueError('Время опыта пошло назад.')
        if t['health'] and stamp<t['health'][-1]['t']:raise ValueError('Время наблюдения пошло назад.')
        t['health'].append({'t':stamp,'assessed':bool(assessed)})
        for event in events:
            key=event['event_id']
            if key in t['baseline_ids']:continue
            if self.data['configuration'].get('channel') is not None and event.get('channel')!=self.data['configuration']['channel']:continue
            if 'first_seen' in event and event['first_seen']<t['start_monotonic']:continue
            if not re.fullmatch(r'[0-9a-f]{32}',key):raise ValueError('Неверный ID события.')
            e=t['events'].setdefault(key,{'event':copy.deepcopy(event),'available_at':stamp,'qualified_at':None})
            e['event']=copy.deepcopy(event)
            if event.get('observation_state')=='observed_repeatedly' and event.get('evidence_saved') and e['qualified_at'] is None:e['qualified_at']=stamp
        self.save()
    def finish(self):
        if not self.active:raise ValueError('Нет активного опыта.')
        stamp=self.stamp()
        if stamp<=self.active['start_monotonic'] or (self.active['health'] and stamp<self.active['health'][-1]['t']):raise ValueError('Неверное время завершения.')
        self.active.update(end_monotonic=stamp,end_utc=now(),status='completed');self.active=None;self.save()
    def interrupt(self,reason):
        if self.active:
            self.action('interrupted',reason,assistance=False);self.active.update(status='interrupted',reviewed=False,end_utc=now(),interruption_reason=reason);self.active=None;self.save()
    def trial(self,key):
        for t in self.data['trials']:
            if t['id']==key:return t
        raise ValueError('Нет такого опыта.')
    def judge(self,trial_id,event_id,result,reason):
        t=self.trial(trial_id)
        if t['status']!='completed' or event_id not in t['events'] or result not in ('match','false','duplicate','ignore') or not reason.strip():raise ValueError('Завершите опыт и укажите обоснование оценки.')
        e=t['events'][event_id]
        if result!='ignore' and e['qualified_at'] is None:raise ValueError('Кандидат не стал подтверждённой повторными наблюдениями карточкой.')
        if result in ('match','duplicate'):
            if t['expected']!='target' or e['event']['class_name']!=t['class_name']:raise ValueError('Класс события не совпадает с заданной целью.')
            matches=[key for key,j in t['judgments'].items() if j['result']=='match' and key!=event_id]
            if result=='match' and matches:raise ValueError('Одной цели соответствует только одно первое обнаружение; оцените повтор отдельно.')
            if result=='duplicate' and not matches:raise ValueError('Сначала укажите основное обнаружение цели.')
            if result=='duplicate' and e['qualified_at']<t['events'][matches[0]]['qualified_at']:raise ValueError('Повтор не может предшествовать первому обнаружению.')
        if result!='match' and t['judgments'].get(event_id,{}).get('result')=='match' and any(j['result']=='duplicate' for j in t['judgments'].values()):raise ValueError('Сначала пересмотрите связанные повторы.')
        t['judgments'][event_id]={'result':result,'reason':reason.strip(),'reviewer':self.data['operator'],'utc':now()};t['reviewed']=False;self.save()
    def review(self,trial_id):
        t=self.trial(trial_id)
        if t['status']!='completed':raise ValueError('Опыт не завершён.')
        if set(t['events'])!=set(t['judgments']):raise ValueError('Проверьте каждую карточку, включая непригодные кандидаты.')
        t['reviewed']=True;self.save()
    def summary(self):
        completed=[t for t in self.data['trials'] if t['status']=='completed'];valid=[t for t in completed if t['reviewed'] and not t['configuration_changed']]
        positives=[t for t in valid if t['expected']=='target'];detected=sum(any(j['result']=='match' for j in t['judgments'].values()) for t in positives)
        duration=sum(t['end_monotonic']-t['start_monotonic'] for t in valid);bad=0.;delays=[]
        for t in valid:
            samples=t['health'];cursor=t['start_monotonic'];state=False
            for sample in samples+[{'t':t['end_monotonic'],'assessed':False}]:
                gap=max(0.,sample['t']-cursor);bad+=gap if not state else max(0.,gap-1.);cursor=sample['t'];state=sample['assessed']
            for key,j in t['judgments'].items():
                if j['result']=='match':delays.append(max(0.,t['events'][key]['qualified_at']-t['start_monotonic']))
        return {'completed_trials':len(completed),'reviewed_valid_trials':len(valid),'pending_review_trials':sum(not t['reviewed'] for t in completed),
          'changed_configuration_trials':sum(t['configuration_changed'] for t in completed),'target_events':len(positives),'detected_events':detected,
          'missed_events':len(positives)-detected,'event_recall':detected/len(positives) if positives else None,
          'false_events':sum(j['result']=='false' for t in valid for j in t['judgments'].values()),
          'false_events_per_minute':sum(j['result']=='false' for t in valid for j in t['judgments'].values())/(duration/60) if duration else None,
          'duplicate_cards':sum(j['result']=='duplicate' for t in valid for j in t['judgments'].values()),'reviewed_duration_s':duration,
          'unassessed_host_poll_s':bad,'unassessed_fraction':bad/duration if duration else None,'first_card_delay_s':delays,
          'assistance_count':sum(a['assistance'] and a['trial_id'] is not None for a in self.data['actions']),
          'all_assistance_actions':sum(a['assistance'] for a in self.data['actions']),'interrupted_trials':sum(t['status']=='interrupted' for t in self.data['trials']),
          'automatic_railway_diagnosis':False,'physical_camera_latency_measured':False}
    def export(self,folder):
        if self.active:raise ValueError('Завершите опыт перед экспортом.')
        folder=Path(folder).resolve()
        if folder.exists() or Path(str(folder)+'.zip').exists() or folder.is_relative_to(self.root):raise ValueError('Выберите новую папку вне текущей серии.')
        source=Path(self.data['event_store']).resolve() if self.data['event_store'] else None
        if source and folder.is_relative_to(source):raise ValueError('Отчёт должен быть вне журнала событий.')
        folder.mkdir(parents=True);write(folder/'protocol.json',self.data);summary=self.summary();write(folder/'summary.json',summary)
        links=[];missing=[]
        for t in self.data['trials']:
            for key in t['events']:
                event_dir=source/'events'/key if source else None
                if event_dir and event_dir.is_dir():
                    dest=folder/'evidence'/key
                    if not dest.exists():shutil.copytree(event_dir,dest,ignore=shutil.ignore_patterns('*.tmp*'))
                    if (dest/'evidence.png').exists():links.append((t['id'],key,'evidence/'+key+'/evidence.png'))
                    else:missing.append(key)
                else:missing.append(key)
        write(folder/'evidence_index.json',{'links':links,'missing':sorted(set(missing))})
        fields=['id','expected','class_name','status','reviewed','configuration_changed']
        with (folder/'trials.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(self.data['trials'])
        body='<h1>Panoptr — протокол серии испытаний</h1><p>Один целевой объект на положительный опыт. Сопоставление карточек выполнено испытателем. Это не приёмка железнодорожного оборудования.</p><pre>'+html.escape(summary_text(summary))+'</pre>'
        for tid,eid,link in links:body+='<p>Опыт '+html.escape(tid)+' · <a href="'+link+'">Кадр '+html.escape(eid)+'</a></p>'
        body+='<p>Подробности конфигурации, оценки и вмешательства: protocol.json. Отсутствующие доказательства: '+html.escape(', '.join(sorted(set(missing))) or 'нет')+'</p>'
        (folder/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:18px sans-serif;max-width:1100px;margin:40px auto}pre{white-space:pre-wrap}</style>'+body,encoding='utf-8')
        with zipfile.ZipFile(str(folder)+'.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in folder.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(folder).as_posix())
        return folder
