"""Persistent cross-run observations. No CV inference, implicit geolocation or train control."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager

SCHEMA = 'panoptr.route.v1'
KINDS = {'person': 120, 'drone': 120, 'obstacle': 600, 'rock': 600,
         'track_defect': 86400, 'contact_defect': 86400}
LABELS = {'person':'Человек', 'drone':'БПЛА', 'obstacle':'Препятствие', 'rock':'Камень',
          'track_defect':'Дефект пути', 'contact_defect':'Контактная сеть'}

def utcnow():
    return datetime.now(timezone.utc)

def stamp(value):
    if not isinstance(value, str): raise ValueError('Нужно время ISO 8601 с часовым поясом.')
    try: result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError: raise ValueError('Неверное время ISO 8601.')
    if result.tzinfo is None: raise ValueError('У времени отсутствует часовой пояс.')
    return result.astimezone(timezone.utc)

def text(value, name, limit=160):
    if not isinstance(value, str) or not value.strip() or len(value)>limit:
        raise ValueError(f'Неверное поле {name}.')
    return value.strip()

def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or not low<=value<=high:
        raise ValueError(f'Неверное поле {name}.')
    return float(value)

def validate(packet, now=None):
    now = now or utcnow()
    if not isinstance(packet, dict) or packet.get('schema') != SCHEMA:
        raise ValueError('Ожидается schema=panoptr.route.v1.')
    train = text(packet.get('train_id'), 'train_id')
    run = text(packet.get('run_id'), 'run_id')
    rows = packet.get('observations')
    if not isinstance(rows,list) or not 1<=len(rows)<=1000: raise ValueError('Нужно 1..1000 наблюдений.')
    result=[]
    for row in rows:
        if not isinstance(row,dict): raise ValueError('Наблюдение должно быть объектом.')
        row = dict(row)
        row['observation_id'] = text(row.get('observation_id'), 'observation_id')
        if row.get('kind') not in KINDS: raise ValueError('Неизвестный kind.')
        observed = stamp(row.get('observed_at'))
        if observed > now+timedelta(minutes=5): raise ValueError('Время события в будущем; проверьте часы поезда.')
        row['observed_at'] = observed.isoformat()
        row['quality_ok'] = row.get('quality_ok') is True
        row['confidence'] = number(row.get('confidence',0), 'confidence',0,1)
        row['train_id']=train; row['run_id']=run
        loc = row.get('location')
        if loc is not None:
            if not isinstance(loc,dict): raise ValueError('Неверное location.')
            loc = dict(loc)
            loc['route_id'] = text(loc.get('route_id'), 'route_id')
            loc['track_id'] = text(loc.get('track_id'), 'track_id')
            loc['chainage_m'] = number(loc.get('chainage_m'),'chainage_m',0,10000000)
            loc['uncertainty_m'] = number(loc.get('uncertainty_m'),'uncertainty_m',0,100000)
            if loc.get('position_kind') not in ('object','train'):
                raise ValueError('position_kind: object либо train. GNSS поезда не является местом объекта.')
            loc['verified'] = loc.get('verified') is True
            lat,lon = loc.get('latitude'),loc.get('longitude')
            if (lat is None)!=(lon is None): raise ValueError('Нужны обе координаты latitude/longitude.')
            if lat is not None:
                loc['latitude']=number(lat,'latitude',-90,90)
                loc['longitude']=number(lon,'longitude',-180,180)
            row['location']=loc
        # Only a small protocol subset is persisted; no client-supplied paths or decisions.
        result.append({k:row.get(k) for k in ('observation_id','kind','observed_at','quality_ok','confidence','train_id','run_id','location')})
    return result

def located(row):
    loc=row.get('location')
    return bool(loc and loc['verified'] and loc['position_kind']=='object' and loc['uncertainty_m']<=25)

class RouteMemory:
    def __init__(self, path):
        self.path=Path(path).resolve(); self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS objects(
              id TEXT PRIMARY KEY, route TEXT NOT NULL, track TEXT NOT NULL, kind TEXT NOT NULL,
              chainage REAL NOT NULL, uncertainty REAL NOT NULL, first_seen TEXT NOT NULL,
              last_seen TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending', confirmed_observation TEXT,
              valid_until TEXT, note TEXT NOT NULL DEFAULT '');
            CREATE INDEX IF NOT EXISTS objects_place ON objects(route,track,kind,chainage);
            CREATE TABLE IF NOT EXISTS observations(
              id TEXT PRIMARY KEY, train TEXT NOT NULL, run TEXT NOT NULL, external_id TEXT NOT NULL,
              payload TEXT NOT NULL, digest TEXT NOT NULL, object_id TEXT REFERENCES objects(id),
              UNIQUE(train,run,external_id));
            CREATE TABLE IF NOT EXISTS audit(
              id INTEGER PRIMARY KEY, at TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
              object_id TEXT, details TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.path,timeout=5)
        c.row_factory=sqlite3.Row; c.execute('PRAGMA foreign_keys=ON')
        try:
            with c: yield c
        finally: c.close()

    def _audit(self,c,actor,action,obj,details,now):
        c.execute('INSERT INTO audit(at,actor,action,object_id,details) VALUES(?,?,?,?,?)',
                  (now.isoformat(),actor,action,obj,json.dumps(details,ensure_ascii=False)))

    def _attach(self,c,row,now):
        if not located(row): return None,'unlocated'
        loc=row['location']; at=stamp(row['observed_at'])
        candidates=c.execute('SELECT * FROM objects WHERE route=? AND track=? AND kind=? AND state NOT IN (\'closed\',\'rejected\')',
                             (loc['route_id'],loc['track_id'],row['kind'])).fetchall()
        matches=[]
        for obj in candidates:
            distance=abs(obj['chainage']-loc['chainage_m'])
            # Uncertainty cannot create arbitrarily large clusters. Transient matches are location incidents, not identities.
            radius=min(30,max(5,obj['uncertainty']+loc['uncertainty_m']))
            span=abs((at-stamp(obj['last_seen'])).total_seconds())
            if distance<=radius and (row['kind'] in ('track_defect','contact_defect') or span<=KINDS[row['kind']]):
                matches.append((distance,obj['id'],obj))
        if matches:
            _,obj_id,obj=min(matches,key=lambda a:(a[0],a[1]))
            c.execute('UPDATE objects SET first_seen=?,last_seen=? WHERE id=?',
                      (min(stamp(obj['first_seen']),at).isoformat(),max(stamp(obj['last_seen']),at).isoformat(),obj_id))
            return obj_id,'repeat'
        obj_id=uuid.uuid4().hex
        c.execute('INSERT INTO objects(id,route,track,kind,chainage,uncertainty,first_seen,last_seen) VALUES(?,?,?,?,?,?,?,?)',
                  (obj_id,loc['route_id'],loc['track_id'],row['kind'],loc['chainage_m'],loc['uncertainty_m'],row['observed_at'],row['observed_at']))
        self._audit(c,'receiver','new',obj_id,{'train':row['train_id'],'run':row['run_id']},now)
        return obj_id,'new'

    def ingest(self,packet,now=None):
        now=now or utcnow(); rows=validate(packet,now)
        result=dict(new=0,repeat=0,duplicate=0,unlocated=0)
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            for row in rows:
                payload=json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)
                digest=hashlib.sha256(payload.encode()).hexdigest()
                old=c.execute('SELECT * FROM observations WHERE train=? AND run=? AND external_id=?',
                              (row['train_id'],row['run_id'],row['observation_id'])).fetchone()
                if old:
                    if old['digest']!=digest: raise ValueError('Конфликт: тот же ID наблюдения с другим содержимым.')
                    result['duplicate']+=1; continue
                obj_id,status=self._attach(c,row,now)
                c.execute('INSERT INTO observations VALUES(?,?,?,?,?,?,?)',
                          (uuid.uuid4().hex,row['train_id'],row['run_id'],row['observation_id'],payload,digest,obj_id))
                result[status]+=1
        return result

    def objects(self,now=None):
        now=now or utcnow()
        with self.connect() as c:
            result=[dict(r) for r in c.execute('SELECT o.*, COUNT(v.id) AS observations,COUNT(DISTINCT v.run || char(0) || v.train) AS runs FROM objects o LEFT JOIN observations v ON v.object_id=o.id GROUP BY o.id ORDER BY o.chainage')]
        for row in result:
            row['fresh']=bool(row['state']=='confirmed' and row['valid_until'] and stamp(row['valid_until'])>now)
            row['stale']=stamp(row['last_seen'])+timedelta(seconds=KINDS[row['kind']])<=now or (row['state']=='confirmed' and not row['fresh'])
        return result

    def unlocated(self):
        with self.connect() as c:
            return [dict(r) for r in c.execute('SELECT * FROM observations WHERE object_id IS NULL ORDER BY rowid DESC')]

    def observations(self,obj_id):
        with self.connect() as c:
            return [dict(r) for r in c.execute('SELECT * FROM observations WHERE object_id=? ORDER BY rowid DESC',(obj_id,))]

    def decide(self,obj_id,action,actor,note='',observation_id=None,now=None):
        now=now or utcnow(); actor=text(actor,'Оператор'); note=str(note)[:2000]
        if action not in ('confirm','reject','close','reopen'): raise ValueError('Неверное решение.')
        if action in ('reject','close') and not note.strip(): raise ValueError('Для закрытия/отклонения укажите основание.')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            obj=c.execute('SELECT * FROM objects WHERE id=?',(obj_id,)).fetchone()
            if not obj: raise ValueError('Объект не найден.')
            valid=None; chosen=None
            if action=='confirm':
                rows=c.execute('SELECT * FROM observations WHERE object_id=?',(obj_id,)).fetchall()
                eligible=[]
                for obs in rows:
                    row=json.loads(obs['payload']); at=stamp(row['observed_at'])
                    if located(row) and row['quality_ok'] and at<=now and at+timedelta(seconds=KINDS[row['kind']])>now:
                        if observation_id is None or observation_id==obs['id']: eligible.append((at,obs['id'],row))
                if not eligible: raise ValueError('Нет свежего пригодного наблюдения с проверенным местом объекта.')
                at,chosen,row=max(eligible,key=lambda item:item[0])
                valid=(at+timedelta(seconds=KINDS[row['kind']])).isoformat()
            state={'confirm':'confirmed','reject':'rejected','close':'closed','reopen':'pending'}[action]
            c.execute('UPDATE objects SET state=?,confirmed_observation=?,valid_until=?,note=? WHERE id=?',
                      (state,chosen,valid,note,obj_id))
            self._audit(c,actor,action,obj_id,{'note':note,'observation_id':chosen,'valid_until':valid},now)

    def locate(self,observation_id,location,actor,now=None):
        now=now or utcnow(); actor=text(actor,'Оператор')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            obs=c.execute('SELECT * FROM observations WHERE id=?',(observation_id,)).fetchone()
            if not obs or obs['object_id']: raise ValueError('Нужно наблюдение без места.')
            row=json.loads(obs['payload']); row['location']=location
            row=validate({'schema':SCHEMA,'train_id':row['train_id'],'run_id':row['run_id'],'observations':[row]},now)[0]
            if not located(row): raise ValueError('Нужно проверенное место объекта с погрешностью не более 25 м.')
            obj_id,status=self._attach(c,row,now)
            # Original receiver payload/digest remain immutable for idempotent retransmission.
            c.execute('UPDATE observations SET object_id=?,payload=? WHERE id=?',(obj_id,json.dumps(row,ensure_ascii=False,sort_keys=True),observation_id))
            self._audit(c,actor,'locate',obj_id,{'observation_id':observation_id,'location':location},now)
            return obj_id

    def warnings(self,route,track,chainage,direction=1,lookahead=10000,run_id='',now=None):
        chainage=number(chainage,'chainage',0,10000000)
        lookahead=number(lookahead,'lookahead',1,100000)
        if direction not in (-1,1): raise ValueError('Направление должно быть -1 или +1.')
        result=[]
        for obj in self.objects(now):
            distance=(obj['chainage']-chainage)*direction
            if obj['route']!=route or obj['track']!=track or not obj['fresh'] or not 0<=distance<=lookahead: continue
            if run_id:
                obs=next(r for r in self.observations(obj['id']) if r['id']==obj['confirmed_observation'])
                if obs['run']==run_id: continue
            result.append({**obj,'distance_m':distance})
        return sorted(result,key=lambda r:r['distance_m'])

    def export(self,path):
        with self.connect() as c:
            data={table:[dict(r) for r in c.execute('SELECT * FROM '+table)] for table in ('objects','observations','audit')}
        data['schema']='panoptr.route.archive.v1'; data['exported_at']=utcnow().isoformat()
        Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def import_file(memory,path):
    path=Path(path)
    if path.stat().st_size>16*1024*1024: raise ValueError('Файл больше 16 МБ; разделите его на пакеты.')
    raw=path.read_text(encoding='utf-8-sig')
    packets=[json.loads(line) for line in raw.splitlines() if line.strip()] if path.suffix.lower()=='.jsonl' else [json.loads(raw)]
    # Prevalidate every packet before starting import. Each packet is an atomic delivery.
    for packet in packets: validate(packet)
    totals=dict(new=0,repeat=0,duplicate=0,unlocated=0)
    for packet in packets:
        for key,value in memory.ingest(packet).items(): totals[key]+=value
    return totals

def session_packets(session,train_id,run_id):
    """Use original evidence metadata; unknown metric geometry stays unknown."""
    session=Path(session)
    if not (session/'session.json').is_file(): raise ValueError('Нет session.json в журнале поезда.')
    for path in sorted((session/'events').glob('*/event.json')):
        event=json.loads(path.read_text(encoding='utf-8'))
        role=event.get('role',''); name=event.get('class_name','').lower()
        if role=='aerial_object' or 'drone' in name or 'uav' in name: kind='drone'
        elif name in ('person','human','pedestrian','человек'): kind='person'
        elif 'rock' in name or 'stone' in name: kind='rock'
        elif role=='overhead_object': kind='contact_defect'
        elif role in ('defect','track_defect') or any(s in name for s in ('defect','missing','crack','fastener','sleeper')): kind='track_defect'
        else: kind='obstacle'
        evidence=event.get('evidence_observation',{})
        prov=evidence.get('provenance',event.get('provenance',{}))
        # A receiver packet or a verified object location may be added by the train adapter.
        location=prov.get('object_location')
        yield {'schema':SCHEMA,'train_id':train_id,'run_id':run_id,'observations':[{
            'observation_id':event['event_id'],'kind':kind,
            'observed_at':event['created_utc'],
            'quality_ok':evidence.get('quality',event.get('quality',{})).get('assessed') is True,
            'confidence':0., 'location':location}]}

def seed_demo(memory):
    now=utcnow()
    def deliver(run,kind,metres,age,quality=True,location=True):
        row={'observation_id':uuid.uuid4().hex,'kind':kind,'observed_at':(now-timedelta(seconds=age)).isoformat(),
             'quality_ok':quality,'confidence':.85,'location':({'route_id':'DEMO-ВСМ','track_id':'I',
             'chainage_m':metres,'uncertainty_m':2,'position_kind':'object','verified':True} if location else None)}
        memory.ingest({'schema':SCHEMA,'train_id':'DEMO-01','run_id':run,'observations':[row]},now)
    deliver('Рейс-А','track_defect',125000,3600)
    deliver('Рейс-Б','track_defect',125002,600)
    deliver('Рейс-Б','person',127000,20)
    deliver('Рейс-А','rock',130000,1800)
    deliver('Рейс-Б','obstacle',131000,10,False)
    deliver('Рейс-Б','drone',0,15,location=False)
    for obj in memory.objects(now):
        if obj['kind'] in ('track_defect','person'): memory.decide(obj['id'],'confirm','Демо-оператор',now=now)
