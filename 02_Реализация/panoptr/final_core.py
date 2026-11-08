"""Frame health, explicit image zones and observation-to-event association.

These are prototype measurements, not railway certification or collision probabilities.
"""
from dataclasses import dataclass,field
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,math,time,uuid
import cv2
import numpy as np
from trained_models import TRAINED_DEFECT_CLASSES,OVERHEAD_CLASSES

def read_image(path):
    from PIL import Image
    with Image.open(path) as im: return np.array(im.convert('RGB'))[:,:,::-1].copy()

DEFAULT_ROLES={**{k:'person' for k in ['person','human']},**{k:'obstacle' for k in ['rock','stone','boulder']},
    **{k:'defect' for k in ['missing','displaced','inverted','deformed','fractured','cracks','crack','corrosion','flakings','shelling','spalling','squats','wheel-burn']},
    **{k:'defect' for k in TRAINED_DEFECT_CLASSES},**{k:'overhead_object' for k in OVERHEAD_CLASSES},
    'drone':'aerial_object','dirt':'normal','unknown':'ignore','arc':'transient','spark':'transient','sparking':'transient'}
DEFAULT_POLICY={'max_age_ms':1500.,'min_blur':0.,'max_dark_fraction':.95,'max_bright_fraction':.95,
    'speed_kmh':0.,'visible_length_m':0.,'min_samples':3.,'near_pixels':20.,'min_zone_overlap':.05,
    'roles':DEFAULT_ROLES,'zone':[],'group_gap_s':1.,'track_iou':.2,'confirm_hits':2,
    'tracker':'iou','path_enabled':False,'geometry_model':'','geometry_confidence':.35,'geometry_image_size':640,
    'geometry_max_age_ms':700.,'geometry_switch':False,'geometry_calibration_file':''}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def save_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8'); temp.replace(path)

def validate_policy(value):
    p=copy.deepcopy(DEFAULT_POLICY); p.update(value)
    if p['tracker'] not in ('iou','bytetrack') or not isinstance(p['path_enabled'],bool) or not isinstance(p['geometry_switch'],bool): raise ValueError('Неверный режим пути/трекера.')
    if not isinstance(p['geometry_model'],str) or not isinstance(p['geometry_calibration_file'],str): raise ValueError('Пути к модели/калибровке должны быть строками.')
    if not isinstance(p['geometry_confidence'],(int,float)) or not 0<=p['geometry_confidence']<=1: raise ValueError('Порог сегментации: от0 до1.')
    if not isinstance(p['geometry_image_size'],int) or not 32<=p['geometry_image_size']<=1920 or not isinstance(p['geometry_max_age_ms'],(int,float)) or not math.isfinite(p['geometry_max_age_ms']) or p['geometry_max_age_ms']<=0: raise ValueError('Неверный размер/срок геометрии.')
    numeric=['max_age_ms','min_blur','max_dark_fraction','max_bright_fraction','speed_kmh','visible_length_m','min_samples','near_pixels','min_zone_overlap','group_gap_s','track_iou','confirm_hits']
    if any(not isinstance(p[k],(int,float)) or not math.isfinite(p[k]) or p[k]<0 for k in numeric): raise ValueError('Параметры контроля должны быть конечными неотрицательными числами.')
    if p['max_age_ms']<=0 or p['group_gap_s']<=0 or p['confirm_hits']<1: raise ValueError('Возраст, интервал и число наблюдений должны быть положительными.')
    if any(p[k]>1 for k in ['max_dark_fraction','max_bright_fraction','min_zone_overlap','track_iou']): raise ValueError('Доли должны находиться между 0 и 1.')
    if not isinstance(p['roles'],dict) or any(v not in ('person','obstacle','defect','transient','overhead_object','aerial_object','component','normal','ignore') for v in p['roles'].values()): raise ValueError('Неизвестная роль класса.')
    zone=p['zone']
    if zone:
        points=np.asarray(zone,float)
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all() or (points<0).any() or (points>1).any() or abs(cv2.contourArea(points.astype(np.float32)))<1e-6:
            raise ValueError('Зона: минимум три точки [x,y] от 0 до 1, невырожденный полигон.')
    return p

def frame_health(frame,captured_at,now,analysis_fps,policy):
    p=validate_policy(policy); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=float(cv2.Laplacian(gray,cv2.CV_64F).var()); dark=float((gray<=8).mean()); bright=float((gray>=247).mean())
    age=max(0.,(now-captured_at)*1000); reasons=[]
    if age>p['max_age_ms']: reasons.append('устаревший кадр')
    if blur<p['min_blur']: reasons.append('смаз/низкая детализация')
    if dark>p['max_dark_fraction']: reasons.append('недостаточная освещённость')
    if bright>p['max_bright_fraction']: reasons.append('засветка')
    coverage=None; coverage_status='not_calibrated'
    if p['speed_kmh']>0 and p['visible_length_m']>0:
        speed=p['speed_kmh']/3.6
        samples=max(0.,analysis_fps)*p['visible_length_m']/speed
        coverage={'speed_kmh':p['speed_kmh'],'visible_length_m':p['visible_length_m'],'expected_samples':samples,
            'sample_spacing_m':speed/analysis_fps if analysis_fps>0 else None,'required_samples':p['min_samples'],
            'kind':'user-supplied uniform-sampling estimate; not proof of detection'}
        coverage_status='insufficient' if samples<p['min_samples'] else 'estimated_sufficient'
        if coverage_status=='insufficient': reasons.append('недостаточное временное покрытие')
    return {'assessed':not reasons,'reasons':reasons,'age_ms':age,'blur_variance':blur,'dark_fraction':dark,'bright_fraction':bright,
        'actual_analysis_fps':analysis_fps,'coverage_status':coverage_status,'coverage':coverage,
        'blur_threshold_enabled':p['min_blur']>0,'blur_threshold_calibrated':False,'warning':'Setting a threshold alone does not prove calibration. Blur threshold and visible length require task-specific experiments.'}

def zone_location(box,shape,policy,mask=None):
    p=validate_policy(policy); h,w=shape[:2]
    if not p['zone']: return {'location':'zone_not_set','overlap':None,'basis':'no configured path zone'}
    poly=np.asarray(p['zone'],np.float32)*[w-1,h-1]; region=np.zeros((h,w),np.uint8); cv2.fillPoly(region,[poly.astype(np.int32)],1)
    if mask is None:
        object_mask=np.zeros((h,w),bool); x1,y1,x2,y2=box
        object_mask[max(0,int(y1)):min(h,int(math.ceil(y2))),max(0,int(x1)):min(w,int(math.ceil(x2)))]=True; basis='bbox overlap; approximate'
    else:
        object_mask=np.asarray(mask,dtype=bool)
        if object_mask.shape!=(h,w): raise ValueError('Маска зоны должна быть в исходном размере.')
        basis='instance mask overlap'
    area=int(object_mask.sum()); overlap=float((object_mask & (region>0)).sum()/area) if area else 0.
    foot=((float(box[0])+float(box[2]))/2,float(box[3])); distance=float(cv2.pointPolygonTest(poly.astype(np.float32),foot,True))
    location='inside' if overlap>=p['min_zone_overlap'] and overlap>0 else 'near' if abs(distance)<=p['near_pixels'] else 'outside'
    return {'location':location,'overlap':overlap,'foot_distance_px':distance,'basis':basis,'metric_distance_m':None}

def iou(a,b):
    x=max(0.,min(a[2],b[2])-max(a[0],b[0])); y=max(0.,min(a[3],b[3])-max(a[1],b[1])); intersection=x*y
    return intersection/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)

class EventAssociator:
    """Explicit simple IoU baseline, per channel/generation/class, one-to-one matches.
    Tracks create one event, but human decision is independent from repeated observations.
    """
    def __init__(self): self.tracks={}; self.events={}
    def reset(self,channel): self.tracks={k:v for k,v in self.tracks.items() if v['channel']!=channel}
    def update(self,channel,generation,sequence,stamp,boxes,names,shape,policy,health,masks=None,track_ids=None,locations=None):
        p=validate_policy(policy); eligible=[]; updates=[]
        for index,row in enumerate(boxes):
            if track_ids is not None and (index>=len(track_ids) or track_ids[index] is None): continue
            if len(row)<6 or not np.isfinite(row[:6]).all() or not 0<=row[4]<=1 or row[2]<=row[0] or row[3]<=row[1]: continue
            name=str(names.get(int(row[5]),int(row[5]))); role=p['roles'].get(name,'ignore')
            if role not in ('person','obstacle','defect','transient','overhead_object','aerial_object'): continue
            location=locations[index] if locations is not None else zone_location(row[:4],shape,p,masks[index] if masks is not None and index<len(masks) else None)
            if role=='overhead_object': location={'location':'not_applicable_overhead','basis':'contact network object; no ground corridor or ground-plane metres','risk':'overhead_review'}
            if role=='aerial_object': location={'location':'not_applicable_aerial','basis':'airborne candidate; ground corridor and ground-plane projection do not apply','risk':'aerial_review','metric_status':'not_applicable_aerial'}
            eligible.append({'box':list(map(float,row[:4])),'confidence':float(row[4]),'class_name':name,'role':role,'zone':location})
            if track_ids is not None: eligible[-1]['external_track_id']=str(track_ids[index])
        active=[(k,t) for k,t in self.tracks.items() if t['channel']==channel and t['generation']==generation and 0<=stamp-t['last_seen']<=p['group_gap_s']]
        matches=[]
        for index,o in enumerate(eligible):
            for key,t in active:
                if t['class_name']==o['class_name']:
                    score=(1. if t.get('external_track_id')==o.get('external_track_id') else 0.) if track_ids is not None else iou(o['box'],t['box'])
                    if score>=p['track_iou']: matches.append((score,index,key))
        assigned={}; used=set()
        for score,index,key in sorted(matches,reverse=True):
            if index not in assigned and key not in used: assigned[index]=key; used.add(key)
        for index,o in enumerate(eligible):
            key=assigned.get(index)
            if key is None:
                key=uuid.uuid4().hex; t={'event_id':key,'channel':channel,'generation':generation,'first_seen':stamp,'last_seen':stamp,'sequence':sequence,
                    'class_name':o['class_name'],'hits':0,'valid_hits':0,'observation_state':'candidate','operator_status':'unreviewed','created_utc':datetime.now(timezone.utc).isoformat(),'peak_confidence':0.}
                self.tracks[key]=t
            else: t=self.tracks[key]
            # Do not count a repeated callback for the same frame as independent confirmation.
            if t['hits'] and sequence<=t['sequence']: continue
            t.update(o); t['hits']+=1; t['last_seen']=stamp; t['sequence']=sequence; t['peak_confidence']=max(t['peak_confidence'],o['confidence'])
            t['valid_hits']+=int(health['assessed'])
            t['observation_state']='observed_repeatedly' if t['valid_hits']>=p['confirm_hits'] else 'candidate'
            t['quality']=health; t['requires_review']=True; t['alarm_probability']=None
            if o['role']=='aerial_object':t['automatic_collision_alarm']=False
            # A poor observation may still preserve a candidate as evidence, but never proves clear track.
            if not health['assessed']: t['observation_state']='unassessed_candidate'
            self.events[key]=copy.deepcopy(t); updates.append(copy.deepcopy(t))
        self.tracks={k:t for k,t in self.tracks.items() if stamp-t['last_seen']<=p['group_gap_s'] or t['channel']!=channel}
        return updates
