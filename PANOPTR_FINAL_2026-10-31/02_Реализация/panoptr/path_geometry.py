"""Visible rail/corridor proposals with provenance, route selection and quality gates.

Scores are engineering heuristics, never a calibrated probability of safety.
No extrapolated rail endpoints, implicit branch selection or uncalibrated metres.
"""
import copy,hashlib,json,math
import numpy as np,cv2

def frame_digest(image): return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
def source_key(reference): return json.dumps(reference,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def polygon_mask(points,shape):
    out=np.zeros(shape[:2],np.uint8); cv2.fillPoly(out,[np.asarray(points,np.float32).round().astype(np.int32)],1); return out.astype(bool)
def mask_iou(a,b): return float((a&b).sum()/max(1,(a|b).sum()))
def unknown_geometry(shape,sequence,generation,stamp,source,reason):
    return {'format':'panoptr_path_v1','shape':list(shape[:2]),'sequence':sequence,'generation':generation,'captured_at':stamp,'source_id':source,
        'frame_sha256':None,'candidates':[],'status':'unknown','reasons':[reason],'scores_are_probabilities':False,'metres_verified':False}

def propose_geometry(masks,boxes,names,shape,sequence=1,generation=1,stamp=0.,source='offline',frame_sha=None,min_conf=.35):
    h,w=shape[:2]
    if h<16 or w<16 or not 0<=min_conf<=1: raise ValueError('Неверная геометрия/порог.')
    result=unknown_geometry(shape,sequence,generation,stamp,source,'нет двух видимых рельсов'); result['frame_sha256']=frame_sha
    semantic={key:np.zeros((h,w),bool) for key in ('rail_left','rail_right','rail','trackbed','ballast')}; confidences=[]
    aliases={'left_rail':'rail_left','right_rail':'rail_right','track':'trackbed','railway_track':'trackbed'}
    for i,row in enumerate(boxes):
        if i>=len(masks) or len(row)<6 or not np.isfinite(row[:6]).all() or not 0<=row[4]<=1: continue
        name=aliases.get(str(names.get(int(row[5]),'')),str(names.get(int(row[5]),'')))
        a=np.asarray(masks[i],bool)
        if a.shape!=(h,w): raise ValueError('Маска должна совпадать с исходным кадром, без letterbox.')
        if name in semantic and row[4]>=min_conf:
            semantic[name]|=a
            if name.startswith('rail'): confidences.append(float(row[4]))
    rails=semantic['rail']|semantic['rail_left']|semantic['rail_right']; result['semantic_masks']=semantic
    # Scan visible rows; separated rail strips generate adjacent pair proposals.
    bins=np.linspace(int(h*.25),h-2,32).astype(int); pairs={}; excessive=False
    for y in bins:
        band=rails[max(0,y-2):min(h,y+3)].any(axis=0).astype(np.uint8); positions=np.flatnonzero(band)
        if len(positions)<2: continue
        groups=np.split(positions,np.flatnonzero(np.diff(positions)>max(3,int(w*.006)))+1)
        groups=[g for g in groups if len(g)>=2]
        if len(groups)>2: excessive=True
        if len(groups)>6: continue
        for i in range(len(groups)-1):
            if semantic['rail_left'].any() and semantic['rail_right'].any():
                # Named left/right rails cannot form the gap BETWEEN two separate tracks.
                left_votes=semantic['rail_left'][max(0,y-2):min(h,y+3),groups[i]].sum()
                wrong_left=semantic['rail_right'][max(0,y-2):min(h,y+3),groups[i]].sum()
                right_votes=semantic['rail_right'][max(0,y-2):min(h,y+3),groups[i+1]].sum()
                wrong_right=semantic['rail_left'][max(0,y-2):min(h,y+3),groups[i+1]].sum()
                if left_votes<=wrong_left or right_votes<=wrong_right: continue
            # Inner edges, not an expanded trackbed/background region.
            left,right=float(groups[i][-1]),float(groups[i+1][0]); width=right-left
            if not max(6,w*.02)<=width<=w*.95: continue
            pairs.setdefault(i,[]).append((float(y),left,right))
    candidates=[]
    for index,rows in pairs.items():
        if len(rows)<8: continue
        ys=np.array([r[0] for r in rows]); ls=np.array([r[1] for r in rows]); rs=np.array([r[2] for r in rows]); widths=rs-ls
        coverage=len(rows)/len(bins); gap=float(np.diff(ys).max()) if len(ys)>1 else h
        # Reject bridges across long invisible sections; no extrapolation to image bottom.
        discontinuity=gap>h*.15; roughness=float(np.median(np.abs(np.diff(widths,2))))/max(1.,float(np.median(widths)))
        mean_conf=min(confidences) if confidences else 0.; quality=min(mean_conf,coverage,max(0.,1-roughness*4))
        polygon=[[float(x),float(y)] for x,y in zip(ls,ys)]+[[float(x),float(y)] for x,y in zip(rs[::-1],ys[::-1])]
        mask=polygon_mask(polygon,(h,w)); reason=[]
        if coverage<.5: reason.append('мало видимых строк рельсов')
        if discontinuity: reason.append('большой разрыв видимости')
        if roughness>.15: reason.append('нестабильная ширина')
        if quality<min_conf: reason.append('низкий эвристический показатель')
        signature=hashlib.sha256(json.dumps(polygon).encode()).hexdigest()
        candidates.append({'id':f'route_{index+1}','polygon_px':polygon,'mask':mask,'signature':signature,'quality_score':quality,
            'min_model_confidence':mean_conf,'row_coverage':coverage,'width_roughness':roughness,'eligible':not reason,'reasons':reason})
    result['candidates']=candidates; result['topology_ambiguous']=excessive or len(candidates)>1
    eligible=[c for c in candidates if c['eligible']]
    result['status']='ambiguous' if result['topology_ambiguous'] and candidates else 'estimated' if len(eligible)==1 else 'low_confidence' if candidates else 'unknown'
    result['reasons']=[] if result['status']=='estimated' else ['несколько веток: подтвердите выбранный путь'] if result['status']=='ambiguous' else ['нет пригодного коридора по двум видимым рельсам']
    if not candidates and semantic['trackbed'].any(): result['reasons'].append('маска trackbed сама по себе не определяет выбранный путь')
    return result

class RouteState:
    def __init__(self): self.selection=None; self.require_confirmation=False
    def reset(self,required=False): self.selection=None; self.require_confirmation=required
    def confirm(self,geometry,candidate_id,reviewer='Оператор'):
        candidate=next((c for c in geometry['candidates'] if c['id']==candidate_id),None)
        if candidate is None or not candidate['eligible'] or not reviewer.strip(): raise ValueError('Подтвердить можно пригодный видимый кандидат с именем оператора.')
        self.selection={'source_id':geometry['source_id'],'generation':geometry['generation'],'shape':geometry['shape'],
            'mask':candidate['mask'].copy(),'count':len(geometry['candidates']),'last_sequence':geometry['sequence'],
            'model_sha256':geometry.get('model_sha256'),
            'reviewer':reviewer,'confirmed_sequence':geometry['sequence'],'confirmed_frame_sha256':geometry['frame_sha256'],'confirmed_signature':candidate['signature']}
        self.require_confirmation=False
    def evaluate(self,geometry,now,sequence,generation,source,max_age_ms=700,switch=False,health_assessed=True):
        g=geometry; chosen=None; reason=[]; status=g['status']
        age=(now-g['captured_at'])*1000
        if g['sequence']!=sequence or g['generation']!=generation or g['source_id']!=source or age<0 or age>max_age_ms or not health_assessed:
            self.reset(required=bool(self.selection) or self.require_confirmation); status='unknown'; reason=['геометрия другого/устаревшего кадра либо непригодное изображение']
        else:
            eligible=[c for c in g['candidates'] if c['eligible']]
            if self.selection:
                s=self.selection
                matches=sorted([(mask_iou(s['mask'],c['mask']),c) for c in eligible],key=lambda x:-x[0])
                compatible=s['source_id']==source and s['generation']==generation and s['shape']==g['shape'] and s['count']==len(g['candidates']) and sequence>=s['last_sequence'] and s['model_sha256']==g.get('model_sha256')
                if compatible and matches and matches[0][0]>=.7 and (len(matches)==1 or matches[0][0]-matches[1][0]>=.1):
                    chosen=matches[0][1]; s['mask']=chosen['mask'].copy(); s['last_sequence']=sequence; status='confirmed'
                else: self.reset(required=True); status='ambiguous'; reason=['выбор пути утрачен: изменились кандидаты/геометрия; подтвердите заново']
            elif len(eligible)==1 and not (switch or g.get('topology_ambiguous') or self.require_confirmation): chosen=eligible[0]; status='estimated'
            elif eligible and (switch or g.get('topology_ambiguous') or self.require_confirmation):
                self.require_confirmation=True; status='ambiguous'; reason=['стрелка/несколько путей либо утраченный выбор: нужен оператор']
        return {'status':status,'reasons':[] if chosen else reason or g.get('reasons',[]),'candidate_id':chosen['id'] if chosen else None,
            'quality_score':chosen['quality_score'] if chosen else None,'corridor_mask':chosen['mask'] if chosen else None,
            'semantic_masks':g.get('semantic_masks',{}),'frame_sha256':g.get('frame_sha256'),'sequence':sequence,'generation':generation,
            'source_id':source,'captured_at':g['captured_at'],'manual_confirmation':{k:v for k,v in self.selection.items() if k not in ('mask','shape')} if self.selection else None,
            'metric_status':'not_calibrated','metres_verified':False,'scores_are_probabilities':False}

def object_context(box,shape,geometry,instance_mask=None,calibration=None):
    h,w=shape[:2]; corridor=geometry.get('corridor_mask'); x1,y1,x2,y2=map(float,box); footprint=np.zeros((h,w),np.uint8)
    # Bottom contact band reduces false overlap of a tall object's top with another path.
    xa,xb=max(0,int(x1)),min(w,int(math.ceil(x2))); ya,yb=max(0,int(y2-max(2,(y2-y1)*.12))),min(h,int(math.ceil(y2)))
    if xa<xb and ya<yb: footprint[ya:yb,xa:xb]=1
    if instance_mask is not None:
        if np.asarray(instance_mask).shape!=(h,w): raise ValueError('Маска объекта должна иметь исходный размер.')
        footprint&=np.asarray(instance_mask,np.uint8)
    size=int(footprint.sum()); result={'location':'geometry_unknown','basis':'bottom contact band; approximate ground contact','overlap':None,
        'geometry_status':geometry['status'],'metric_distance_m':None,'metric_ground_xy_m':None,'metric_status':geometry.get('metric_status','not_calibrated'),'ballast_overlap':None}
    if corridor is not None and size:
        overlap=float((footprint.astype(bool)&corridor).sum()/size); distance=cv2.distanceTransform((~corridor).astype(np.uint8),cv2.DIST_L2,5)
        near=float(distance[footprint.astype(bool)].min())<=max(4,w*.015)
        result.update(location='inside' if overlap>=.5 else 'near' if near else 'outside',overlap=overlap)
        ballast=geometry.get('semantic_masks',{}).get('ballast')
        if ballast is not None: result['ballast_overlap']=float((footprint.astype(bool)&ballast).sum()/size)
    if calibration is not None and corridor is not None:
        metric=calibration.project([(x1+x2)/2,min(h-1,y2)],geometry['source_id'],[h,w]); result.update(metric)
    elif calibration is not None: result['metric_status']='geometry_unknown'
    return result

class GroundCalibration:
    def __init__(self,data):
        self.data=data; fit=np.asarray(data.get('fit_points',[]),float); check=np.asarray(data.get('check_points',[]),float)
        if fit.ndim!=2 or check.ndim!=2 or fit.shape[1]!=4 or check.shape[1]!=4 or len(fit)<4 or len(check)<3 or not np.isfinite(fit).all() or not np.isfinite(check).all(): raise ValueError('Нужны≥4 опорных и≥3 независимых проверочных точки [u,v,X,Y].')
        if any(np.linalg.norm(a[:2]-b[:2])<1e-6 for a in fit for b in check): raise ValueError('Проверочные точки не должны повторять опорные.')
        threshold=data.get('max_check_error_m',.1)
        if not isinstance(threshold,(float,int)) or not math.isfinite(threshold) or threshold<=0 or not data.get('reviewer') or not data.get('source_id'): raise ValueError('Нужны источник, проверяющий и конечный положительный допуск.')
        self.H,_=cv2.findHomography(fit[:,:2],fit[:,2:],0)
        if self.H is None or not np.isfinite(self.H).all() or np.linalg.cond(self.H)>1e8: raise ValueError('Вырожденная/неустойчивая калибровка.')
        xy=cv2.perspectiveTransform(check[:,:2].astype(np.float64)[None],self.H)[0]; errors=np.linalg.norm(xy-check[:,2:],axis=1)
        self.rms=float(np.sqrt(np.mean(errors**2))); self.max_error=float(errors.max()); self.valid=self.max_error<=threshold
        self.hull=cv2.convexHull(np.vstack([fit[:,:2],check[:,:2]]).astype(np.float32))
    def project(self,pixel,source,shape):
        if not self.valid: return {'metric_status':'calibration_check_failed'}
        if source!=self.data['source_id'] or list(shape)!=self.data.get('shape'): return {'metric_status':'calibration_source_or_resolution_mismatch'}
        if cv2.pointPolygonTest(self.hull,tuple(map(float,pixel)),False)<0: return {'metric_status':'outside_validated_image_region'}
        point=np.asarray(pixel,float); q=self.H@np.array([point[0],point[1],1.])
        if not np.isfinite(q).all() or abs(q[2])<1e-9: return {'metric_status':'projection_invalid'}
        return {'metric_status':'ground_plane_user_calibrated','metric_ground_xy_m':(q[:2]/q[2]).tolist(),'calibration_check_rms_m':self.rms,
            'calibration_check_max_error_m':self.max_error,'metric_distance_m':None,'metric_note':'Ground contact approximation, not object height/range safety certification.'}

def serializable_geometry(g):
    def clean(v):
        if isinstance(v,np.ndarray): return None
        if isinstance(v,dict): return {k:clean(x) for k,x in v.items() if not isinstance(x,np.ndarray) and k not in ('semantic_masks','corridor_mask')}
        if isinstance(v,list): return [clean(x) for x in v]
        return v
    return clean(g)
