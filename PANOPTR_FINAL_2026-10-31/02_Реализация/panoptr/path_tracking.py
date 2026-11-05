"""Same detections for legacy IoU and installed Ultralytics ByteTrack.

Class-isolated, channel/generation scoped; predicted boxes are never evidence.
"""
from types import SimpleNamespace
import numpy as np
from final_core import iou,EventAssociator,validate_policy
from path_geometry import object_context,serializable_geometry

class TrackingBackend:
    def __init__(self,mode='iou',gap_s=1.,fps=20,min_iou=.2):
        if mode not in ('iou','bytetrack') or not np.isfinite([gap_s,fps,min_iou]).all() or gap_s<=0 or fps<=0 or not 0<=min_iou<=1: raise ValueError('Неверный режим сопровождения.')
        self.mode=mode; self.gap_s=gap_s; self.fps=fps; self.min_iou=min_iou; self.streams={}; self.count=0; self.lost_episodes=0; self.refound_episodes=0
    def reset(self,channel=None):
        if channel is None: self.streams={}
        else: self.streams={k:v for k,v in self.streams.items() if k[0]!=channel}
    def update(self,channel,generation,sequence,stamp,boxes,shape):
        key=(channel,generation); state=self.streams.setdefault(key,{'last_seq':-1,'stamp':None,'tracks':{},'byte':{},'visible':set(),'lost':set()})
        if sequence<=state['last_seq']: return [None]*len(boxes)
        if state['stamp'] is not None and (stamp<=state['stamp'] or stamp-state['stamp']>self.gap_s):
            self.reset(channel); return self.update(channel,generation,sequence,stamp,boxes,shape)
        rows=np.asarray(boxes,float).reshape(-1,6); ids=[None]*len(rows)
        valid=[i for i,r in enumerate(rows) if np.isfinite(r).all() and 0<=r[4]<=1 and r[2]>r[0] and r[3]>r[1] and r[5]>=0 and r[5]==int(r[5])]
        if self.mode=='iou':
            state['tracks']={k:v for k,v in state['tracks'].items() if 0<=stamp-v['stamp']<=self.gap_s}
            possible=[]
            for index in valid:
                for identifier,t in state['tracks'].items():
                    score=iou(rows[index,:4],t['box'])
                    if rows[index,5]==t['cls'] and score>=self.min_iou: possible.append((score,index,identifier))
            used=set()
            for score,index,identifier in sorted(possible,reverse=True):
                if ids[index] is None and identifier not in used: ids[index]=identifier; used.add(identifier)
            for index in valid:
                if ids[index] is None: self.count+=1; ids[index]=f'iou:{self.count}'
                state['tracks'][ids[index]]={'box':rows[index,:4].tolist(),'cls':rows[index,5],'stamp':stamp}
        else:
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.engine.results import Boxes
            # Upstream constructors reset a global ID counter. Prevent collisions
            # when another class/channel creates its own tracker.
            class ScopedBYTE(BYTETracker):
                @staticmethod
                def reset_id(): pass
            classes=set(state['byte'])|{int(rows[i,5]) for i in valid}
            for cls in classes:
                if cls not in state['byte']:
                    args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,
                        track_buffer=max(1,int(round(self.fps*self.gap_s))),match_thresh=.8,fuse_score=True)
                    state['byte'][cls]=ScopedBYTE(args)
                tracker=state['byte'][cls]; indices=[i for i in valid if int(rows[i,5])==cls]; selected=rows[indices].astype(np.float32).reshape(-1,6)
                for track in tracker.update(Boxes(selected,shape[:2])):
                    local=int(track[-1])
                    if not 0<=local<len(indices): raise ValueError('ByteTrack вернул неверный индекс детекции.')
                    ids[indices[local]]=f'byte:{cls}:{int(track[4])}'
        present={i for i in ids if i is not None}; newly_lost=state['visible']-present; self.lost_episodes+=len(newly_lost)
        self.refound_episodes+=len(present&state['lost']); state['lost']|=newly_lost; state['lost']-=present
        state.update(last_seq=sequence,stamp=stamp,visible=present); return ids

class PathEventPipeline:
    def __init__(self,mode='iou',gap_s=1.,min_iou=.2):
        self.backend=TrackingBackend(mode,gap_s,min_iou=min_iou); self.events=EventAssociator(); self.context=[]; self.suppressed_outside=0
    def reset(self,channel): self.backend.reset(channel); self.events.reset(channel)
    def update(self,channel,generation,sequence,stamp,boxes,names,shape,policy,health,masks=None,geometry=None,calibration=None):
        p=validate_policy(policy); ids=self.backend.update(channel,generation,sequence,stamp,boxes,shape); locations=[]; event_rows=[]; event_ids=[]; event_masks=[]; self.context=[]
        for i,row in enumerate(boxes):
            if len(row)<6 or not np.isfinite(row[:6]).all() or row[2]<=row[0] or row[3]<=row[1]: continue
            name=str(names.get(int(row[5]),'')); role=p['roles'].get(name,'ignore'); mask=masks[i] if masks is not None and i<len(masks) else None
            if role=='overhead_object': where={'location':'not_applicable_overhead','basis':'contact network object; no ground corridor or ground-plane metres','metric_status':'not_applicable_overhead'}
            elif role=='aerial_object': where={'location':'not_applicable_aerial','basis':'airborne candidate; no ground corridor or ground-plane metres','metric_status':'not_applicable_aerial'}
            elif geometry is None:
                from final_core import zone_location
                where=zone_location(row[:4],shape,p,mask); where['geometry_status']='manual_zone' if p['zone'] else 'unknown'
            else: where=object_context(row[:4],shape,geometry,mask,calibration)
            risk='diagnostic_candidate' if role=='defect' else 'path_candidate' if where['location']=='inside' else 'near_path_review' if where['location']=='near' else 'outside_context' if where['location']=='outside' else 'geometry_unknown_review'
            if name=='ballast': risk='background_ballast'
            elif role=='overhead_object': risk='overhead_review'
            elif role=='aerial_object': risk='aerial_review'
            elif role=='obstacle' and where['location']=='inside' and (where.get('ballast_overlap') or 0)>=.6: risk='ballast_overlap_review'
            self.context.append({'index':i,'class_name':name,'track_id':ids[i],'zone':where,'risk':risk,'automatic_collision_alarm':False})
            if name=='ballast': continue
            # Outside stones remain auditable context; never a path alarm/card by class alone.
            if role=='obstacle' and geometry is not None and where['location']=='outside': self.suppressed_outside+=1; continue
            if role not in ('person','obstacle','defect','transient','overhead_object','aerial_object'): continue
            event_rows.append(row); event_ids.append(ids[i]); event_masks.append(mask); locations.append({**where,'risk':risk})
        updates=self.events.update(channel,generation,sequence,stamp,event_rows,names,shape,p,health,None,event_ids,locations)
        for event in updates:
            event['tracker']=self.backend.mode; event['automatic_collision_alarm']=False
            event['geometry']=serializable_geometry(geometry) if geometry else {'status':'manual_zone' if p['zone'] else 'unknown','metric_status':'not_calibrated'}
            if event['role']=='aerial_object':event['geometry']={'status':'not_applicable_aerial','metric_status':'not_applicable_aerial'}
            event['context_state']=event['zone']['risk']; self.events.events[event['event_id']]=event
        return updates
