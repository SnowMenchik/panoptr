"""Frozen-detection tracker comparison. Requires explicit reviewed visible GT IDs."""
from pathlib import Path
import json,csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from path_tracking import TrackingBackend
from final_core import iou,save_json,EventAssociator

def compare(payload,output=None):
    frames=payload['frames']; times=[float(f['time_s']) for f in frames]
    if not frames or not all(f.get('reviewed') is True for f in frames) or any(not np.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])): raise ValueError('Нужны проверенные GT для каждого кадра и строго возрастающее время.')
    per_mode=[]; mapping=payload['names']; mapping={int(k):v for k,v in mapping.items()}; shape=payload['shape']
    duration=max(.001,times[-1]-times[0]+(float(np.median(np.diff(times))) if len(times)>1 else .05)); rows=[]
    for mode in ('iou','bytetrack'):
        tracker=TrackingBackend(mode,payload.get('max_gap_s',1)); assoc=EventAssociator(); per_gt={}; all_ids=set(); matched_ids=set(); missed=0; switches=0; interruptions=0; previous={}; misses_by_gt=set(); unmatched_observations=0; matched_cards=set()
        for sequence,f in enumerate(frames,1):
            boxes=np.asarray(f['detections'],float).reshape(-1,6)
            if any(not np.isfinite(r).all() or not 0<=r[4]<=1 or r[2]<=r[0] or r[3]<=r[1] or r[5]!=int(r[5]) or int(r[5]) not in mapping for r in boxes): raise ValueError('Некорректные замороженные детекции.')
            ids=tracker.update(0,1,sequence,f['time_s'],boxes,shape); visible=f['ground_truth']; indexes=[i for i,v in enumerate(ids) if v is not None]
            if len({gt['id'] for gt in visible})!=len(visible): raise ValueError('ПовторGT ID в одном кадре.')
            for gt in visible:
                b=np.asarray(gt.get('box',[]),float)
                if b.shape!=(4,) or not np.isfinite(b).all() or not 0<=b[0]<b[2]<=shape[1] or not 0<=b[1]<b[3]<=shape[0]: raise ValueError('Некорректная GT рамка.')
            cards=assoc.update(0,1,sequence,f['time_s'],boxes,mapping,shape,{}, {'assessed':True},track_ids=ids)
            card_by_track={e['external_track_id']:e['event_id'] for e in cards}
            all_ids.update(ids[i] for i in indexes); costs=np.ones((len(visible),len(indexes)),float)
            for a,gt in enumerate(visible):
                if not gt.get('id') or gt.get('class_name') not in mapping.values(): raise ValueError('GT: нужны устойчивый id и известный класс.')
                for b,index in enumerate(indexes):
                    if mapping[int(boxes[index,5])]==gt['class_name']: costs[a,b]=1-iou(gt['box'],boxes[index,:4])
            matches={}; used=set()
            if costs.size:
                aa,bb=linear_sum_assignment(costs)
                for a,b in zip(aa,bb):
                    if costs[a,b]<=1-payload.get('match_iou',.3): matches[int(a)]=indexes[int(b)]; used.add(indexes[int(b)])
            unmatched_observations+=len(indexes)-len(used)
            for a,gt in enumerate(visible):
                identifier=gt['id']; data=per_gt.setdefault(identifier,{'tracks':set(),'cards':set(),'first_gt':f['time_s'],'first_hit':None,'visible':0,'misses':0}); data['visible']+=1
                if a not in matches:
                    missed+=1; data['misses']+=1; misses_by_gt.add(identifier)
                    if previous.get(identifier,{}).get('visible_hit'): interruptions+=1
                    old=previous.get(identifier,{}); previous[identifier]={**old,'visible_hit':False}; observed=None
                else:
                    observed=ids[matches[a]]; data['tracks'].add(observed); matched_ids.add(observed)
                    card=card_by_track.get(observed)
                    if card is not None: data['cards'].add(card); matched_cards.add(card)
                    if data['first_hit'] is None: data['first_hit']=f['time_s']
                    old=previous.get(identifier,{})
                    if old.get('track_id') is not None and old['track_id']!=observed: switches+=1
                    previous[identifier]={'track_id':observed,'visible_hit':True}
                rows.append({'mode':mode,'time_s':f['time_s'],'gt_id':identifier,'track_id':observed,'hit':observed is not None})
        per_mode.append({'mode':mode,'gt_objects':len(per_gt),'objects_with_any_visible_miss':len(misses_by_gt),'objects_never_tracked':sum(not v['tracks'] for v in per_gt.values()),
            'missed_visible_observations':missed,'visible_observations':sum(v['visible'] for v in per_gt.values()),'tracking_interruptions':interruptions,'identity_switches':switches,
            'repeat_event_candidates':sum(max(0,len(v['tracks'])-1) for v in per_gt.values()),'false_track_candidates':len(all_ids-matched_ids),
            'event_cards_without_path_filter':len(assoc.events),'repeat_event_cards':sum(max(0,len(v['cards'])-1) for v in per_gt.values()),'false_event_cards':len(set(assoc.events)-matched_cards),
            'false_track_candidates_per_minute':len(all_ids-matched_ids)/(duration/60),'unmatched_tracked_observations':unmatched_observations,
            'mean_first_track_delay_s':float(np.mean([v['first_hit']-v['first_gt'] for v in per_gt.values() if v['first_hit'] is not None])) if any(v['first_hit'] is not None for v in per_gt.values()) else None,
            'backend_lost_episodes':tracker.lost_episodes,'backend_refound_episodes':tracker.refound_episodes})
    report={'format':'panoptr_tracker_comparison_v1','scope':payload.get('scope','user_supplied_reviewed_GT'),'same_input_detections':True,
        'duration_s':duration,'metrics':per_mode,'rows':rows,'GPU_used':False,
        'definitions':'Repeat candidates=sum(max(unique matched track IDs per GT-1,0)). Event cards are actual EventAssociator cards on the same detections, default policy, path filter disabled; repeat cards=count of excess matched event IDs per GT, false cards=cards never matched with GT. These are candidates requiring review, not operator-confirmed alarms. Visible misses include detector misses and tracker withholding. Backend lost episodes are disappearance of previously emitted IDs, not independent GT loss. First delay is first tracked detection, not end-to-end operator latency.',
        'limits':'CPU association test, no detector inference. Synthetic tests do not establish real accuracy. Identity switches, false tracks and fragmentation trade off; ByteTrack is not declared universally better.',
        'sources':['https://docs.ultralytics.com/modes/track/','https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py']}
    if output:
        out=Path(output); out.mkdir(parents=True,exist_ok=False); save_json(out/'comparison.json',report)
        with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(per_mode[0])); writer.writeheader(); writer.writerows(per_mode)
        table=['# IoU и ByteTrack: одинаковые детекции','Статус: '+report['scope'],'','| Показатель | IoU | ByteTrack |','|---|---:|---:|']
        table += [f"| {k} | {per_mode[0][k]} | {per_mode[1][k]} |" for k in per_mode[0] if k!='mode']; table += ['',report['definitions'],report['limits']]
        (out/'comparison.md').write_text('\n'.join(table),encoding='utf-8')
    return report

def synthetic_tracking_fixture():
    frames=[]
    for i in range(32):
        x=20+5*i if i<6 else 50+12*(i-6); box=[x,100,x+25,145]; hidden=i in (13,14); score=.12 if i in (9,10) else .9
        det=[] if hidden else [box+[score,0]]
        frames.append({'time_s':i*.05,'reviewed':True,'detections':det,'ground_truth':[] if hidden else [{'id':'person_a','class_name':'person','box':box}]})
    return {'shape':[240,640],'names':{0:'person'},'frames':frames,'scope':'synthetic_software_QA_only','max_gap_s':1.,'match_iou':.3}
