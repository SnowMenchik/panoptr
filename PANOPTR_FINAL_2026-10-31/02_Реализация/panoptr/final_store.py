"""Bounded asynchronous evidence storage; disk failures never block acquisition."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,queue,threading,time,copy,zipfile
import cv2
from PIL import Image
from final_core import save_json

class EventStore:
    def __init__(self,root,max_queue=24):
        self.root=Path(root).resolve(); self.root.mkdir(parents=True,exist_ok=False)
        self.queue=queue.Queue(maxsize=max_queue); self.error=''; self.dropped=0; self.decisions={}; self._lock=threading.Lock(); self._closed=False
        self.thread=threading.Thread(target=self._worker,name='evidence-writer',daemon=True); self.thread.start()
        save_json(self.root/'session.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'model_outputs_are_not_ground_truth':True,'format':'panoptr_events_v1'})
    def submit(self,event,frames,metadata):
        if self._closed: return False
        identifier=event.get('event_id','')
        if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier): raise ValueError('Неверный ID события.')
        item=(copy.deepcopy(event),[(float(t),image.copy()) for t,image in frames],copy.deepcopy(metadata))
        try: self.queue.put_nowait(item); return True
        except queue.Full: self.dropped+=1; self.error='Очередь записи заполнена: доказательство не сохранено.'; return False
    def _worker(self):
        while True:
            try: item=self.queue.get(timeout=.2)
            except queue.Empty:
                if self._closed: return
                continue
            try:
                if item is None: return
                if isinstance(item[0],str) and item[0]=='telemetry':
                    with (self.root/'telemetry.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(item[1],ensure_ascii=False,allow_nan=False)+'\n')
                    continue
                event,frames,metadata=item; folder=self.root/'events'/event['event_id']; folder.mkdir(parents=True,exist_ok=True)
                event['provenance']=metadata
                # Persist once: repeated detections update metadata without overwriting the original evidence.
                if frames and not (folder/'evidence.png').exists():
                    temporary=folder/'evidence.tmp.png'
                    Image.fromarray(cv2.cvtColor(frames[-1][1],cv2.COLOR_BGR2RGB)).save(temporary); temporary.replace(folder/'evidence.png')
                    clip_frames=[(t,f) for t,f in frames if f.shape==frames[-1][1].shape]
                    if len(clip_frames)>=2:
                        duration=clip_frames[-1][0]-clip_frames[0][0]; fps=(len(clip_frames)-1)/duration if duration>0 else 5.
                        h,w=clip_frames[-1][1].shape[:2]; writer=cv2.VideoWriter(str(folder/'context.avi'),cv2.VideoWriter_fourcc(*'MJPG'),max(1.,min(60.,fps)),(w,h))
                        if not writer.isOpened(): raise IOError('AVI codec unavailable')
                        try:
                            for _,f in clip_frames: writer.write(f)
                        finally: writer.release()
                        save_json(folder/'clip_frames.json',{'timestamps_monotonic':[t for t,_ in clip_frames],
                            'constant_fps_preview':True,'exact_timing':'see timestamp map; AVI spacing is approximate','fps':fps})
                with self._lock:
                    old=json.loads((folder/'event.json').read_text(encoding='utf-8')) if (folder/'event.json').exists() else {}
                    event['evidence_observation']=old.get('evidence_observation',{'box':event['box'],'sequence':event['sequence'],'quality':event['quality'],'provenance':metadata})
                    event['evidence_ready_at_monotonic']=old.get('evidence_ready_at_monotonic',time.monotonic())
                    if event['event_id'] in self.decisions: event.update(self.decisions[event['event_id']])
                    event['evidence_saved']=(folder/'evidence.png').exists(); save_json(folder/'event.json',event)
                with (self.root/'observations.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
            except Exception as exc: self.error='Ошибка записи: '+str(exc)
            finally: self.queue.task_done()
    def events(self):
        rows=[]
        for path in (self.root/'events').glob('*/event.json'):
            try:
                row=json.loads(path.read_text(encoding='utf-8')); review=path.parent/'technical_review.json'
                if review.is_file():
                    try:row['technical_review']=json.loads(review.read_text(encoding='utf-8'))
                    except (OSError,ValueError):row['technical_review_error']='Не удалось прочитать отдельную запись обследования.'
                rows.append(row)
            except (OSError,ValueError): pass
        return sorted(rows,key=lambda r:r['first_seen'])
    def save_review(self,identifier,values):
        from rail_reference import validate_review
        if not isinstance(identifier,str) or len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):raise ValueError('Неверный event_id.')
        folder=self.root/'events'/identifier
        with self._lock:
            row=json.loads((folder/'event.json').read_text(encoding='utf-8'))
            review=validate_review(row['class_name'],values);review['recorded_utc']=datetime.now(timezone.utc).isoformat()
            save_json(folder/'technical_review.json',review)
        return review
    def telemetry(self,row):
        if self._closed: return False
        try: self.queue.put_nowait(('telemetry',copy.deepcopy(row))); return True
        except queue.Full: self.dropped+=1; self.error='Очередь записи заполнена; часть телеметрии не сохранена.'; return False
    def decide(self,identifier,status,note=''):
        if status not in ('accepted','rejected','closed'): raise ValueError('Неизвестное решение оператора.')
        if not identifier or len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier): raise ValueError('Неверный event_id.')
        path=self.root/'events'/identifier/'event.json'
        with self._lock:
            row=json.loads(path.read_text(encoding='utf-8')); decision={'operator_status':status,'operator_note':note,'decision_utc':datetime.now(timezone.utc).isoformat()}
            self.decisions[identifier]=decision; row.update(decision); save_json(path,row)
            with (self.root/'decisions.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps({'event_id':identifier,**decision},ensure_ascii=False)+'\n')
    def flush(self): self.queue.join()
    def export(self,output):
        self.flush(); output=Path(output).resolve()
        if output.exists() or Path(str(output)+'.zip').exists() or output.is_relative_to(self.root): raise ValueError('Для отчёта нужна новая папка вне журнала.')
        output.mkdir(parents=True); rows=self.events(); save_json(output/'events.json',rows)
        fields=['event_id','channel','class_name','role','observation_state','operator_status','hits','peak_confidence','first_seen','last_seen','evidence_saved']
        with (output/'events.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
        telemetry=[]
        if (self.root/'telemetry.jsonl').exists():
            for line in (self.root/'telemetry.jsonl').read_text(encoding='utf-8').splitlines():
                try: telemetry.append(json.loads(line))
                except ValueError: pass
        import numpy as np
        latencies=[r['end_to_end_ms'] for r in telemetry if r.get('kind')=='accepted_result']
        evidence_latencies=[max(0.,(r['evidence_ready_at_monotonic']-r['evidence_observation']['provenance']['captured_at_monotonic'])*1000) for r in rows if r.get('evidence_saved') and 'captured_at_monotonic' in r.get('evidence_observation',{}).get('provenance',{})]
        save_json(output/'telemetry.json',telemetry)
        save_json(output/'summary.json',{'events':len(rows),'operator_accepted':sum(r['operator_status']=='accepted' for r in rows),
            'operator_rejected':sum(r['operator_status']=='rejected' for r in rows),'storage_error':self.error,'dropped_evidence_jobs':self.dropped,
            'accepted_result_p95_end_to_end_ms':float(np.percentile(latencies,95)) if latencies else None,
            'delivered_frame_to_evidence_ready_p95_ms':float(np.percentile(evidence_latencies,95)) if evidence_latencies else None,
            'latency_definitions':{'end_to_end_ms':'After VideoCapture.read delivery in host process to result callback; before evidence I/O. Not physical camera exposure-to-card latency.',
                'evidence_ready':'Host delivered frame to image/clip preparation before publishing event metadata. GUI display and operator delay are not included.'},
            'rejected_stale_or_mismatched_results':sum(r.get('kind')=='rejected_result' for r in telemetry),
            'recall':None,'false_positives_per_minute':None,'reason':'Independent ground truth and observation duration are required for quality metrics.'})
        with zipfile.ZipFile(str(output)+'.zip','w',zipfile.ZIP_DEFLATED) as z:
            for path in output.iterdir(): z.write(path,path.name)
            for path in (self.root/'events').rglob('*'):
                if path.is_file() and '.tmp' not in path.name: z.write(path,'evidence/'+path.relative_to(self.root/'events').as_posix())
        return output
    def close(self):
        if not self._closed:
            self._closed=True
            try: self.queue.put_nowait(None)
            except queue.Full: pass
            self.thread.join(timeout=3)
            if self.thread.is_alive(): self.error='Запись ещё не завершилась; журнал сохранён частично.'
            try: save_json(self.root/'status.json',{'closed':not self.thread.is_alive(),'storage_error':self.error,'dropped_evidence_jobs':self.dropped})
            except OSError: pass

class ArchiveStore(EventStore):
    def __init__(self,root):
        self.root=Path(root).resolve()
        if not (self.root/'session.json').is_file(): raise ValueError('Выберите папку с session.json.')
        self.queue=queue.Queue(); self._lock=threading.Lock(); self._closed=True; self.error=''; self.dropped=0; self.decisions={}
        status=json.loads((self.root/'status.json').read_text(encoding='utf-8')) if (self.root/'status.json').exists() else {}
        self.error=status.get('storage_error','') or ('Журнал ещё открыт или завершение сессии не зафиксировано.' if not status.get('closed') else '')
        self.dropped=status.get('dropped_evidence_jobs',0)
    def close(self): pass
