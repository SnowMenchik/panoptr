"""Export frozen detections for manual GT review. Does not infer or mark GT reviewed.

python export_path_tracking_template.py telemetry.jsonl output.json CHANNEL GENERATION
"""
import json,sys
from pathlib import Path

def make_template(records,channel,generation):
    rows=[r for r in records if r.get('kind')=='path_context' and r.get('channel')==channel and r.get('generation')==generation]
    if not rows: raise ValueError('Нет записей path_context выбранного канала/поколения.')
    def signature(r): return json.dumps([r['source_ref'],r['shape'],r['names'],r['object_model_sha256']],sort_keys=True,ensure_ascii=False)
    if len({signature(r) for r in rows})!=1: raise ValueError('Источник, разрешение или модель изменились: разделите записи.')
    rows.sort(key=lambda r:r['sequence'])
    file_time=all(r.get('source_time_s') is not None for r in rows)
    times=[float(r['source_time_s'] if file_time else r['captured_at']) for r in rows]
    if any(b<=a for a,b in zip(times,times[1:])): raise ValueError('Повторные кадры или нестрогое время.')
    frames=[{'time_s':t-times[0],'reviewed':False,'ground_truth':[], 'detections':r['boxes'],
        'reference':{'sequence':r['sequence'],'frame_sha256':r['frame_sha256'],'source_time_s':r.get('source_time_s'),
            'captured_at':r['captured_at'],'image_assessed':r['image_assessed']}} for t,r in zip(times,rows)]
    return {'scope':'real_frozen_detections_pending_independent_manual_GT','shape':rows[0]['shape'],'names':rows[0]['names'],
        'source_ref':rows[0]['source_ref'],'object_model_sha256':rows[0]['object_model_sha256'],'frames':frames,
        'time_basis':'source_media_time' if file_time else 'capture_monotonic_time','match_iou':.3,'max_gap_s':1.,
        'instructions':'Review every frame against original recording; add persistent GT id, class_name, box, then reviewed=true. Unusable images need separate interval annotation, not empty GT. Empty ground_truth with reviewed=false is unlabelled, not negative.'}

if __name__=='__main__':
    source,destination=map(Path,sys.argv[1:3]); output=make_template([json.loads(s) for s in source.read_text(encoding='utf-8').splitlines() if s.strip()],int(sys.argv[3]),int(sys.argv[4]))
    with destination.open('x',encoding='utf-8') as f: json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(str(destination))
