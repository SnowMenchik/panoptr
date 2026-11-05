"""Write evidence-based release status; fail if any required check did not pass."""
from bootstrap import setup
setup()
from pathlib import Path
import ast,json,re,shutil
from core import read,dump,digest
from annotation_models import MODES

def newest(work,pattern,filename):
    choices=sorted(p for p in work.glob(pattern) if (p/filename).is_file())
    if not choices: raise AssertionError('No report: '+filename)
    return choices[-1]/filename

def run():
    module=Path(__file__).resolve().parent; work=module.parents[2]/'10_Лаборатория_данных'
    for path in module.glob('*.py'): ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    log=(work/'all_unit_tests.log').read_text(encoding='utf-8'); assert re.search(r'Ran 22 tests.*\n\s*\nOK',log,re.S),log[-1000:]
    models_file=newest(work,'QA_annotation_models_*','models_check.json'); model_results=read(models_file)
    assert all(r['actual_cpu_inference_pass'] for r in model_results)
    extra_file=newest(work,'QA_annotation_extra_*','extra_check.json'); extras=read(extra_file); assert all(r['actual_cpu_inference_pass'] for r in extras)
    video_file=newest(work,'QA_annotation_video_*','video_check.json'); assert read(video_file)['actual_cpu_inference_pass']
    gui_file=newest(work,'QA_annotation_gui_*','gui_check.json'); assert read(gui_file)['pass']
    segment_file=newest(work,'QA_segment_*','segment_check.json'); assert read(segment_file)['pass']
    actual={r['mode'] for r in model_results}|{'semantic_model','sam2_video'}
    capabilities=[]
    for mode,title in MODES.items():
        capabilities.append({'mode':mode,'title':title,'implemented':True,'actual_cpu_inference_pass':mode in actual,
            'status':'tested_available_weights' if mode in actual else 'blocked_missing_official_gated_weights',
            'quality_validated_on_railway_data':False,
            'note':'Actual inference applies only to the specific weights in QA reports, not every model family/checkpoint.'})
    assert sum(r['actual_cpu_inference_pass'] for r in capabilities)==14
    dump(work/'annotation_capabilities.json',{'ultralytics_version':'8.4.115','modes':capabilities,'reports':[str(models_file),str(extra_file),str(video_file)],
        'untested_variations':['RT-DETR adapter','original SAM 1 checkpoint','SAM 3/SAM 3.1 checkpoints'],
        'sam3_weights_documentation':'https://docs.ultralytics.com/models/sam-3/'})
    previous=read(work/'verification.json'); previous.setdefault('previous_base_verification',dict(previous))
    sha=digest(module.parent/'models/person_security_yolo26n.pt'); assert sha=='f4d5173fff4d2e4c87ec1542282f1a98fbb9338d34ef8be291d7e39a2b7dbb33'
    previous.update(date='2026-10-12',pass_check=True,unit_tests_passed=22,python_syntax_pass=True,
        annotation_gui_report=str(gui_file),annotation_models_report=str(models_file),annotation_extra_report=str(extra_file),
        annotation_video_report=str(video_file),yolo_segment_cpu_report=str(segment_file),
        annotation_modes_implemented=18,annotation_modes_actual_cpu_inference_pass=14,
        annotation_modes_blocked_by_missing_weights=['sam3_text','sam3_exemplar','sam3_video','sam3_video_text'],
        test_data='Generated geometric fixtures and Ultralytics packaged bus.jpg; no real railway training started',
        gpu_training_started_by_agent=False,real_dataset_training_started_by_agent=False,
        existing_application_person_weights_unchanged=True,person_sha256=sha,
        limitations=['SAM 3/3.1 adapters need official access and weights; no actual inference verification yet',
            'RT-DETR/original SAM 1 adapters not exercised with separate weights',
            'Small UNet baseline, not pretrained railway semantic weights',
            'COCO import trains bbox only; RLE instance masks preserved in annotation exports for external backend',
            'Repeated val is not independent test; human-confirmed pseudo-labels do not prove quality',
            'Universal export may need the final submission format adapter'])
    previous['pass']=True; dump(work/'verification.json',previous)
    # Reusable checked editor demo with actual source masks. Do not include model binaries in the zip.
    demo=work/'DEMO_Авторазметка'; demo.mkdir(exist_ok=True)
    dump(demo/'demo.json',{'project':str(gui_file.parent/'project'),'semantic_export':str(gui_file.parent/'semantic_export'),
        'prepared_dataset':str(gui_file.parent/'prepared'),'gpu_used':False,'not_railway_training_data':True})
    (demo/'Открыть_проект.txt').write_text('Вкладка 4 → Открыть проект → '+str(gui_file.parent/'project')+'\nТолько проверочный пример интерфейса.\n',encoding='utf-8')
    import importlib.metadata
    deps=[]
    for name in ('clip','einops'):
        dist=importlib.metadata.distribution(name); direct=dist.read_text('direct_url.json')
        deps.append({'name':name,'version':dist.version,'location':str(dist.locate_file('')),'direct_url':json.loads(direct) if direct else None})
    dump(work/'annotation_dependency_provenance.json',deps)
    print(json.dumps({'pass':True,'tests':22,'actual_modes':14,'gated_modes':4,'gui_report':str(gui_file)},ensure_ascii=False,indent=2))

if __name__=='__main__': run()
