from dataset_lab.bootstrap import setup
setup()
import sys,json
from pathlib import Path
from final_core import save_json
from final_tools import readiness,training_registry,duplicate_review,review_predictions,rehearse

def main(config):
    cfg=json.loads(Path(config).read_text(encoding='utf-8')); action=cfg['action']; app_root=Path(__file__).resolve().parent
    if action=='rehearse': result=rehearse(cfg,cfg['output'])
    elif action=='compare':
        from final_compare import compare
        result=compare(cfg,cfg['output'])
    elif action=='camera_design':
        from camera_design import calculate,export_report
        result=calculate(cfg['parameters']); export_report(result,cfg['output'])
    elif action.startswith('event_'):
        from event_benchmark import infer,score,compare_reports,audit_sources
        if action=='event_infer': result=infer(cfg['frozen'],cfg['model'],cfg['output'])
        elif action=='event_score': result=score(cfg['frozen'],cfg['predictions'],cfg['output'])
        elif action=='event_compare': result=compare_reports(cfg['first'],cfg['second'],cfg['output'])
        elif action=='event_audit': result=audit_sources(cfg['project'],cfg['roots'],cfg['output'],int(cfg.get('limit',1000)))
        else: raise ValueError('Неизвестная операция оценки событий.')
    else:
        output=Path(cfg['output'])
        if output.exists(): raise ValueError('Выберите новый файл отчёта.')
        if action=='ready': result=readiness(app_root,cfg.get('models'))
        elif action=='registry': result=training_registry(app_root.parents[1])
        elif action=='duplicates': result=duplicate_review(cfg['train_images'],cfg['val_images'],int(cfg.get('limit',500)),int(cfg.get('threshold',6)))
        elif action=='review': result=review_predictions(cfg['predictions'],cfg.get('ground_truth'),cfg.get('second'))
        else: raise ValueError('Неизвестная операция.')
        save_json(output,result)
    count=len(result.get('queue',result.get('candidates',result.get('runs',result.get('scenarios',[])))))
    print(json.dumps({'action':action,'output':cfg['output'],'records':count,'ready':result.get('ready'),'execution_pass':result.get('execution_pass'),
        'note':'Full result saved in report; datasets and training not modified.'},ensure_ascii=False,indent=2),flush=True); return result

if __name__=='__main__':
    try:
        result=main(sys.argv[1])
        if result.get('ready',True) is False or result.get('execution_pass',True) is False: sys.exit(2)
    except Exception as exc: print('ОШИБКА: '+str(exc),file=sys.stderr,flush=True); sys.exit(1)
