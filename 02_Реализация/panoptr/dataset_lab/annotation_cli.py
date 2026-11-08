from bootstrap import setup
setup()
import argparse,json,sys
from core import read
from cli import training_lock

def main(config):
    cfg=read(config); action=cfg['action']
    if action not in ('create','export','download','check','video','batch'): raise ValueError('Неизвестная операция.')
    from annotation import create_project,export_project
    if action=='create': result=create_project(cfg['images'],cfg['project'],cfg['classes'],cfg.get('group_regex',''),cfg.get('independent_images',False))
    elif action=='export': result=export_project(cfg['project'],cfg['output'],cfg['export_kind'],cfg.get('overlap','error'))
    elif action=='download':
        from annotation_downloads import download_public,download_clip
        result={'models':download_public(cfg['output']),'clip':download_clip(cfg['output'])}
    else:
        from annotation_models import preflight,run_batch,video_project
        if action=='check': result=preflight(cfg)
        else:
            with training_lock(cfg.get('device','cpu')):
                result=video_project(cfg,cfg['project'],cfg['classes'],int(cfg.get('limit',300)),int(cfg.get('every',1))) if action=='video' else run_batch(cfg['project'],cfg,int(cfg.get('limit',0)),cfg.get('frame_id'),cfg.get('prompt'))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('config'); args=p.parse_args()
    try: main(args.config)
    except Exception as exc: print('ОШИБКА: '+str(exc),file=sys.stderr,flush=True); sys.exit(1)
