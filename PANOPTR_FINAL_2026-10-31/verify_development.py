"""Repeatable offline regression suite after editing this portable source tree."""
from pathlib import Path
from datetime import datetime
import os,sys,subprocess,json
ROOT=Path(__file__).resolve().parent
APP=ROOT/'02_Реализация/panoptr'
def main():
    target=ROOT/'data/tests'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');target.mkdir(parents=True)
    temp=target/'tmp';temp.mkdir()
    env=os.environ.copy();env.update(QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TEMP=str(temp),TMP=str(temp),PYTHONUTF8='1')
    suites=[('application',APP,['test_camera_design','test_event_benchmark','test_final','test_path_understanding','test_route_memory','test_virtual_stand','test_finale','test_profile_compatibility','test_gpu_launch']),
            ('data_lab',APP/'dataset_lab',['test_annotation','test_lab'])]
    result=[]
    for name,cwd,modules in suites:
        print('Checking',name,'...',flush=True)
        with (target/(name+'.log')).open('w',encoding='utf-8') as log:
            code=subprocess.run([sys.executable,'-B','-m','unittest',*modules,'-v'],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        result.append({'suite':name,'passed':code==0,'log':name+'.log'})
        print('PASS' if code==0 else 'FAILED',name,flush=True)
    passed=all(x['passed'] for x in result)
    (target/'result.json').write_text(json.dumps({'passed':passed,'suites':result,'production_training_started':False},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Logs:',target,flush=True)
    return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
