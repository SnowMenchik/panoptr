"""Check the active environment without modifying it or requiring optional assets."""
from pathlib import Path
import importlib, json, multiprocessing as mp, sys
ROOT=Path(__file__).resolve().parents[2]
APP=Path(__file__).resolve().parent

def spawn_echo(queue):
    queue.put(sys.executable)

def main():
    mp.freeze_support()
    required=['numpy','PIL','cv2','torch','torchvision','ultralytics','PySide6.QtWidgets','psutil','yaml','scipy','matplotlib','lap']
    if sys.platform == 'win32': required += ['pygrabber','comtypes']
    modules={}; errors=[]
    for name in required:
        try:
            mod=importlib.import_module(name)
            if name=='cv2' and not hasattr(mod,'VideoCapture'): raise RuntimeError('VideoCapture missing')
            modules[name]=str(getattr(mod,'__file__','built-in'))
        except Exception as exc:
            errors.append(f'{name}: {exc}')
    model_names=['rail9_finetuned_2026.pt','drone_broad10_interim_2026.pt']
    missing=[str(APP/'models'/name) for name in model_names if not (APP/'models'/name).is_file()]
    if missing: errors.append('Не найдены основные веса: запустите 00_Скачать_модели.bat')
    worker_ok=False
    try:
        ctx=mp.get_context('spawn'); queue=ctx.Queue(); process=ctx.Process(target=spawn_echo,args=(queue,)); process.start()
        actual=queue.get(timeout=20); process.join(20)
        if process.is_alive(): process.terminate(); process.join()
        worker_ok=process.exitcode==0 and Path(actual).resolve()==Path(sys.executable).resolve()
        if not worker_ok: errors.append('Ошибка дочернего процесса Python')
        queue.close()
    except Exception as exc: errors.append(f'Дочерний процесс: {exc}')
    cuda={'available':False}
    try:
        import torch
        cuda['available']=bool(torch.cuda.is_available())
        if cuda['available']: cuda['device']=torch.cuda.get_device_name(0)
    except Exception: pass
    if '--integrity' in sys.argv: errors.append('Проверка manifest не поддерживается в этой сборке: manifest.json отсутствует.')
    if '--models' in sys.argv: errors.append('Инференс моделей не проверен этой командой; запустите отдельный тест с реальным видео.')
    report={'passed':not errors,'python':sys.version,'executable':sys.executable,'modules':modules,'worker_ok':worker_ok,'cuda':cuda,'errors':errors,'scope':'Импорт библиотек и наличие двух основных весов; камеры и инференс не проверяются.'}
    (ROOT/'data').mkdir(exist_ok=True)
    (ROOT/'data/last_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0 if not errors else 1
if __name__=='__main__': raise SystemExit(main())
