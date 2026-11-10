"""Fetch optional models from a fixed commit; never overwrite local user weights."""
import hashlib, json, os, tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
APP = ROOT / '02_Реализация' / 'panoptr'

def git_blob_hash(path, size):
    h = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    data = json.loads((ROOT/'model_sources.json').read_text(encoding='utf-8'))
    failed=[]
    for rel, (sha, size) in data['files'].items():
        target = APP / rel
        if target.exists():
            if target.is_file() and target.stat().st_size == size and git_blob_hash(target, size) == sha:
                print('OK', rel, flush=True)
            else:
                print('Пропущен существующий файл (не перезаписываем):', rel, flush=True)
                failed.append(rel)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url='https://raw.githubusercontent.com/SnowMenchik/q/'+data['commit']+'/'+quote('PANOPTR_FINAL_2026-10-31/02_Реализация/panoptr/'+rel)
        tmp=None
        try:
            print('Загрузка:', rel, flush=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
                tmp=Path(f.name)
                with urlopen(Request(url, headers={'User-Agent':'PANOPTR-model-downloader'}),timeout=120) as response:
                    while block := response.read(1024*1024): f.write(block)
            if tmp.stat().st_size != size or git_blob_hash(tmp,size)!=sha:
                raise ValueError('Размер или Git blob hash не совпадает')
            # Avoid replacement if another process created the file during download.
            os.link(tmp, target)
        except Exception as exc:
            print('ОШИБКА:', rel, exc, flush=True); failed.append(rel)
        finally:
            if tmp: tmp.unlink(missing_ok=True)
    if failed:
        print('Не загружено:', ', '.join(failed)); return 1
    print('Все основные модели проверены.'); return 0
if __name__=='__main__': raise SystemExit(main())
