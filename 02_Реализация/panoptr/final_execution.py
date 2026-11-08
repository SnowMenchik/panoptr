"""Same Windows lock as the user's 08/09 training queues; no implicit installation."""
from contextlib import contextmanager
from pathlib import Path
import os

@contextmanager
def gpu_guard(device,lock_path=None):
    if str(device) in ('cpu','CPU'): yield; return
    path=Path(lock_path) if lock_path else Path(__file__).resolve().parents[2]/'08_Обучение/runs/.training.lock'
    path.parent.mkdir(parents=True,exist_ok=True); f=path.open('a+b'); f.seek(0)
    if path.stat().st_size==0: f.write(b'0'); f.flush(); f.seek(0)
    acquired=False
    try:
        if os.name=='nt':
            import msvcrt
            try: msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1); acquired=True
            except OSError: raise RuntimeError('GPU занят очередью обучения Panoptr. Выберите CPU или дождитесь завершения.')
        else:
            import fcntl
            try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); acquired=True
            except OSError: raise RuntimeError('GPU занят обучением.')
        yield
    finally:
        if acquired:
            f.seek(0)
            if os.name=='nt': msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(f,fcntl.LOCK_UN)
        f.close()
