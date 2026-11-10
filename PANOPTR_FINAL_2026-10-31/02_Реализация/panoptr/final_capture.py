"""Local recorded-video source; timestamps identify decoded frames, not real camera latency."""
import shutil
import time
from pathlib import Path
import cv2

def _is_lfs_pointer(p: Path) -> bool:
    try:
        if not p.is_file() or p.stat().st_size > 1024:
            return False
        head = p.read_bytes()[:64]
        return head.startswith(b"version https://git-lfs.github.com")
    except Exception:
        return False

class FileCapture:
    def __init__(self,loop=False):
        self.cap=None; self.error=None; self.captured_at=0.; self.started_at=0.; self.sequence=0; self.cached=None; self.finished=False; self.next_at=0.; self.source_time=0.; self.loop=bool(loop); self.path=None
    def start(self,path,width=0,height=0,fps=0):
        self.stop()
        target=Path(path)
        if not target.is_file(): raise ValueError('Видеофайл не найден.')
        if _is_lfs_pointer(target):
            fallback=Path(__file__).resolve().parent/'finale_examples/rail_fallback.avi'
            if fallback.is_file() and not _is_lfs_pointer(fallback):
                try:
                    shutil.copyfile(str(fallback), str(target))
                except Exception:
                    target=fallback
        self.path=str(target); self.cap=cv2.VideoCapture(self.path); self.error=None
        if not self.cap.isOpened(): self.cap.release(); self.cap=None; raise ValueError('Видео не открывается.')
        self.actual_fps=float(self.cap.get(cv2.CAP_FPS if hasattr(cv2,'CAP_FPS') else cv2.CAP_PROP_FPS)) or 25.; self.sequence=0; self.finished=False; self.started_at=time.monotonic(); self.next_at=self.started_at
    def latest(self):
        if self.cap is None: return -1,None
        now=time.monotonic()
        if not self.finished and now>=self.next_at:
            ok,frame=self.cap.read()
            if not ok and self.loop and self.path:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                ok,frame=self.cap.read()
                if not ok:
                    self.cap.release(); self.cap=cv2.VideoCapture(self.path)
                    if self.cap.isOpened(): ok,frame=self.cap.read()
            if ok:
                self.sequence+=1; self.cached=frame; self.captured_at=now; self.source_time=(self.sequence-1)/self.actual_fps
                self.next_at=now+1/self.actual_fps; self.error=None
            else: self.finished=True; self.error='Видео завершено; новые кадры не поступают.'
        return self.sequence,self.cached
    def frame_age(self): return time.monotonic()-(self.captured_at or self.started_at)
    def stop(self):
        if self.cap is not None: self.cap.release()
        self.cap=None; self.cached=None
