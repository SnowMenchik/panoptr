"""Explicit local segmentation weights, CPU only; no automatic downloads/training."""
from pathlib import Path
import time,numpy as np
from path_geometry import propose_geometry,frame_digest
from final_core import sha256

class PathSegmenter:
    def __init__(self): self.model=None; self.key=None
    def infer(self,frame,weights,sequence,generation,stamp,source,min_conf=.35,imgsz=640):
        path=Path(weights).resolve()
        if not path.is_file(): raise ValueError('Нет локальных весов сегментации пути.')
        stat=path.stat(); key=(str(path),stat.st_size,stat.st_mtime_ns)
        if key!=self.key:
            from ultralytics import YOLO
            model=YOLO(str(path))
            if model.task!='segment' or not set(map(str,model.names.values()))&{'rail_left','rail_right','rail','left_rail','right_rail','trackbed','railway_track','track'}: raise ValueError('Нужна сегментационная модель с классами рельсов/пути; COCO не подходит.')
            self.model=model; self.key=key; self.sha=sha256(path)
        started=time.perf_counter(); prediction=self.model.predict(frame,device='cpu',imgsz=imgsz,conf=min_conf,verbose=False,retina_masks=True)[0]
        rows=prediction.boxes.data.detach().cpu().numpy() if prediction.boxes is not None else np.zeros((0,6))
        masks=[]
        if prediction.masks is not None:
            data=prediction.masks.data
            if tuple(data.shape[-2:])!=frame.shape[:2]:
                from ultralytics.utils.ops import scale_masks
                data=scale_masks(data[None].float(),frame.shape[:2])[0]
            masks=[m.detach().cpu().numpy()>.5 for m in data]
        after=path.stat()
        if (after.st_size,after.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns): raise ValueError('Checkpoint изменился во время чтения; используйте завершённые веса.')
        g=propose_geometry(masks,rows,prediction.names,frame.shape,sequence,generation,stamp,source,frame_digest(frame),min_conf)
        g.update(model_sha256=self.sha,model_path=str(path),device='cpu',segmentation_ms=(time.perf_counter()-started)*1000)
        return g
