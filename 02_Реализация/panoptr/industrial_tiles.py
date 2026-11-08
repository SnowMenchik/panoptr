"""Hardware-independent line packet integrity and overlap assembly.

Vendor CameraLink/10GigE SDK binding is NOT provided by this tested component.
Reject discontinuities and changes instead of manufacturing continuous coverage.
"""
from dataclasses import dataclass
import math,numpy as np

@dataclass
class Strip:
    image: np.ndarray
    first_line: int
    last_line: int
    first_time: float
    last_time: float
    pitch_mm: float
    distance_kind: str

class LineAssembler:
    def __init__(self,width=4096,rows=512,stride=448,pitch_mm=.6):
        if any(not isinstance(v,int) or isinstance(v,bool) for v in (width,rows,stride)) or width<1 or rows<1 or not 1<=stride<=rows or not math.isfinite(pitch_mm) or pitch_mm<=0: raise ValueError('Неверная геометрия сборки.')
        self.width=width; self.rows=rows; self.stride=stride; self.pitch_mm=float(pitch_mm); self.pending=[]; self.expected=None; self.last_time=None; self.gaps=0
    def reset(self): self.pending=[]; self.expected=None; self.last_time=None
    def push(self,image,first_line,timestamps,measured_distance=False):
        a=np.asarray(image); times=list(map(float,timestamps))
        if a.ndim!=2 or a.shape[1]!=self.width or a.dtype!=np.uint8 or len(times)!=len(a) or not len(a): raise ValueError('Нужны Mono8 строки заданной ширины и время каждой строки.')
        if first_line<0 or first_line!=int(first_line) or not all(math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])): raise ValueError('Неверные счётчики/времена строк.')
        if self.expected is not None and (first_line!=self.expected or times[0]<=self.last_time):
            self.gaps+=1; self.reset()
        # A mode/pitch/encoder-quality change cannot share one strip.
        if self.pending and self.pending[-1][3]!=bool(measured_distance): self.gaps+=1; self.reset()
        self.pending.extend((row.copy(),int(first_line)+i,times[i],bool(measured_distance)) for i,row in enumerate(a))
        self.expected=int(first_line)+len(a); self.last_time=times[-1]; result=[]
        while len(self.pending)>=self.rows:
            part=self.pending[:self.rows]
            result.append(Strip(np.stack([r[0] for r in part]),part[0][1],part[-1][1],part[0][2],part[-1][2],self.pitch_mm,
                'external_encoder_assumed_calibrated' if measured_distance else 'nominal_pitch_not_measured'))
            del self.pending[:self.stride]
        return result

def rail_tiles(strip,x_ranges=((250,762),(3334,3846))):
    out=[]
    for start,end in x_ranges:
        if not 0<=start<end<=strip.image.shape[1]: raise ValueError('ROI вне полосы.')
        out.append({'image':strip.image[:,start:end].copy(),'x_offset':start,'y_offset':0,
            'first_line':strip.first_line,'last_line':strip.last_line,'distance_kind':strip.distance_kind})
    return out

def map_box(box,tile):
    if len(box)!=4 or not all(math.isfinite(x) for x in box) or box[2]<=box[0] or box[3]<=box[1]: raise ValueError('Неверная рамка.')
    h,w=tile['image'].shape[:2]
    if not 0<=box[0]<box[2]<=w or not 0<=box[1]<box[3]<=h: raise ValueError('Рамка вне исходного ROI; сначала снимите letterbox.')
    return [box[0]+tile['x_offset'],box[1],box[2]+tile['x_offset'],box[3]]
