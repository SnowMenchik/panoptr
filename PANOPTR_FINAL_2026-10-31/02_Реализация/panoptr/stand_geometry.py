"""Editable metre-based meshes for a virtual bench and generic vehicle mounting.

Concept assembly, not manufacturing-ready CAD or an actual train body drawing.
"""
from dataclasses import dataclass
from pathlib import Path
import math,json
import numpy as np

@dataclass
class Mesh:
    key:str
    name:str
    vertices:np.ndarray
    faces:list
    color:tuple
    role:str='structure'

def box(key,name,center,size,color,role='structure'):
    points=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)*np.array(size)/2+center
    return Mesh(key,name,points,[[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]],color,role)

def cylinder(key,name,start,end,radius,color,role='structure',segments=16):
    a,b=np.array(start,float),np.array(end,float); axis=b-a; axis/=np.linalg.norm(axis)
    helper=np.array([0,0,1.]) if abs(axis[2])<.9 else np.array([1.,0,0]); right=np.cross(axis,helper); right/=np.linalg.norm(right); up=np.cross(axis,right)
    ring=np.array([radius*(math.cos(i*2*math.pi/segments)*right+math.sin(i*2*math.pi/segments)*up) for i in range(segments)])
    points=np.concatenate([ring+a,ring+b]); faces=[list(reversed(range(segments))),list(range(segments,2*segments))]
    faces += [[i,(i+1)%segments,(i+1)%segments+segments,i+segments] for i in range(segments)]
    return Mesh(key,name,points,faces,color,role)

def transform(mesh,origin,angle_deg=0,translation=(0,0,0)):
    a=math.radians(angle_deg); rot=np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])
    mesh.vertices=(mesh.vertices-origin)@rot.T+origin+translation; return mesh

def camera_module(prefix,position,pitch=0,explode=0.,front=False):
    p=np.array(position,float); out=[]; dark=(36,48,60); orange=(226,130,43); metal=(126,143,151)
    # Segment panels create a visible service cutaway, rather than a filled cube.
    size=np.array([.18,.32,.19]) if not front else np.array([.18,.28,.19])
    w,l,h=size; width=.008
    panels=[('left',[-w/2,0,0],[width,l,h]),('right',[w/2+explode*.18,0,0],[width,l,h]),
      ('top',[0,0,h/2+explode*.18],[w,l,width]),('bottom',[0,0,-h/2],[w,l,width]),
      ('rear',[0,-l/2,0],[w,width,h])]
    for key,c,d in panels: out.append(box(prefix+'_'+key,'Термокожух: '+key,p+c,d,metal,'housing'))
    body=[.074,.0788,.074] if front else [.062,.0369,.062]
    out.append(box(prefix,'MV-CH250-90TC-C-NF' if front else 'MV-CL042-91CM-V2',p+[0,-.045,0],body,orange,'camera'))
    out.append(cylinder(prefix+'_lens','50мм C-mount' if front else '35мм F-mount и адаптер',p+[0,-.01,0],p+[0,.07,0],.027,dark,'lens'))
    out.append(box(prefix+'_window','Сменное защитное окно, размер проектный',p+[0,l/2+explode*.16,0],[.085,.004,.085],(74,188,196),'window'))
    out.append(box(prefix+'_heater','PTC / теплоотвод / датчик',p+[0,-.03,-.065],[.08,.10,.018],(190,64,54),'heater'))
    out.append(box(prefix+'_TEC','Резерв активного охлаждения: размер условный, SKU поТЗ',p+[0,-.03,-h/2],[.06,.06,.012],(62,124,180),'cooler'))
    out.append(box(prefix+'_seal','Рамка уплотнения окна',p+[0,l/2-.01,0],[.105,.012,.105],dark,'seal'))
    for x in (-.038,.038):
        for z in (-.038,.038): out.append(cylinder(prefix+'_screw','Стяжной винт концепции',p+[x,l/2,z],p+[x,l/2+.015,z],.005,metal,'fastener'))
    for i in range(5): out.append(box(prefix+'_rib','Ребро теплоотвода',p+[i*.025-.05,-.03,-h/2-.015],[.006,.13,.028],metal,'heat_sink'))
    return [transform(m,p,-pitch) for m in out]

def actor(kind,y=1.2,x=-.76):
    out=[]
    if kind=='person':
        out.append(box('target','Фигурка человека: геометрический эталон',[x,y,.95],[.4,.25,.7],(231,161,63),'target'))
        out.append(cylinder('target','Голова',[x,y,1.4],[x,y,1.64],.12,(233,190,150),'target'))
        for dx in (-.12,.12): out.append(cylinder('target','Нога',[x+dx,y,.08],[x+dx,y,.6],.07,(47,69,89),'target'))
    elif kind=='rock':
        p=np.array([[x-.2,y-.15,0],[x+.2,y-.15,0],[x+.16,y+.2,0],[x-.16,y+.2,0],[x-.1,y-.1,.32],[x+.1,y+.1,.45]])
        out.append(Mesh('target','Камень: геометрический эталон',p,[[0,3,2,1],[0,1,4],[1,2,5,4],[2,3,5],[3,0,4,5],[4,5,3]],(123,116,99),'target'))
    else:
        out.append(box('target','Нормальная / дефектная вставка',[x,y,.014],[.17,.13,.028],(86,103,112),'target'))
        if kind!='missing':
            out.append(box('target','Крепление: '+kind,[x,y,.042],[.08,.08,.03],(196,82,58) if kind=='inverted' else (140,155,159),'target'))
        else: out.append(box('target','Место отсутствующего крепления',[x,y,.03],[.06,.06,.004],(24,29,32),'target'))
    return out

def scene(mode='bench',kind='missing',position=1.2,explode=0.,shutter=False,lamp='idle',gauge_mm=1520.):
    if isinstance(gauge_mm,bool) or not isinstance(gauge_mm,(int,float)) or not math.isfinite(gauge_mm) or not 10<=gauge_mm<=2000:raise ValueError('Колея: 10–2000 мм.')
    # Small educational models use a uniformly scaled concept, not camera acceptance.
    scale=gauge_mm/1520. if gauge_mm<300 else 1.
    base_gauge=1520. if scale!=1 else gauge_mm
    half=base_gauge/2000+.075/2  # Gauge is the clear distance between inner head edges.
    out=[]; steel=(150,164,171); frame=(62,88,105)
    length=4 if mode=='bench' else 18
    out.append(box('base','Основание / плита пути',[0,0,-.16],[2.25,length,.15],(101,115,114)))
    for side in (-1,1):
        x=side*half
        out.append(box('rail',f'Рельс: условное сечение, колея{gauge_mm:g}мм',[x,0,-.02],[.075,length,.04],steel))
        out.append(box('rail_web','Стенка рельса',[x,0,-.055],[.015,length,.04],steel))
        out.append(box('rail_foot','Подошва рельса',[x,0,-.09],[.15,length,.03],steel))
    # Slab/fasteners instead of silently assuming ballast is the actual VSM track.
    for y in np.arange(-length/2+.3,length/2,.6):
        for x in (-half,half):
            out.append(box('normal','Нормальное крепление',[x,y,.016],[.16,.12,.032],(110,122,127)))
    if mode=='bench':
        for x in (-1.1,1.1):
            for y in (-1.8,1.8):
                out.append(box('post','Профильная стойка концепции',[x,y,1.25],[.08,.08,2.5],frame))
                out.append(box('foot','Регулируемая опора',[x,y,-.27],[.17,.17,.08],(44,52,59)))
        out.append(box('beam','Балка камеры',[0,0,2.42],[2.28,.08,.08],frame))
        out.append(box('stage','Каретка для сменных элементов',[0,position,-.11],[1.95,.3,.06],(55,93,118)))
        out.append(box('guard','Экран оператора / ограждение',[1.24,0,.7],[.015,3.6,.95],(69,136,156)))
    else:
        out.append(box('vehicle','Условная диагностическая платформа; не чертёж состава',[0,-4,1.55],[2.8,8,2.8],(207,216,219)))
        out.append(box('glass','Лобовое окно условной платформы',[0,.025,2.65],[1.75,.02,.45],(48,103,125)))
        for y in (-6.4,-1.5):
            for x in (-1.15,1.15): out.append(cylinder('wheel','Колесо: условная геометрия',[x-.07,y,.4],[x+.07,y,.4],.4,(50,57,66)))
    h=2*(35/.007)/4096*math.sin(math.radians(60)); cam_y=0 if mode=='bench' else .12
    out += camera_module('CAM_LINE',[0,cam_y,h],60,explode)
    out += camera_module('CAM_FRONT',[.35,cam_y,2.55],.32,explode,True)
    out.append(box('bracket','Кронштейн линейной камеры: концепция',[0,cam_y,h+.17],[.06,.12,.32],frame,'mount'))
    out.append(box('bracket','Кронштейн передней камеры: концепция',[.35,cam_y,2.44],[.06,.12,.16],frame,'mount'))
    for x in (-1.,1.):
        out.append(cylinder('scan_ray','Расчётная линия сканирования',[0,cam_y,h],[x,cam_y+1.2207,.015],.006,(84,214,216),'optics'))
    out.append(box('scan_line','Полоса сканирования: ширина2м',[0,cam_y+1.2207,.018],[2.,.012,.006],(84,214,216),'optics'))
    for x in (-half,half):
        out.append(box('light','Подсветка поТЗ: мощность/яркость ещё не измерены',[x,1.22,.6],[.3,.10,.06],(216,187,96)))
    out.append(box('cabinet','Шкаф питания/компьютер',[1.65,-1.1,.6],[.65,.58,1.1],(72,94,116)))
    out.append(box('computer','Nuvo10208GC: номинальный корпус',[1.65,-1.1,.83],[.268,.4,.196],(37,47,56)))
    lamp_color={'idle':(65,164,136),'event':(224,67,59),'unassessed':(236,166,58)}[lamp]
    out.append(cylinder('indicator','Демонстрационная лампа24В',[1.65,-1.1,1.25],[1.65,-1.1,1.44],.06,lamp_color,'indicator'))
    if shutter: out.append(box('shutter','Закрытая заслонка виртуального объектива',[0,cam_y+.16,h-.19],[.22,.08,.20],(22,24,27)))
    out += actor(kind,position,0 if kind in ('person','rock') else -half)
    if scale!=1:
        for m in out:m.vertices*=scale
    return out

def export_meshes(meshes,folder,label):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True); obj=[]; mtl=[]; index=1; stl=['solid '+label]
    for i,m in enumerate(meshes):
        obj += ['o '+m.key+'_'+str(i),'usemtl c'+str(i)]
        mtl += ['newmtl c'+str(i),'Kd '+' '.join(str(c/255) for c in m.color)]
        obj += ['v '+' '.join(f'{v:.7f}' for v in p) for p in m.vertices]
        for f in m.faces:
            obj.append('f '+' '.join(str(index+j) for j in f))
            for j in range(1,len(f)-1):
                vertices=m.vertices[[f[0],f[j],f[j+1]]]*1000
                normal=np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]); n=np.linalg.norm(normal); normal=normal/n if n else normal
                stl += ['facet normal '+' '.join(map(str,normal)),'outer loop']+['vertex '+' '.join(map(str,p)) for p in vertices]+['endloop','endfacet']
        index+=len(m.vertices)
    (folder/(label+'.obj')).write_text('mtllib '+label+'.mtl\n'+'\n'.join(obj),encoding='utf-8')
    (folder/(label+'.mtl')).write_text('\n'.join(mtl),encoding='utf-8')
    (folder/(label+'.stl')).write_text('\n'.join(stl+['endsolid '+label]),encoding='utf-8')
    data=[{'key':m.key,'name':m.name,'role':m.role,'bounds_m':[m.vertices.min(axis=0).tolist(),m.vertices.max(axis=0).tolist()]} for m in meshes]
    (folder/(label+'_parts.json')).write_text(json.dumps({'obj_units':'metres','stl_units':'millimetres','manufacturing_ready':False,'parts':data},ensure_ascii=False,indent=2),encoding='utf-8')

def view_transform(eye,target):
    eye=np.asarray(eye,float); f=np.asarray(target,float)-eye; f/=np.linalg.norm(f)
    r=np.cross(f,[0,0,1]); r/=np.linalg.norm(r); up=np.cross(r,f); return eye,np.stack([r,-up,f])

def project(vertices,eye,target,width,height,focal=None):
    origin,rotation=view_transform(eye,target); points=(vertices-origin)@rotation.T
    focal=focal or min(width,height)*1.15
    if (points[:,2]<=.01).any(): return None,None
    pixels=np.column_stack([width/2+focal*points[:,0]/points[:,2],height/2+focal*points[:,1]/points[:,2]])
    return pixels,points[:,2]
