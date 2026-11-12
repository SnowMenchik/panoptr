"""Pinhole engineering estimates for near ground details and forward obstacles.

World: X right, Y travel direction, Z up (metres). Camera at (0,0,height),
pitch down from horizon. Static target, constant translation and optional pitch rate.
No learned inference, equipment claims, braking control or thermal certification.
"""
from pathlib import Path
import copy,csv,json,math,hashlib
import numpy as np

SOURCES=[{'title':'OpenCV: camera calibration and projection','url':'https://docs.opencv.org/doc/doxygen/html/d4/d93/group__calib.html'},
         {'title':'Basler: electronic shutter types','url':'https://docs.baslerweb.com/electronic-shutter-types'},
         {'title':'Basler: acquisition timing','url':'https://docs.baslerweb.com/acquisition-timing-information'}]

COMMON={'speed_kmh':360.,'width_px':1920,'height_px':1080,'model_size_px':1280,
    'hfov_deg':40.,'sensor_width_mm':6.4,'intrinsics_mode':'fov','fx_px':2000.,'fy_px':2000.,'cx_px':960.,'cy_px':540.,
    'camera_height_m':1.,'pitch_deg':90.,'lateral_offset_m':0.,'distance_m':0.,
    'object_width_m':.01,'object_extent_m':.02,'exposure_ms':.01,
    'capture_fps':240.,'analysis_fps':120.,'predict_ms':5.,'pipeline_ms':5.,
    'pitch_rate_deg_s':0.,'shutter':'global','readout_ms':0.,'min_pixels':16.,
    'max_blur_px':2.,'max_readout_motion_px':2.,'required_observations':3,'required_lead_s':0.,
    'range_min_m':1.,'range_max_m':1000.}

def preset(scenario):
    p=copy.deepcopy(COMMON); p['scenario']=scenario
    if scenario=='forward':
        p.update({'width_px':3840,'height_px':2160,'hfov_deg':30.,'camera_height_m':2.5,
            'pitch_deg':1.,'distance_m':300.,'object_width_m':.5,'object_extent_m':1.7,
            'exposure_ms':.2,'capture_fps':60.,'analysis_fps':30.,'predict_ms':20.,'pipeline_ms':20.,'required_lead_s':2.})
    elif scenario!='near': raise ValueError('Сценарий: near или forward.')
    return p

def validate(values):
    scenario=values.get('scenario','near'); p=preset(scenario); p.update(values)
    strings={'scenario','intrinsics_mode','shutter'}
    for key in p:
        if key in strings: continue
        v=p[key]
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v): raise ValueError('Нужно конечное число: '+key)
    for k in ('width_px','height_px','model_size_px','required_observations'):
        if p[k]!=int(p[k]) or not 1<=p[k]<=32768: raise ValueError('Неверное целое: '+k)
        p[k]=int(p[k])
    positive=['camera_height_m','object_width_m','object_extent_m','capture_fps','min_pixels','range_max_m']
    if any(p[k]<=0 for k in positive): raise ValueError('Размеры, частота камеры и порог детализации должны быть положительными.')
    nonnegative=['speed_kmh','sensor_width_mm','exposure_ms','analysis_fps','predict_ms','pipeline_ms','readout_ms','max_blur_px','max_readout_motion_px','required_lead_s','range_min_m']
    if any(p[k]<0 for k in nonnegative): raise ValueError('Отрицательные скорости, времена и пределы не поддерживаются.')
    if not .1<=p['hfov_deg']<=160 or not 0<=p['pitch_deg']<=90 or abs(p['pitch_rate_deg_s'])>1000:
        raise ValueError('Неверный угол/угловая скорость.')
    if p['scenario']=='forward' and (p['distance_m']<=0 or p['range_min_m']>=p['range_max_m']):
        raise ValueError('Для передней камеры нужны положительная дальность и возрастающий интервал.')
    if p['shutter'] not in ('global','rolling') or p['intrinsics_mode'] not in ('fov','calibrated'):
        raise ValueError('Неверный режим затвора/геометрии.')
    if p['intrinsics_mode']=='calibrated':
        if p['fx_px']<=0 or p['fy_px']<=0 or not 0<=p['cx_px']<=p['width_px'] or not 0<=p['cy_px']<=p['height_px']:
            raise ValueError('Неверные fx/fy/cx/cy калибровки.')
    if p['speed_kmh']>1000 or p['exposure_ms']>1000 or p['range_max_m']>100000:
        raise ValueError('Расчётный диапазон превышен.')
    return p

def intrinsics(p):
    w,h=p['width_px'],p['height_px']
    if p['intrinsics_mode']=='fov':
        fx=w/(2*math.tan(math.radians(p['hfov_deg']/2))); fy=fx; cx=w/2; cy=h/2
    else: fx,fy,cx,cy=[p[k] for k in ('fx_px','fy_px','cx_px','cy_px')]
    # Upsampling never creates additional independent detail.
    scale=min(1.,p['model_size_px']/max(w,h))
    return {'fx_px':fx,'fy_px':fy,'cx_px':cx,'cy_px':cy,'model_scale':scale,
        'actual_hfov_deg':math.degrees(math.atan(cx/fx)+math.atan((w-cx)/fx)),
        'actual_vfov_deg':math.degrees(math.atan(cy/fy)+math.atan((h-cy)/fy)),
        'lens_focal_mm':fx*p['sensor_width_mm']/w if p['sensor_width_mm'] else None,
        'pixel_pitch_um':p['sensor_width_mm']*1000/w if p['sensor_width_mm'] else None,
        'mode':p['intrinsics_mode'],'calibration_verified':False}

def corners(p,distance=None):
    x=p['lateral_offset_m']; y=p['distance_m'] if distance is None else distance; a=p['object_width_m']/2
    b=p['object_extent_m']
    if p['scenario']=='near': return np.array([[x-a,y-b/2,0],[x+a,y-b/2,0],[x+a,y+b/2,0],[x-a,y+b/2,0]],float)
    return np.array([[x-a,y,0],[x+a,y,0],[x+a,y,b],[x-a,y,b]],float)

def trajectory(p,k,points,times):
    times=np.asarray(times,float); world=points[None,:,:].repeat(len(times),axis=0)
    world[:,:,1]-=p['speed_kmh']/3.6*times[:,None]
    y=world[:,:,1]; z=world[:,:,2]-p['camera_height_m']; x=world[:,:,0]
    angle=np.radians(p['pitch_deg']+p['pitch_rate_deg_s']*times)[:,None]
    down=-y*np.sin(angle)-z*np.cos(angle); depth=y*np.cos(angle)-z*np.sin(angle)
    if np.any(depth<=1e-8): return None
    return np.stack([k['cx_px']+k['fx_px']*x/depth,k['cy_px']+k['fy_px']*down/depth],axis=-1)

def trail_span(trail):
    if trail is None: return None
    # Diameter of each sampled corner path, not sum of object width and blur.
    delta=trail[:,None,:,:]-trail[None,:,:,:]
    return float(np.sqrt((delta*delta).sum(axis=-1)).max())

def footprint(p,k,distance,include_trail=False):
    points=corners(p,distance); central=trajectory(p,k,points,[0])
    if central is None: return {'projectable':False,'fully_visible':False,'reason':'target behind focal plane'}
    poly=central[0]; wh=poly.max(axis=0)-poly.min(axis=0)
    exposure=p['exposure_ms']/1000
    trail=trajectory(p,k,points,np.linspace(-exposure/2,exposure/2,17))
    blur=trail_span(trail); valid=trail is not None
    visible=bool(valid and (trail[:,:,0]>=0).all() and (trail[:,:,0]<=p['width_px']).all() and
        (trail[:,:,1]>=0).all() and (trail[:,:,1]<=p['height_px']).all())
    result={'projectable':True,'fully_visible':visible,'sensor_width_px':float(wh[0]),'sensor_height_px':float(wh[1]),
        'model_width_px':float(wh[0]*k['model_scale']),'model_height_px':float(wh[1]*k['model_scale']),
        'sensor_min_px':float(min(wh)),'model_min_px':float(min(wh)*k['model_scale']),
        'sensor_blur_px':blur,'model_blur_px':blur*k['model_scale'] if blur is not None else None,
        'world_corners_m':points.tolist(),'sensor_polygon_px':poly.tolist()}
    result['meets_pixels']=result['model_min_px']>=p['min_pixels']
    result['meets_blur']=result['model_blur_px'] is not None and result['model_blur_px']<=p['max_blur_px']+1e-9
    result['readable_estimate']=visible and result['meets_pixels'] and result['meets_blur']
    if include_trail: result['exposure_trails_sensor_px']=trail.tolist() if trail is not None else None
    return result

def geometric_interval(p,k):
    """Exact full-target image inequalities at zero exposure and current orientation.
    Coefficients are affine in Y; positive depth is required for all four corners.
    """
    theta=math.radians(p['pitch_deg']); sn,cs=math.sin(theta),math.cos(theta)
    lo,hi=-math.inf,math.inf
    for x,yoff,z in corners(p,0):
        dz=z-p['camera_height_m']; da,db=cs,yoff*cs-dz*sn
        va,vb=-sn,-yoff*sn-dz*cs
        inequalities=[(da,db-1e-8),
            (k['cx_px']*da,k['cx_px']*db+k['fx_px']*x),
            ((p['width_px']-k['cx_px'])*da,(p['width_px']-k['cx_px'])*db-k['fx_px']*x),
            (k['cy_px']*da+k['fy_px']*va,k['cy_px']*db+k['fy_px']*vb),
            ((p['height_px']-k['cy_px'])*da-k['fy_px']*va,(p['height_px']-k['cy_px'])*db-k['fy_px']*vb)]
        for a,b in inequalities:
            if abs(a)<1e-12:
                if b<0: return None
            elif a>0: lo=max(lo,-b/a)
            else: hi=min(hi,-b/a)
    if lo>hi: return None
    bounds=(-p['range_max_m'],p['range_max_m']) if p['scenario']=='near' else (p['range_min_m'],p['range_max_m'])
    a,b=max(lo,bounds[0]),min(hi,bounds[1])
    if a>=b: return None
    return {'min_m':float(a),'max_m':float(b),'length_m':float(b-a),'bounded_by_search':bool(lo<bounds[0] or hi>bounds[1]),
        'definition':'full target inside sensor at exposure midpoint; near may extend before and behind camera'}

def readable_intervals(p,k,geo):
    if not geo: return [],[],None
    xs=np.linspace(geo['min_m'],geo['max_m'],501); rows=[]
    for x in xs:
        f=footprint(p,k,float(x)); rows.append({'distance_m':float(x),
            'model_min_px':f.get('model_min_px'),'model_blur_px':f.get('model_blur_px'),
            'readable':f.get('readable_estimate',False)})
    def boundary(a,b,left):
        for _ in range(25):
            mid=(a+b)/2; good=footprint(p,k,mid).get('readable_estimate',False)
            if good==left: a=mid
            else: b=mid
        return (a+b)/2
    segments=[]; start=None
    for i,r in enumerate(rows):
        if r['readable'] and start is None:
            start=r['distance_m'] if i==0 else boundary(xs[i-1],xs[i],False)
        if start is not None and (not r['readable'] or i==len(rows)-1):
            end=r['distance_m'] if r['readable'] else boundary(xs[i-1],xs[i],True)
            if end>start: segments.append([float(start),float(end)])
            start=None
    return segments,rows,float(xs[1]-xs[0])

def exposure_limit(p,k):
    if p['speed_kmh']==0 and p['pitch_rate_deg_s']==0: return None
    q=copy.deepcopy(p); a,b=0.,1000.
    for _ in range(45):
        q['exposure_ms']=(a+b)/2
        f=footprint(q,k,q['distance_m']); blur=f.get('model_blur_px')
        if blur is not None and blur<=p['max_blur_px']: a=q['exposure_ms']
        else: b=q['exposure_ms']
    return a

def calculate(values):
    p=validate(values); k=intrinsics(p); speed=p['speed_kmh']/3.6
    target=footprint(p,k,p['distance_m'],True); geo=geometric_interval(p,k)
    segments,sweep,step=readable_intervals(p,k,geo); length=sum(b-a for a,b in segments)
    lifetime=length/speed if speed else None
    exposure_cap=1000/p['exposure_ms'] if p['exposure_ms'] else None
    capture=min(p['capture_fps'],exposure_cap) if exposure_cap else p['capture_fps']
    service=1000/p['predict_ms'] if p['predict_ms'] else None
    effective=min(capture,p['analysis_fps'],service) if service else min(capture,p['analysis_fps'])
    latency=(p['predict_ms']+p['pipeline_ms'])/1000
    wait=1/effective if effective else None
    response=wait+latency+p['exposure_ms']/1000 if wait is not None else None
    observations=lifetime*effective if lifetime is not None else None
    minimum_count=sum(math.floor((b-a)/speed*effective+1e-10) for a,b in segments) if speed else None
    required_rate=None
    if speed and segments:
        # Conservative phase guarantee across all half-open intervals, not just E[N].
        times=[(b-a)/speed for a,b in segments]; low,high=0.,p['required_observations']/max(times)+1e-8
        for _ in range(45):
            mid=(low+high)/2
            if sum(math.floor(mid*t+1e-10) for t in times)>=p['required_observations']: high=mid
            else: low=mid
        required_rate=high
    best_segment=max((b-a for a,b in segments),default=0.)
    best_time=best_segment/speed if speed else None
    timely_segments=[(a,b) for a,b in segments if speed and response is not None and (b-a)/speed>=response]
    lead=(max(b for a,b in timely_segments)/speed-response) if p['scenario']=='forward' and timely_segments else None
    native=target.get('sensor_min_px'); small=target.get('model_min_px'); ratio=p['min_pixels']/native if native and native>0 else None
    req_width=math.ceil(p['width_px']*ratio) if ratio else None
    req_model=max(p['width_px'],p['height_px'])*ratio if ratio else None
    desired_fx_ratio=p['min_pixels']/small if small and small>0 else None
    max_fov=(math.degrees(2*math.atan(math.tan(math.radians(k['actual_hfov_deg']/2))/desired_fx_ratio))
        if desired_fx_ratio and p['intrinsics_mode']=='fov' else None)
    shutter_envelope=None
    if p['shutter']=='rolling' and p['readout_ms']>0:
        trace=trajectory(p,k,corners(p),np.linspace(-p['readout_ms']/2000,p['readout_ms']/2000,33))
        span=trail_span(trace); shutter_envelope=span*k['model_scale'] if span is not None else None
    checks={'target_inside_frame':target.get('fully_visible',False),'target_resolution':target.get('meets_pixels',False),
        'target_exposure_blur':target.get('meets_blur',False),
        'observation_count':minimum_count is not None and minimum_count>=p['required_observations'],
        'response_within_contiguous_window':best_time is not None and response is not None and response<=best_time}
    if p['scenario']=='forward': checks['requested_lead_time']=lead is not None and lead>=p['required_lead_s']
    if p['shutter']=='rolling': checks['rolling_readout_motion_budget']=shutter_envelope is not None and shutter_envelope<=p['max_readout_motion_px']
    warnings=[]
    if p['model_size_px']>max(p['width_px'],p['height_px']): warnings.append('Увеличение входа сверх исходника не добавляет независимых деталей; масштаб ограничен 1.')
    if p['exposure_ms']==0:
        checks['nonzero_exposure']=False; warnings.append('Нулевая выдержка — идеальный геометрический предел; изображение с достаточным светом этим не подтверждается.')
    if p['predict_ms']==0:
        checks['nonzero_predict_time']=False; warnings.append('Нулевое время predict — идеальный предел, не измеренная производительность.')
    if service and p['analysis_fps']>service: warnings.append('Частота анализа ограничена predict одного последовательного обработчика.')
    if p['analysis_fps']>capture: warnings.append('Частота анализа ограничена поступлением независимых кадров.')
    if exposure_cap and p['capture_fps']>exposure_cap: warnings.append('Заданная выдержка длиннее периода кадров; частота ограничена 1/выдержка.')
    if p['shutter']=='rolling': warnings.append('Rolling shutter: readout создаёт межстрочную деформацию отдельно от смаза экспозиции; точная деформация/направление строк не рассчитаны.' if p['readout_ms'] else 'Rolling shutter: время считывания не задано; межстрочная деформация не оценена.')
    if speed==0: warnings.append('Нулевая скорость: конечная длительность прохода и N не определены. Частота обработки показана отдельно.')
    if geo and geo['bounded_by_search']: warnings.append('Геометрическое окно обрезано заданным интервалом дальностей.')
    if p['pitch_rate_deg_s']: warnings.append('Угловая скорость — постоянный введённый pitch rate; это не измеренная вибрация и не синусоидальная модель.')
    if len(segments)>1: warnings.append('Есть несколько читаемых окон. N суммируется, а своевременность проверяется для самого длинного непрерывного окна.')
    requirements={'sensor_width_px_at_same_aspect_and_fov':req_width,
        'model_size_px_for_current_angular_geometry':math.ceil(req_model/32)*32 if req_model else None,
        'current_native_resolution_sufficient':native is not None and native>=p['min_pixels'],
        'max_hfov_deg_for_current_model_pixels':max_fov,
        'max_exposure_ms_at_design_target':exposure_limit(p,k),
        'min_analysis_fps_for_required_observations':required_rate,
        'min_analysis_fps_geometry_only_lower_bound':p['required_observations']*speed/geo['length_m'] if speed and geo else None,
        'max_predict_ms_if_serial_and_zero_extra_latency':1000/required_rate if required_rate else None,
        'suggested_pitch_to_target_centre_deg':math.degrees(math.atan2(p['camera_height_m']-(p['object_extent_m']/2 if p['scenario']=='forward' else 0),p['distance_m'])),
        'note':'Separate pixel/exposure/throughput requirements, not a jointly validated camera selection. Changing lens or resolution requires recalculation of visibility, blur and window; resizing alone cannot restore detail.'}
    report={'format':'panoptr_camera_design_v1','inputs':p,'intrinsics':k,'target':target,
        'coverage':{'geometric_window':geo,'readable_segments_m':segments,'readable_length_m':length,
            'readable_duration_s':lifetime,'effective_analysis_fps':effective,'effective_capture_fps_estimate':capture,
            'serial_service_fps':service,'expected_observations':observations,
            'uniform_phase_min_observations':minimum_count,
            'uniform_phase_max_observations':sum(math.ceil((b-a)/speed*effective-1e-10) for a,b in segments) if speed else None,
            'sample_spacing_m':speed/effective if effective else None,'numeric_window_grid_step_m':step},
        'motion':{'speed_m_s':speed,'distance_during_exposure_m':speed*p['exposure_ms']/1000,
            'distance_during_predict_m':speed*p['predict_ms']/1000,'distance_during_pipeline_and_predict_m':speed*latency,
            'worst_uniform_first_result_s_estimate':response,'distance_before_first_result_m_estimate':speed*response if response is not None else None,
            'closing_time_at_design_distance_s':p['distance_m']/speed if speed and p['scenario']=='forward' else None,
            'lead_at_first_readable_range_s_estimate':lead,'rolling_full_readout_motion_envelope_model_px':shutter_envelope},
        'data_rate':{'raw_RGB8_MB_s':p['width_px']*p['height_px']*p['capture_fps']*3/1e6,
            'raw_mono8_MB_s':p['width_px']*p['height_px']*p['capture_fps']/1e6,
            'note':'Uncompressed nominal capture payload only; excludes protocol overhead, compression, copies, GPU I/O and dropped frames.'},
        'requirements':requirements,'checks':checks,'all_design_checks_met':all(checks.values()),'warnings':warnings,
        'sources':SOURCES,'range_sweep':sweep,
        'assumptions':['Ideal rectilinear pinhole, square pixels in FOV mode; no lens distortion, defocus or MTF.',
            'Near: flat rectangular ground defect, width X and length Y. Forward: upright rectangle in X/Z plane, base Z=0.',
            'Static object, straight camera motion Y, fixed height; optional constant pitch angular rate. No target walking speed.',
            'Full target visible over sampled exposure; blur is maximum sampled corner path diameter (17 exposure samples).',
            'Model detail uses native resolution capped at scale=1 and aspect-preserving longest-side resize; no tiling/super-resolution.',
            'Readable windows sampled at 501 ranges then transition-refined; a very narrow unsampled window may be missed.',
            'Constant independent frame/analysis rate and one serial worker; sampling phase bounds use half-open windows and conservative sum per segment.',
            'Pipeline latency is additive user input; no queueing, load jitter or measured camera latency. Predict timing supplied by user.',
            'Pixels, blur, observations and lead thresholds are user design targets, not railway regulations or calibrated neural-network Recall.',
            'No illumination/SNR, vibration spectrum, aerodynamic, thermal, electrical or stopping-distance certification.']}
    report['input_signature']=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return report

def export_report(report,output):
    from final_core import save_json
    out=Path(output)
    if out.exists(): raise ValueError('Выберите новую папку расчёта.')
    out.mkdir(parents=True); save_json(out/'report.json',report); save_json(out/'inputs.json',report['inputs'])
    rows=report['range_sweep']
    if rows:
        with (out/'range_sweep.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines=['# Расчёт камеры Panoptr','Сценарий: '+report['inputs']['scenario'],
        'Расчёт идеальной геометрии; не испытание камеры или модели.','', '| Условие | Выполнено |','|---|---|']
    lines += [f'| {k} | {"да" if v else "нет"} |' for k,v in report['checks'].items()]
    for section in ('target','coverage','motion','requirements','data_rate'):
        lines+=['','## '+section]+[f'- {k}: {v}' for k,v in report[section].items() if k not in ('exposure_trails_sensor_px','world_corners_m','sensor_polygon_px')]
    lines+=['','## Допущения']+['- '+x for x in report['assumptions']]
    lines+=['','## Предупреждения']+['- '+x for x in report['warnings']]
    lines+=['','## Источники']+[f"- [{s['title']}]({s['url']})" for s in SOURCES]
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    # Standard static chart, suitable for export; computations don't depend on plotting.
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        xs=[r['distance_m'] for r in rows]; fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
        axes[0].plot(xs,[r['model_min_px'] for r in rows]); axes[0].axhline(report['inputs']['min_pixels'],color='red',ls='--')
        axes[0].set_ylabel('Короткая сторона, px модели'); axes[1].plot(xs,[r['model_blur_px'] for r in rows])
        axes[1].axhline(report['inputs']['max_blur_px'],color='red',ls='--'); axes[1].set_ylabel('Смаз за выдержку, px модели'); axes[1].set_xlabel('Координата вдоль пути, м')
        for ax in axes:
            ax.grid(alpha=.25)
            for a,b in report['coverage']['readable_segments_m']: ax.axvspan(a,b,color='green',alpha=.12)
        fig.suptitle('Расчёт идеальной камеры; параметры оборудования не измерены',fontsize=11); fig.tight_layout(); fig.savefig(out/'range_plot.png',dpi=160); plt.close(fig)
    return out
