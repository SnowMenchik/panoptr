"""Equipment envelope estimates, not deployment acceptance."""
import math
from camera_design import calculate,preset

def engineering(speed_kmh=400.,line_pitch_mm=.6,defect_mm=10.,exposure_us=3.,tile_rows=512,stride_rows=448):
    for value in (speed_kmh,line_pitch_mm,defect_mm,exposure_us):
        if not isinstance(value,(float,int)) or not math.isfinite(value) or value<=0: raise ValueError('Нужны положительные конечные значения.')
    if not isinstance(tile_rows,int) or not isinstance(stride_rows,int) or not 1<=stride_rows<=tile_rows: raise ValueError('Неверный размер/шаг плитки.')
    v=speed_kmh/3.6; rate=v/(line_pitch_mm/1000); sensor=4096; fov=2.; fx=35/.007
    depth=fov*fx/sensor; pitch=60.; height=depth*math.sin(math.radians(pitch)); ahead=depth*math.cos(math.radians(pitch))
    raw=sensor*rate/1e6; tiles=2*rate/stride_rows
    front=preset('forward'); front.update({'width_px':1920,'height_px':1080,'model_size_px':1280,
        'intrinsics_mode':'calibrated','fx_px':20000.,'fy_px':20000.,'cx_px':960.,'cy_px':540.,
        'sensor_width_mm':4.8,'hfov_deg':math.degrees(2*math.atan(1920/40000)),
        'pitch_deg':math.degrees(math.atan2(2.5-.85,300)),'distance_m':300.,'speed_kmh':speed_kmh,
        'exposure_ms':.2,'capture_fps':41.5,'analysis_fps':30.,'predict_ms':20.,'pipeline_ms':20.,'range_max_m':600.})
    camera=calculate(front)
    pressure=.5*1.225*v*v; front_area=.28*.14; force=pressure*1.2*front_area
    result={'speed_kmh':speed_kmh,'line_pitch_mm':line_pitch_mm,'line_rate_required_hz':rate,'camera_max_line_rate_hz':195000,
        'line_rate_margin_fraction':195000/rate-1,'line_rate_pass':rate<=195000,
        'mono8_only_at_target_rate':True,'mode':'1-Line, CameraLink80-bit, 85MHz, Mono8; no TDI/multi-light assumed',
        'defect_independent_px_cross':defect_mm/(fov*1000/sensor),'defect_independent_px_along':defect_mm/line_pitch_mm,
        'smear_along_px':v*exposure_us/1e6/(line_pitch_mm/1000),'exposure_us':exposure_us,
        'capture_line_period_us':1e6/rate,'exposure_period_pass':3<=exposure_us<=1e6/rate,
        'raw_capture_MB_s':raw,'raw_storage_TB_hour':raw*3600/1e6,'raw_4TB_duration_hour':4/(raw*3600/1e6),
        'frame_grabber_continuous_MB_s':1600.,'grabber_payload_fraction':raw/1600.,'camera_link_payload_Gbps':raw*8/1000,
        'line_field_width_m':fov,'line_camera_pitch_deg':pitch,'line_camera_height_m':height,'line_intersection_ahead_m':ahead,
        'line_lens_focal_mm':35.,'line_lens_sensor_width_mm':28.672,'line_slant_distance_m':depth,
        'tile_rows':tile_rows,'stride_rows':stride_rows,'rail_roi_width_px':512,'two_rail_tiles_per_second':tiles,
        'tile_build_time_ms':tile_rows/rate*1000,'single_gpu_mean_service_ms_required':1000/tiles,
        'two_gpu_mean_service_ms_per_gpu_required':2000/tiles,
        'neural_throughput_verified':False,'real_capture_verified':False,'illumination_SNR_verified':False,
        'forward_camera_calculation':camera,'forward_full_to1280_short_px_at300m':.5*20000/300*(1280/5120),
        'forward_roi_to1280_short_px_at300m':camera['target']['model_width_px'],
        'forward_10cm_rock_short_px_at300m':.1*20000/300*(1280/1920),
        'dynamic_pressure_Pa':pressure,'assumed_Cd':1.2,'assumed_area_m2':front_area,
        'preliminary_force_N':force,'preliminary_mount_design_force_N_factor3':force*3,
        'stress_verified':False,'scope':'Flat static scene and nominal uniform scan. Vibration, height, lens MTF, illumination, slip and operating approvals unverified.'}
    return result

def thermal(ambient_c=-40.,target_c=5.,area_m2=.16,u_w_m2k=4.,camera_w=8.,heater_w=50.,mass_kg=2.5,cp=900.):
    vals=(ambient_c,target_c,area_m2,u_w_m2k,camera_w,heater_w,mass_kg,cp)
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in vals) or min(area_m2,u_w_m2k,heater_w,mass_kg,cp)<=0 or camera_w<0: raise ValueError('Неверные тепловые параметры.')
    losses=max(0.,target_c-ambient_c)*area_m2*u_w_m2k; net=heater_w+camera_w-losses
    return {'ambient_c':ambient_c,'target_c':target_c,'assumed_U_W_m2K':u_w_m2k,'area_m2':area_m2,
        'estimated_steady_loss_W':losses,'heater_W':heater_w,'camera_W':camera_w,'positive_steady_margin':net>0,
        'no_loss_warmup_lower_bound_s':mass_kg*cp*max(0.,target_c-ambient_c)/(heater_w+camera_w),
        'estimated_hot_case_max_R_K_W_linecam':(45-40)/(camera_w+10) if camera_w+10>0 else None,
        'note':'U/mass and solar heat10W are assumptions. No startup time guarantee, full transient or measured thermal/IP validation. Camera disabled until interior reaches allowed temperature.'}

def scale_similarity(scale,full_speed,full_object,full_fov,line_rate):
    if not 0<scale<=1 or min(full_speed,full_object,full_fov,line_rate)<=0: raise ValueError('Неверный масштаб.')
    return {'scale':scale,'model_speed_m_s':full_speed*scale,'model_object_m':full_object*scale,
        'model_fov_m':full_fov*scale,'pixels_ratio_unchanged':full_object/full_fov,
        'lines_during_object_unchanged':full_object/full_speed*line_rate,
        'note':'Only pixel/time similarity under corresponding optics and line pitch. No aerodynamic, thermal, vibration or camera capability proof.'}
