"""Explicit model versions, including a frozen interim checkpoint from run10."""
from pathlib import Path
import json

MODEL_SPECS=[
    {'run':'01_rail9_finetune','filename':'rail9_finetuned_2026.pt','title':'01 · Рельсы — дообученные 9 классов · кандидат','imgsz':1280,'status':'candidate',
     'purpose':'Дефекты рельсов, скреплений и шпал; полнота требует проверки на рабочем видео.'},
    {'run':'02_neu_defect_seg','filename':'rail_surface_neu_seg_2026.pt','title':'02 · Поверхность рельса — маски дефектов · кандидат','imgsz':640,'status':'candidate_near_surface',
     'purpose':'Маски дефектов на близких снимках поверхности; не сегментация рельсов/коридора.'},
    {'run':'03_multiclass8','filename':'rail_multiclass8_2026.pt','title':'03 · Дефекты 8 классов · эксперимент','imgsz':1280,'status':'experimental',
     'purpose':'Невысокая полнота; дополнительное исследование, не основная модель защиты.'},
    {'run':'04_railfod_obstacles','filename':'overhead_railfod4_2026.pt','title':'04 · Контактная сеть — посторонние предметы · кандидат','imgsz':960,'status':'candidate_synthetic',
     'purpose':'Гнёзда, пакеты, лёгкие предметы, шары на контактной сети. Нужен реальный независимый контроль.'},
    {'run':'05_rail400_baseline','filename':'rail400_detect8_2026.pt','title':'05 · Rail400 — 8 классов · резерв','imgsz':1280,'status':'experimental_reserve',
     'purpose':'Малый набор и невысокая полнота. Dirt/unknown не объявляются дефектами.'}]

TRAINED_DEFECT_CLASSES=['rail_crack','rail_break','rail_surface_damage','rail_corrosion','fastener_defect',
    'fastener_missing','joint_fishplate_defect','sleeper_defect','track_defect_generic','rail_surface_defect',
    'missing-fastener','damage','gap','d_dent','d_crush','d_scratch','d_slant']
OVERHEAD_CLASSES=['niaocao','suliaodai','piaofuwu','qiqiu']
PERSON_MODEL_SPECS=[{'run':'07_person_railbench','filename':'person_railbench07_2026.pt',
    'title':'07 · Человек — RAIL-BENCH · 1 класс · кандидат','imgsz':1536,'status':'candidate_real_validation',
    'purpose':'Строго person. Дообучена на железнодорожных кадрах; независимый test и оценка событий ещё необходимы.'},
    {'run':'08_person_synrailobs_replay','filename':'person_synrailobs08_2026.pt',
    'title':'08 · Человек — синтетическое дообучение · резерв','imgsz':1536,'status':'experimental_reserve_recall_regression',
    'purpose':'Строго person. Precision выше, но полнота и mAP50 ниже 07 на реальном validation; 07 остаётся основной.'}]

ROCK_MODEL_SPECS=[{'run':'09_rock_synrailobs','filename':'rock_synrailobs09_2026.pt',
    'title':'09 · Камни — эксперимент · слабая проверка на реальных кадрах','imgsz':1280,'status':'experimental_external_recall_failure',
    'purpose':'Строго rock. На реальном MR48 recall0.56%, mAP50 0.5%: не рекомендуется как основная модель. Для сравнения/макета; проверить коридор, балласт и оператором.'}]

DRONE_MODEL_SPECS=[{'run':'10_drone_broad_v2','filename':'drone_broad10_interim_2026.pt',
    'title':'10 · БПЛА — общий класс · промежуточная · best 53','imgsz':1280,'status':'interim_training_stopped',
    'purpose':'Один класс drone. Обучение остановлено пользователем после 65 эпох; установлена лучшая версия с эпохи 53. Воздушное наблюдение для проверки оператором; тип, дальность и риск столкновения не определяются.'}]

def available_models(root):
    return [dict(s,path=str(Path(root)/'models'/s['filename'])) for s in MODEL_SPECS+PERSON_MODEL_SPECS+ROCK_MODEL_SPECS+DRONE_MODEL_SPECS if (Path(root)/'models'/s['filename']).is_file()]

def trained_config(root):
    root=Path(root); cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    # A separate explicit profile; no substitution of the user's saved startup settings.
    for channel,spec in zip(cfg['cameras'],[MODEL_SPECS[0],MODEL_SPECS[3]]):
        channel.update(model=str(root/'models'/spec['filename']),image_size=spec['imgsz'],device='CPU')
    return cfg

def person07_config(root):
    cfg=trained_config(root); spec=PERSON_MODEL_SPECS[0]
    cfg['cameras'][1].update(model=str(Path(root)/'models'/spec['filename']),image_size=spec['imgsz'],device='CPU')
    return cfg

def person08_config(root):
    """Explicit reserve profile. The recommended person07 profile stays intact."""
    cfg=trained_config(root); spec=PERSON_MODEL_SPECS[1]
    cfg['cameras'][1].update(model=str(Path(root)/'models'/spec['filename']),image_size=spec['imgsz'],device='CPU')
    return cfg

def rock09_config(root):
    """Explicit rail-defects + experimental rock profile; saved settings stay intact."""
    cfg=trained_config(root); spec=ROCK_MODEL_SPECS[0]
    cfg['cameras'][1].update(model=str(Path(root)/'models'/spec['filename']),image_size=spec['imgsz'],device='CPU')
    return cfg

def drone10_config(root):
    """Explicit rail-defects + interim aerial observation profile."""
    cfg=trained_config(root);spec=DRONE_MODEL_SPECS[0]
    cfg['cameras'][1].update(model=str(Path(root)/'models'/spec['filename']),image_size=spec['imgsz'],device='CPU')
    return cfg
