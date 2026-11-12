"""Read-only supplied rail catalogue; never translates YOLO classes to diagnoses."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parent
RAIL_CLASSES={'rail_crack','rail_break','rail_surface_damage','rail_corrosion','track_defect_generic','rail_surface_defect',
 'corrosion','cracks','flakings','shelling','spalling','squats','wheel-burn','damage','gap','d_dent','d_crush','d_scratch','d_slant'}
def catalogue():return json.loads((ROOT/'reference/rail_2499r.json').read_text(encoding='utf-8'))
def advice(class_name):
    relevant=class_name in RAIL_CLASSES
    return {'relevant':relevant,'class_name':class_name,'normative_code':None,'automatic_diagnosis':False,
      'text':('Визуальный признак. Код № 2499р, ДР/ОДР, размеры и допустимая скорость моделью не установлены. Нужны сведения о месте, причине и проверка специалистом; скрытый дефект камерой не исключается.' if relevant else
              '№ 2499р относится к дефектам рельсов. Для этого класса автоматическое сопоставление с кодами рельсов не применяется.'),
      'source_edition':'Предоставленная редакция от 13.11.2024; последующие изменения здесь не проверены.'}
def valid_codes():
    codes=set()
    for row in catalogue()['entries']:
        key=row['code_context'] or ''
        match=re.match(r'^(\d{2})\.(.*)$',key)
        if match:
            codes.update(match[1]+'.'+digit for digit in re.findall(r'[0-4]',match[2]))
    return codes
def validate_review(class_name,values):
    status=values.get('status','visual_only');code=str(values.get('code','')).strip();reviewer=str(values.get('reviewer','')).strip();note=str(values.get('note','')).strip();method=str(values.get('method','')).strip()
    if status not in ('visual_only','needs_measurement','expert_recorded'):raise ValueError('Неизвестный статус обследования.')
    if code and not advice(class_name)['relevant']:raise ValueError('Код дефекта рельса неприменим к выбранному классу.')
    if code and code not in valid_codes():raise ValueError('Выберите код из предоставленного каталога. Класс модели не является кодом.')
    if status!='expert_recorded' and code:raise ValueError('Код вводится только как отдельная запись специалиста.')
    if status=='expert_recorded' and (not code or not reviewer or not note or not method):raise ValueError('Для записи специалиста нужны код, автор, способ проверки и обоснование.')
    return {'status':status,'code':code or None,'reviewer':reviewer,'note':note,'method':method,
            'automatic_diagnosis':False,'standard':'2499р / предоставленная ред. 13.11.2024'}
