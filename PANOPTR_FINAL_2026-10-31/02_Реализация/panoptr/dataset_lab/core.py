"""Non-destructive dataset import, audit, grouping and preparation."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import random
import re
import shutil

import numpy as np
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def under(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Путь выходит за выбранную папку: ' + str(name))
    return path

def files(root, extensions):
    root = Path(root).resolve()
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in extensions and p.resolve().is_relative_to(root))

def mask_values(path):
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 2:
        values, counts = np.unique(arr, return_counts=True)
        return {str(int(v)): int(n) for v, n in zip(values, counts)}
    if arr.ndim == 3 and arr.shape[2] == 3:
        values, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
        return {','.join(map(str, v)): int(n) for v, n in zip(values, counts)}
    raise ValueError('Маска должна быть индексной/серой или RGB; RGBA требует явного преобразования.')

def decode_mask(path, mapping, class_count):
    with Image.open(path) as im:
        arr = np.array(im)
    out = np.full(arr.shape[:2], 255, dtype=np.uint8)
    covered = np.zeros(arr.shape[:2], dtype=bool)
    for key, value in mapping.items():
        value = int(value)
        if value != -1 and not 0 <= value < class_count:
            raise ValueError('Некорректный целевой класс: ' + str(value))
        if arr.ndim == 2:
            hit = arr == int(key)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            rgb = tuple(map(int, key.split(',')))
            if len(rgb) != 3:
                raise ValueError('Для RGB-маски нужен ключ R,G,B.')
            hit = np.all(arr == rgb, axis=2)
        else:
            raise ValueError('Неподдерживаемый формат маски.')
        out[hit] = 255 if value == -1 else value
        covered |= hit
    if not covered.all():
        raise ValueError('В маске есть значения без явного соответствия классу.')
    return out

def key(path, root, suffix=''):
    rel = Path(path).relative_to(root).with_suffix('').as_posix()
    if suffix and rel.endswith(suffix):
        rel = rel[:-len(suffix)]
    return rel.casefold()

def pair_inputs(cfg):
    if not cfg.get('images') or not Path(cfg['images']).is_dir():
        raise ValueError('Выберите существующую папку изображений.')
    if not cfg.get('annotations') or not Path(cfg['annotations']).exists():
        raise ValueError('Выберите существующие аннотации.')
    image_root = Path(cfg['images']).resolve()
    images = files(image_root, IMAGE_EXTS)
    issues = []
    if not images:
        issues.append('В папке нет изображений.')
    if cfg['format'] == 'coco_bbox':
        coco = read(cfg['annotations'])
        categories = {int(c['id']): str(c['name']) for c in coco['categories']}
        if len(categories) != len(coco['categories']):
            raise ValueError('COCO содержит повторяющиеся category_id.')
        ids = sorted(categories)
        names = [categories[i] for i in ids]
        if cfg.get('classes') != names:
            raise ValueError('Классы COCO должны соответствовать categories, отсортированным по id: ' + str(names))
        anns = collections.defaultdict(list)
        for a in coco['annotations']:
            anns[a['image_id']].append(a)
        records = []
        seen_ids = set()
        seen_paths = set()
        for im in coco['images']:
            if im['id'] in seen_ids:
                issues.append('Повторяющийся COCO image_id: ' + str(im['id']))
            seen_ids.add(im['id'])
            p = under(image_root, im['file_name'])
            if p in seen_paths:
                issues.append('Один COCO-файл описан несколько раз: ' + str(p))
            seen_paths.add(p)
            records.append({'image': str(p), 'annotation': str(Path(cfg['annotations']).resolve()),
                            'explicit_group':im.get('group',''),
                            'coco_image': im, 'coco_annotations': anns[im['id']],
                            'category_map': {str(v): i for i, v in enumerate(ids)}})
        for identifier in set(anns) - seen_ids:
            issues.append('Аннотации с отсутствующим COCO image_id: ' + str(identifier))
        return records, issues, []
    label_root = Path(cfg['annotations']).resolve()
    labels = files(label_root, {'.txt'} if cfg['format'] in ('yolo_bbox','yolo_segment') else {'.png', '.tif', '.tiff', '.bmp'})
    records = []
    used = set()
    if cfg.get('pairs_csv'):
        with Path(cfg['pairs_csv']).open(encoding='utf-8-sig', newline='') as stream:
            table = list(csv.DictReader(stream))
        for row in table:
            p = under(image_root, row['image'])
            label = under(label_root, row['annotation'])
            records.append({'image': str(p), 'annotation': str(label), 'explicit_group': row.get('group', '').strip()})
            used.add(label)
        listed = {Path(r['image']) for r in records}
        if len(listed) != len(records):
            issues.append('CSV сопоставления повторяет одно изображение.')
        for p in set(images) - listed:
            issues.append('Изображение отсутствует в CSV: ' + str(p))
    else:
        indexed = collections.defaultdict(list)
        image_keys = collections.Counter(key(p, image_root) for p in images)
        for p in labels:
            indexed[key(p, label_root, cfg.get('mask_suffix', ''))].append(p)
        for p in images:
            k = key(p, image_root)
            candidates = indexed[k]
            if len(candidates) != 1 or image_keys[k] != 1:
                issues.append('Нет однозначной пары: ' + str(p) + ' (масок/меток: ' + str(len(candidates)) + ')')
                continue
            records.append({'image': str(p), 'annotation': str(candidates[0])})
            used.add(candidates[0])
    unused = [str(p) for p in labels if p not in used]
    return records, issues, unused

def boxes_for(record, cfg, width, height):
    count = len(cfg['classes'])
    boxes = []
    if cfg['format'] == 'yolo_bbox':
        for line in Path(record['annotation']).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError('Нужны YOLO bbox из 5 значений; полигоны нельзя читать как bbox.')
            c = int(values[0]); x, y, w, h = map(float, values[1:])
            if not 0 <= c < count or not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h) <= 0 or x-w/2 < -1e-7 or y-h/2 < -1e-7 or x+w/2 > 1+1e-7 or y+h/2 > 1+1e-7:
                raise ValueError('Класс или координаты bbox вне допустимого диапазона.')
            boxes.append([c,x,y,w,h])
    else:
        if record['coco_image']['width'] != width or record['coco_image']['height'] != height:
            raise ValueError('COCO width/height не совпадают с изображением.')
        for a in record['coco_annotations']:
            if a.get('iscrowd', 0) or a.get('ignore', 0):
                raise ValueError('COCO crowd/ignore требует отдельной политики; кадр не преобразован автоматически.')
            x,y,w,h = map(float, a['bbox'])
            if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h) <= 0 or min(x,y) < 0 or x+w > width+1e-5 or y+h > height+1e-5:
                raise ValueError('Некорректный COCO bbox.')
            c = record['category_map'][str(a['category_id'])]
            boxes.append([c,(x+w/2)/width,(y+h/2)/height,w/width,h/height])
    return boxes

def segments_for(record,cfg):
    rows=[]
    for line in Path(record['annotation']).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        a=line.split(); c=int(a[0]); coords=list(map(float,a[1:]))
        if len(coords)<6 or len(coords)%2 or not 0<=c<len(cfg['classes']) or not all(math.isfinite(v) and 0<=v<=1 for v in coords):
            raise ValueError('Неверный YOLO polygon.')
        points=np.array(coords).reshape(-1,2)
        area=abs(np.dot(points[:,0],np.roll(points[:,1],1))-np.dot(points[:,1],np.roll(points[:,0],1)))/2
        if len(np.unique(points,axis=0))<3 or min(np.ptp(points,axis=0))<=0 or area<=1e-12: raise ValueError('Вырожденный полигон.')
        rows.append([c,*coords])
    return rows

def audit(cfg):
    if cfg['format'] not in ('semantic_masks', 'yolo_bbox', 'yolo_segment', 'coco_bbox'):
        raise ValueError('Неизвестный формат.')
    classes = cfg.get('classes', [])
    records, issues, unused = pair_inputs(cfg)
    if not classes or len(classes) != len(set(classes)) or any(not str(c).strip() for c in classes):
        issues.append('Нужно явно задать уникальные названия классов.')
    if cfg['format'] == 'semantic_masks' and not 2 <= len(classes) <= 254:
        issues.append('Семантическая задача требует 2–254 класса, включая фон.')
    checked = []
    discovered = collections.Counter()
    image_hashes = collections.defaultdict(list)
    class_counts = collections.Counter()
    image_root = Path(cfg['images']).resolve()
    pattern = re.compile(cfg['group_regex']) if cfg.get('group_regex') else None
    for index, original in enumerate(records):
        try:
            r = dict(original)
            with Image.open(r['image']) as im:
                im.load(); width,height = im.size
                if im.getexif().get(274, 1) != 1:
                    raise ValueError('EXIF-поворот: сначала согласуйте ориентацию изображения и разметки.')
            if min(width,height) < 1:
                raise ValueError('Пустое изображение.')
            r.update(width=width,height=height,sha256=digest(r['image']))
            rel = Path(r['image']).relative_to(image_root)
            group = r.get('explicit_group', '')
            if not group and pattern:
                match = pattern.search(rel.with_suffix('').as_posix())
                if not match or not match.groups():
                    raise ValueError('Регулярное выражение групп не сопоставилось или не содержит группу захвата.')
                group = match.group(1)
            if not group and cfg.get('independent_images'):
                group = 'image:' + r['sha256']
            if not group and rel.parent != Path('.'):
                group = 'folder:' + rel.parent.as_posix()
            if not group:
                raise ValueError('Неизвестная сцена: задайте group в CSV/regex или подтвердите независимость изображений.')
            r['group'] = group
            if cfg['format'] == 'semantic_masks':
                with Image.open(r['annotation']) as mask:
                    mask.load()
                    if mask.size != (width,height):
                        raise ValueError('Размер маски не совпадает с изображением; автоподгонка запрещена.')
                values = mask_values(r['annotation']); discovered.update(values)
                target = decode_mask(r['annotation'], cfg.get('mask_mapping', {}), len(classes))
                valid = target != 255
                if not valid.any():
                    raise ValueError('Все пиксели ignore; кадр не пригоден для обучения.')
                v,n = np.unique(target[valid],return_counts=True)
                class_counts.update({int(a):int(b) for a,b in zip(v,n)})
                r['annotation_sha256'] = hashlib.sha256(target.tobytes()).hexdigest()
            elif cfg['format']=='yolo_segment':
                segments=segments_for(r,cfg); class_counts.update(int(s[0]) for s in segments); r['segments']=segments
                r['annotation_sha256']=hashlib.sha256(json.dumps(segments).encode()).hexdigest()
            else:
                boxes = boxes_for(r,cfg,width,height)
                class_counts.update(int(b[0]) for b in boxes)
                r['boxes'] = boxes
                r['annotation_sha256'] = hashlib.sha256(json.dumps(sorted(boxes)).encode()).hexdigest()
            image_hashes[r['sha256']].append(r)
            checked.append(r)
        except Exception as exc:
            issues.append(str(original['image']) + ': ' + str(exc))
        if index % 100 == 0:
            print('Проверено', index+1, '/', len(records), flush=True)
    duplicates = []
    for h, same in image_hashes.items():
        if len(same) > 1:
            duplicates.append([r['image'] for r in same])
            if len({r['annotation_sha256'] for r in same}) > 1:
                issues.append('Одинаковые изображения имеют разную разметку: ' + str(duplicates[-1]))
            groups = {r['group'] for r in same}
            if len(groups) > 1:
                issues.append('Дубликат попал в разные сцены: ' + str(duplicates[-1]))
    if unused:
        issues.append('Остались несопоставленные аннотации: ' + str(len(unused)) + '. Исправьте входные папки или CSV.')
    if len({r['group'] for r in checked}) < 2:
        issues.append('Нужно минимум две независимые группы для train/val.')
    return {'version':1,'configuration':cfg,'ready':not issues,'pairs':records,'records':checked,'issues':issues,
            'unused_annotations':unused,'duplicates':duplicates,'mask_values':dict(discovered),
            'class_counts':{str(k):v for k,v in class_counts.items()},'group_count':len({r['group'] for r in checked}),
            'scope':'File/pair/geometry/explicit class checks; group provenance must be supplied by the user. No manual ground-truth correctness guarantee.'}

def prepare(cfg, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Выберите новую папку: существующие результаты не перезаписываются.')
    for field in ('images','annotations'):
        source = Path(cfg[field]).resolve()
        if source.is_dir() and output.is_relative_to(source):
            raise ValueError('Выходная папка не должна находиться внутри исходных данных.')
    report = audit(cfg)
    if not report['ready']:
        raise ValueError('Подготовка заблокирована: ' + '\n'.join(report['issues'][:20]))
    ratio = float(cfg.get('val_fraction', 0.2))
    if not 0 < ratio < 1:
        raise ValueError('Доля val должна быть между 0 и 1.')
    groups = sorted({r['group'] for r in report['records']})
    random.Random(int(cfg.get('seed',20261012))).shuffle(groups)
    held = set(groups[:min(len(groups)-1,max(1,round(len(groups)*ratio)))])
    output.mkdir(parents=True)
    dump(output/'status.json', {'ready':False,'reason':'Подготовка ещё не завершена.'})
    manifest = []
    seen = set()
    for r in report['records']:
        if r['sha256'] in seen:
            continue
        seen.add(r['sha256'])
        split = 'val' if r['group'] in held else 'train'
        stem = r['sha256'][:24]
        image = output/'images'/split/(stem + Path(r['image']).suffix.lower())
        image.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(r['image'],image)
        if digest(image)!=r['sha256']:
            raise ValueError('Исходное изображение изменилось во время подготовки: '+r['image'])
        kind = 'masks' if cfg['format']=='semantic_masks' else 'labels'
        label = output/kind/split/(stem + ('.png' if kind=='masks' else '.txt'))
        label.parent.mkdir(parents=True,exist_ok=True)
        if kind=='masks':
            Image.fromarray(decode_mask(r['annotation'],cfg['mask_mapping'],len(cfg['classes']))).save(label)
        else:
            annotation_rows=r['segments'] if cfg['format']=='yolo_segment' else r['boxes']
            label.write_text('\n'.join(str(b[0])+' '+' '.join(f'{v:.10f}' for v in b[1:]) for b in annotation_rows)+'\n' if annotation_rows else '',encoding='utf-8')
        manifest.append({**r,'split':split,'prepared_image':image.relative_to(output).as_posix(),
                         'prepared_annotation':label.relative_to(output).as_posix()})
    counts = collections.Counter(r['split'] for r in manifest)
    # A class present only in train/val makes validation misleading. Report rather than invent a split.
    coverage = {}
    for split in ('train','val'):
        c = collections.Counter()
        for r in manifest:
            if r['split'] != split: continue
            if cfg['format']=='semantic_masks':
                with Image.open(output/r['prepared_annotation']) as im: arr=np.array(im)
                c.update(int(v) for v in np.unique(arr) if v!=255)
            else: c.update(int(b[0]) for b in r.get('segments',r.get('boxes',[])))
        coverage[split] = {str(i):c[i] for i in range(len(cfg['classes']))}
    dataset = {'version':1,'task':'semantic' if cfg['format']=='semantic_masks' else 'segment' if cfg['format']=='yolo_segment' else 'detect',
               'classes':cfg['classes'],'ignore_index':255,'counts':dict(counts),'class_image_coverage':coverage,
               'seed':int(cfg.get('seed',20261012)),'source_configuration':cfg,
               'warning':'Group splitting is only as reliable as supplied scene IDs. No independent test split has been created.'}
    dump(output/'audit.json',report); dump(output/'manifest.json',manifest); dump(output/'dataset.json',dataset)
    if dataset['task'] in ('detect','segment'):
        import yaml
        (output/'data.yaml').write_text(yaml.safe_dump({'path':str(output).replace('\\','/'),'train':'images/train','val':'images/val',
            'nc':len(cfg['classes']),'names':dict(enumerate(cfg['classes']))},allow_unicode=True,sort_keys=False),encoding='utf-8')
    dump(output/'status.json',{'ready':True,'task':dataset['task'],'counts':dict(counts),'training_started':False})
    make_launchers(output,dataset)
    return dataset

def make_launchers(output,dataset):
    script = Path(__file__).resolve().parent/'cli.py'
    settings={'epochs':30,'image_size':512 if dataset['task']=='semantic' else 960,'batch':2,'lr':0.001,
              'device':'cpu','workers':0,'seed':dataset['seed'],'checkpoint':'','task':dataset['task']}
    dump(output/'training.json',settings)
    for name,command in [('01_CHECK','check'),('02_PILOT','pilot'),('03_TRAIN','train'),('04_RESUME','resume')]:
        # UTF-8 BAT switches codepage before reading commands containing Russian paths.
        text='@echo off\nchcp 65001 >nul\nsetlocal\nset "PYTHONUTF8=1"\n'
        text+='if not defined PANOPTR_PYTHON set "PANOPTR_PYTHON='+str(Path(sys_executable()))+'"\n'
        text+='"%PANOPTR_PYTHON%" "'+str(script)+'" '+command+' --dataset "'+str(output)+'"\n'
        text+='set "LAB_EXIT=%errorlevel%"\nif not defined PANOPTR_NO_PAUSE pause\nexit /b %LAB_EXIT%\n'
        (output/(name+'.bat')).write_bytes(text.replace('\n','\r\n').encode('utf-8'))

def sys_executable():
    import sys
    return sys.executable

def verify_prepared(dataset):
    root=Path(dataset).resolve(); meta=read(root/'dataset.json'); rows=read(root/'manifest.json')
    if not read(root/'status.json')['ready']: raise ValueError('Набор не готов.')
    hashes=collections.defaultdict(set); groups=collections.defaultdict(set); counts=collections.Counter()
    for r in rows:
        p=under(root,r['prepared_image']); lab=under(root,r['prepared_annotation'])
        with Image.open(p) as im: im.load(); size=im.size
        if digest(p)!=r['sha256']: raise ValueError('Изображение изменилось: '+str(p))
        if meta['task']=='semantic':
            with Image.open(lab) as im: im.load(); arr=np.array(im)
            if im.size!=size or arr.ndim!=2 or any(int(v)!=255 and not 0<=int(v)<len(meta['classes']) for v in np.unique(arr)):
                raise ValueError('Некорректная подготовленная маска: '+str(lab))
            if hashlib.sha256(arr.tobytes()).hexdigest()!=r['annotation_sha256']: raise ValueError('Маска изменена: '+str(lab))
        elif meta['task']=='segment':
            parsed=segments_for({'annotation':str(lab)},{'classes':meta['classes']})
            if len(parsed)!=len(r['segments']) or any(not np.allclose(a,b,rtol=0,atol=1e-8) for a,b in zip(parsed,r['segments'])):
                raise ValueError('Полигон изменён: '+str(lab))
        else:
            parsed=boxes_for({'annotation':str(lab)}, {'format':'yolo_bbox','classes':meta['classes']},*size)
            if len(parsed)!=len(r['boxes']) or not np.allclose(np.asarray(parsed),np.asarray(r['boxes']),rtol=0,atol=1e-8):
                raise ValueError('Разметка изменена: '+str(lab))
        hashes[r['sha256']].add(r['split']); groups[r['group']].add(r['split']); counts[r['split']]+=1
    if any(len(v)>1 for v in hashes.values()) or any(len(v)>1 for v in groups.values()): raise ValueError('Утечка между train/val.')
    if dict(counts)!=meta['counts'] or not counts['train'] or not counts['val']: raise ValueError('Несогласованные выборки.')
    return {'pass':True,'task':meta['task'],'counts':dict(counts),'classes':meta['classes'],'gpu_used':False}
