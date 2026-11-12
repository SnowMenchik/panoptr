"""Local adapters; no packages/weights downloaded implicitly at inference time."""
from pathlib import Path
import importlib.util,json,os,hashlib
import numpy as np
from PIL import Image
from core import read,dump,digest
from annotation import project,frame,save_frame,add_object,all_frames,mask_bbox

MODES={
 'yolo_bbox':'YOLO / RT-DETR — рамки', 'yolo_segment':'YOLO segmentation — маски',
 'detector_sam':'Детектор → SAM / SAM 2 / MobileSAM — маски',
 'sam_prompt':'SAM / SAM 2 / SAM 3 / MobileSAM — точки и рамка',
 'sam_everything':'SAM / SAM 2 — предложения всех масок',
 'fastsam_prompt':'FastSAM — точки и рамка', 'fastsam_everything':'FastSAM — все маски',
 'fastsam_text':'FastSAM — текстовая подсказка',
 'world_text':'YOLO-World — классы по тексту',
 'yoloe_text':'YOLOE — классы по тексту', 'yoloe_visual':'YOLOE — пример объекта',
 'yoloe_free':'YOLOE prompt-free — обнаружение известных понятий',
 'sam3_text':'SAM 3/3.1 — понятия по тексту', 'sam3_exemplar':'SAM 3/3.1 — примеры рамками',
 'semantic_model':'U-Net лаборатории — попиксельные предложения',
 'sam2_video':'SAM 2 — распространение масок по видео',
 'sam3_video':'SAM 3 — распространение масок по видео',
 'sam3_video_text':'SAM 3 — понятия по видео',
}

def asset_root(cfg):
    return Path(cfg.get('assets_dir') or Path(__file__).resolve().parents[3]/'10_Лаборатория_данных/annotation_models').resolve()

def offline_clip(cfg):
    import clip
    if not callable(getattr(clip,'load',None)) or not hasattr(clip,'clip'):
        raise ValueError('Локальный CLIP неполон или недоступен для чтения. Проверьте файлы и права доступа в dataset_lab/annotation_vendor/clip; текстовый режим не запущен.')
    original=getattr(clip.load,'_panoptr_original',clip.load); folder=asset_root(cfg)/'clip'
    def local_load(name,*args,**kwargs):
        if name in clip.clip._MODELS:
            local=folder/Path(clip.clip._MODELS[name]).name
            if not local.is_file(): raise ValueError('Нет локального CLIP: '+str(local)+'. Выполните отдельную загрузку текстовых весов.')
            name=str(local)
        return original(name,*args,**kwargs)
    local_load._panoptr_original=original; clip.load=local_load

def options(cfg):
    return {'device':cfg.get('device','cpu').replace('cuda:',''),'imgsz':int(cfg.get('image_size',640)),
            'conf':float(cfg.get('confidence',.25)),'iou':float(cfg.get('iou',.7)),
            'max_det':int(cfg.get('max_objects',100)),'verbose':False,'save':False}

def preflight(cfg):
    mode=cfg['mode']
    if mode not in MODES: raise ValueError('Неизвестный вариант разметки.')
    if cfg.get('device','cpu') not in ('cpu','cuda:0','cuda:1'): raise ValueError('Неверное устройство.')
    if not 0<=float(cfg.get('confidence',.25))<=1 or not 32<=int(cfg.get('image_size',640))<=2048: raise ValueError('Неверный порог/размер.')
    fields=['model']+(['sam_model'] if mode=='detector_sam' else [])
    missing=[f for f in fields if not cfg.get(f) or not Path(cfg[f]).is_file()]
    if missing: raise ValueError('Нет локальных весов: '+', '.join(missing)+'. Выберите или скачайте их отдельно; автоскачивание при запуске отключено.')
    if mode in ('world_text','yoloe_text','fastsam_text','sam3_text','sam3_video_text'):
        if not cfg.get('texts'): raise ValueError('Нужны текстовые названия объектов.')
        if not importlib.util.find_spec('clip'): raise ValueError('Текстовый режим требует локального CLIP. Установите зависимости разметки отдельно от текущего обучения.')
        required=asset_root(cfg)/('mobileclip2_b.ts' if mode=='yoloe_text' else 'clip/ViT-B-32.pt')
        if mode in ('world_text','yoloe_text','fastsam_text') and not required.is_file(): raise ValueError('Нет локального текстового энкодера: '+str(required))
    if mode=='yoloe_visual' and (not cfg.get('reference_image') or not Path(cfg['reference_image']).is_file() or not cfg.get('visual_boxes')):
        raise ValueError('Нужен файл примера и рамки visual_boxes с class_id.')
    if mode.endswith('video') or mode=='sam3_video_text':
        if not cfg.get('video') or not Path(cfg['video']).is_file(): raise ValueError('Выберите локальное видео.')
        if 'sam3' in mode and 'sam3.1' in Path(cfg['model']).name: raise ValueError('SAM 3.1 multiplex не поддерживается установленным Ultralytics для видео; используйте sam3.pt.')
    # Keep Ultralytics auto-install off in this worker, independent from the training runtime.
    os.environ['YOLO_AUTOINSTALL']='false'
    return {'configuration_pass':True,'mode':mode,'weights':{f:digest(cfg[f]) for f in fields},'gpu_used':False}

def original_masks(result):
    if result.masks is None: return []
    masks=result.masks.data
    if tuple(masks.shape[-2:])!=tuple(result.orig_shape):
        from ultralytics.utils.ops import scale_masks
        masks=scale_masks(masks[None].float(),result.orig_shape)[0]
    return [m.detach().cpu().numpy()>.5 for m in masks]

def convert_result(result,cfg,classes,force_class=None,source='model'):
    masks=original_masks(result); boxes=result.boxes; proposals=[]
    for i in range(len(masks) if masks else len(boxes) if boxes is not None else 0):
        cls=int(boxes.cls[i].item()) if boxes is not None and i<len(boxes) else None
        names=result.names
        name=str(names.get(cls,'')) if cls is not None and isinstance(names,dict) else str(names[cls]) if cls is not None and 0<=cls<len(names) else ''
        mapping=cfg.get('class_map',{})
        target=force_class if force_class is not None else mapping.get(name,mapping.get(str(cls)))
        # Names identical to user-defined target names are an explicit visible mapping default.
        if target is None and name in classes: target=classes.index(name)
        if target is None and cfg['mode'] in ('yolo_bbox','yolo_segment','detector_sam','world_text','yoloe_text','yoloe_visual','yoloe_free','sam3_text','sam3_video_text') and not cfg.get('keep_unknown',False):
            continue
        if target is not None and (not isinstance(target,int) or not 0<=target<len(classes)): raise ValueError('Неверное соответствие классов модели.')
        mask=masks[i] if masks else None
        if mask is not None and not mask.any(): continue
        bbox=mask_bbox(mask) if mask is not None else boxes.xyxy[i].detach().cpu().tolist()
        score=float(boxes.conf[i].item()) if boxes is not None and i<len(boxes) else None
        if score is not None and score<float(cfg.get('confidence',.25)): continue
        proposals.append({'class_id':target,'bbox':bbox,'mask':mask,'confidence':score,'source':source,
                          'extra':{'source_class_id':cls,'source_class_name':name}})
    return proposals

class Adapter:
    def __init__(self,cfg,classes):
        self.cfg=cfg; self.classes=classes; self.mode=cfg['mode']; self.provenance=preflight(cfg); self.model=None
        import torch
        torch.set_num_threads(2)
        from ultralytics import YOLO,YOLOWorld,YOLOE,SAM,FastSAM,RTDETR
        constructors={'world_text':YOLOWorld,'yoloe_text':YOLOE,'yoloe_visual':YOLOE,'yoloe_free':YOLOE,
            'sam_prompt':SAM,'sam_everything':SAM,'fastsam_prompt':FastSAM,'fastsam_everything':FastSAM,'fastsam_text':FastSAM}
        if self.mode.startswith('sam3_') or self.mode=='sam2_video':
            from ultralytics.models import sam
            cls={'sam3_text':'SAM3SemanticPredictor','sam3_exemplar':'SAM3SemanticPredictor','sam2_video':'SAM2VideoPredictor',
                 'sam3_video':'SAM3VideoPredictor','sam3_video_text':'SAM3VideoSemanticPredictor'}[self.mode]
            self.model=getattr(sam,cls)(overrides={**options(cfg),'imgsz':int(cfg.get('sam_image_size',1024)),'task':'segment','mode':'predict','model':cfg['model']})
        elif self.mode=='semantic_model':
            from semantic import load_checkpoint,device_for
            self.model,self.checkpoint=load_checkpoint(cfg['model'],device_for(cfg.get('device','cpu')))
            if self.checkpoint['classes']!=['background',*classes]: raise ValueError('Семантические классы должны быть background + классы проекта.')
        else:
            constructor=constructors.get(self.mode,RTDETR if 'rtdetr' in Path(cfg['model']).name.lower() else YOLO)
            self.model=constructor(cfg['model'])
            if self.mode=='detector_sam':
                if self.model.task!='detect': raise ValueError('Связке нужен детектор bbox.')
                self.sam=SAM(cfg['sam_model'])
            if self.mode=='yolo_bbox' and self.model.task!='detect': raise ValueError('Выбраны рамки: нужен детектор.')
            if self.mode=='yolo_segment' and self.model.task!='segment': raise ValueError('Выбраны маски: нужны веса YOLO-seg.')
            if self.mode in ('world_text','yoloe_text'):
                offline_clip(cfg)
                # MobileCLIPTS resolves its published asset name relative to cwd.
                previous=Path.cwd()
                try:
                    os.chdir(asset_root(cfg)); self.model.set_classes(cfg['texts'])
                finally: os.chdir(previous)
            if self.mode=='fastsam_text': offline_clip(cfg)

    def predict(self,image,prompt=None):
        cfg=self.cfg; opts=options(cfg); prompt=prompt or {}; mode=self.mode
        if mode in ('sam_prompt','sam_everything'): opts['imgsz']=int(cfg.get('sam_image_size',1024))
        force=prompt.get('class_id') if mode in ('sam_prompt','fastsam_prompt','sam3_exemplar') else None
        if mode in ('sam_prompt','fastsam_prompt'):
            if not prompt.get('points') and not prompt.get('boxes'): raise ValueError('Поставьте положительную/отрицательную точку или рамку.')
            kwargs={k:v for k,v in {'points':prompt.get('points'),'labels':prompt.get('labels'),'bboxes':prompt.get('boxes')}.items() if v is not None}
            result=self.model.predict(str(image),**kwargs,**opts)[0]
        elif mode=='detector_sam':
            detector=self.model.predict(str(image),**opts)[0]; detections=convert_result(detector,cfg,self.classes)
            proposals=[]
            # Process each box separately: do not assume SAM output order equals detector class order.
            for d in detections:
                result=self.sam.predict(str(image),bboxes=[d['bbox']],**{**opts,'imgsz':int(cfg.get('sam_image_size',1024))})[0]
                masks=original_masks(result)
                if masks:
                    proposals.append({**d,'bbox':mask_bbox(masks[0]),'mask':masks[0],'source':'detector_sam'})
            return proposals
        elif mode in ('sam3_text','sam3_exemplar'):
            self.model.set_image(str(image))
            kwargs={'text':cfg['texts']} if mode=='sam3_text' else {'bboxes':prompt.get('boxes',cfg.get('exemplar_boxes',[]))}
            if not next(iter(kwargs.values())): raise ValueError('Нужны текст или примеры рамками.')
            result=self.model(**kwargs)[0]
        elif mode=='fastsam_text': result=self.model.predict(str(image),texts=cfg['texts'],**opts)[0]
        elif mode=='yoloe_visual':
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
            boxes=cfg['visual_boxes']; targets=sorted({int(b['class_id']) for b in boxes}); to_local={v:i for i,v in enumerate(targets)}
            visual={'bboxes':np.array([b['bbox'] for b in boxes]),'cls':np.array([to_local[b['class_id']] for b in boxes])}
            result=self.model.predict(str(image),refer_image=cfg['reference_image'],visual_prompts=visual,predictor=YOLOEVPSegPredictor,**opts)[0]
            # Visual class IDs refer directly to project classes, not autogenerated objectN names.
            cfg={**cfg,'class_map':{str(i):target for i,target in enumerate(targets)}}
        elif mode=='semantic_model':
            from semantic import predict_image,device_for
            import cv2
            with Image.open(image) as im: original=im.convert('RGB')
            arr=predict_image(self.model,original,self.checkpoint['settings']['image_size'],device_for(cfg.get('device','cpu')))
            proposals=[]
            for c in range(1,len(self.checkpoint['classes'])):
                n,components=cv2.connectedComponents((arr==c).astype(np.uint8))
                for i in range(1,n):
                    mask=components==i
                    if int(mask.sum())>=int(cfg.get('min_area',16)):
                        proposals.append({'class_id':c-1,'bbox':mask_bbox(mask),'mask':mask,'confidence':None,'source':'semantic_model','extra':{}})
            return proposals
        elif mode=='sam_everything':
            # A modest point grid is configurable and avoids an unexpectedly huge CPU job.
            if self.model.predictor is None:
                self.model.predictor=self.model._smart_load('predictor')(overrides={**self.model.overrides,**opts,'task':'segment','mode':'predict'},_callbacks=self.model.callbacks)
                self.model.predictor.setup_model(model=self.model.model,verbose=False)
                from functools import partial
                self.model.predictor.generate=partial(self.model.predictor.generate,points_stride=int(cfg.get('grid_points',8)))
            result=self.model.predict(str(image),**opts)[0]
        else: result=self.model.predict(str(image),**opts)[0]
        proposals=convert_result(result,cfg,self.classes,force,source=mode)
        if mode in ('sam_everything','fastsam_everything','fastsam_text'):
            for p in proposals: p['class_id']=None  # Geometry proposal alone does not establish its semantic class.
        return proposals[:int(cfg.get('max_objects',100))]

def apply(root,identifier,proposals,provenance,replace=False):
    data=frame(root,identifier)
    if data['status'] in ('confirmed','edited','rejected'): raise ValueError('Проверенную/исправленную/отклонённую разметку автоматически не заменяем.')
    if data['inference_complete'] and not replace: return False
    data['objects']=[]
    for p in proposals: add_object(root,data,**p)
    data['status']='unreviewed'; data['inference_complete']=True; data['model_provenance']=provenance
    save_frame(root,data,'model_proposals'); return True

def run_batch(root,cfg,limit=0,identifier=None,prompt=None):
    if identifier and cfg['mode'] not in ('sam_prompt','fastsam_prompt','sam3_exemplar'):
        raise ValueError('Уточнение выбранного объекта требует режима SAM/FastSAM по точкам и рамке либо SAM 3 по примеру.')
    meta=project(root); adapter=Adapter(cfg,meta['classes']); rows=[frame(root,identifier)] if identifier else all_frames(root)
    todo=rows if identifier else [r for r in rows if r['status']=='unreviewed' and not r['inference_complete']]
    if limit: todo=todo[:limit]
    done=0
    for r in todo:
        if digest(r['image'])!=r['image_sha256']: raise ValueError('Исходное изображение изменилось.')
        proposals=adapter.predict(r['image'],prompt)
        # Interactive refinement updates the selected object and retains other reviewed objects.
        if identifier:
            data=frame(root,r['id']); target=(prompt or {}).get('replace_object')
            if target: data['objects']=[o for o in data['objects'] if o['id']!=target]
            for p in proposals: add_object(root,data,**p)
            data['status']='edited'; data['model_provenance']=adapter.provenance; save_frame(root,data,'interactive_model_refinement')
        else: apply(root,r['id'],proposals,adapter.provenance,replace=bool(identifier))
        done+=1; print('Размечено',done,'/',len(todo),flush=True)
    return {'processed':done,'mode':cfg['mode'],'all_results_unreviewed':not bool(identifier),'human_confirmation_required':True,'training_started':False}

def video_project(cfg,output,classes,limit=300,every=1):
    import cv2
    root=Path(output).resolve()
    if root.exists(): raise ValueError('Для видео нужна новая папка проекта.')
    if every<1 or limit<1: raise ValueError('Шаг и предел должны быть положительными.')
    adapter=Adapter(cfg,classes); opts={'source':cfg['video'],'stream':True}
    if cfg['mode']=='sam3_video_text': opts['text']=cfg['texts']
    else:
        opts.update({k:v for k,v in {'points':cfg.get('points'),'labels':cfg.get('point_labels'),'bboxes':cfg.get('boxes')}.items() if v is not None})
        if not cfg.get('points') and not cfg.get('boxes'): raise ValueError('Для видео нужны точки или рамки первого кадра.')
    group='video_'+digest(cfg['video'])[:16]; source=root.parent/(root.name+'_frames')
    if source.exists(): raise ValueError('Папка извлечённых кадров уже существует.')
    folder=source/group; folder.mkdir(parents=True); frames=[]; results=adapter.model(**opts)
    try:
        for index,result in enumerate(results):
            if index%every: continue
            image=folder/f'frame_{index:08d}.png'; Image.fromarray(cv2.cvtColor(result.orig_img,cv2.COLOR_BGR2RGB)).save(image)
            proposals=convert_result(result,cfg,classes,force_class=cfg.get('prompt_class') if cfg['mode']!='sam3_video_text' else None,source=cfg['mode'])
            frames.append((image,proposals,index))
            if len(frames)>=limit: break
    finally:
        if hasattr(results,'close'): results.close()
    if not frames: raise ValueError('Не получено кадров видео.')
    create=__import__('annotation').create_project(source,root,classes)
    by_path={r['image']:r for r in all_frames(root)}
    for image,proposals,index in frames:
        r=by_path[str(image)]; apply(root,r['id'],proposals,{**adapter.provenance,'source_video':cfg['video'],'frame_index':index})
    return {**create,'mode':cfg['mode'],'video_frames':len(frames),'all_results_unreviewed':True}
