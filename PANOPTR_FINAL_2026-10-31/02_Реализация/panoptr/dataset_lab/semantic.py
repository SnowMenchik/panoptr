"""Small semantic baseline. Masks stay pixel masks, never bounding boxes."""
import csv
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from core import read, dump, verify_prepared

FORMAT = 'panoptr_semantic_unet_v1'

class Block(nn.Module):
    def __init__(self,a,b):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU(),
                               nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU())
    def forward(self,x): return self.net(x)

class SmallUNet(nn.Module):
    def __init__(self,nc):
        super().__init__(); self.a=Block(3,16); self.b=Block(16,32); self.c=Block(32,64)
        self.d=Block(96,32); self.e=Block(48,16); self.head=nn.Conv2d(16,nc,1)
    def forward(self,x):
        a=self.a(x); b=self.b(F.max_pool2d(a,2)); c=self.c(F.max_pool2d(b,2))
        d=self.d(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        return self.head(self.e(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))

def letterbox(image,mask,size):
    w,h=image.size; scale=min(size/w,size/h); nw=max(1,round(w*scale)); nh=max(1,round(h*scale))
    left=(size-nw)//2; top=(size-nh)//2
    canvas=Image.new('RGB',(size,size),(0,0,0)); canvas.paste(image.resize((nw,nh),Image.Resampling.BILINEAR),(left,top))
    target=np.full((size,size),255,dtype=np.int64)
    if mask is not None:
        target[top:top+nh,left:left+nw]=np.asarray(Image.fromarray(mask).resize((nw,nh),Image.Resampling.NEAREST))
    tensor=torch.from_numpy(np.array(canvas,dtype=np.float32).transpose(2,0,1).copy())/255.0
    return tensor,torch.from_numpy(target),(left,top,nw,nh)

class SegData(Dataset):
    def __init__(self,root,rows,size): self.root=Path(root); self.rows=rows; self.size=size
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(self.root/r['prepared_image']) as im: image=im.convert('RGB')
        with Image.open(self.root/r['prepared_annotation']) as im: mask=np.array(im)
        x,y,_=letterbox(image,mask,self.size)
        if random.random()<.5: x=x.flip(-1); y=y.flip(-1)
        return x,y

def metrics(matrix,classes):
    result=[]; tp=np.diag(matrix); union=matrix.sum(0)+matrix.sum(1)-tp
    for i,name in enumerate(classes):
        denominator=matrix[i].sum()+matrix[:,i].sum()
        result.append({'class_id':i,'class_name':name,'iou':float(tp[i]/union[i]) if union[i] else None,
                       'dice':float(2*tp[i]/denominator) if denominator else None,'gt_pixels':int(matrix[i].sum())})
    ious=[r['iou'] for r in result if r['iou'] is not None]
    return {'mean_iou':float(np.mean(ious)) if ious else None,'pixel_accuracy':float(tp.sum()/matrix.sum()) if matrix.sum() else None,
            'per_class':result,'confusion_matrix':matrix.tolist(),'absent_classes_excluded_from_mean':True}

def load_checkpoint(path,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('format')!=FORMAT: raise ValueError('Нужен checkpoint семантической модели лаборатории, а не YOLO .pt.')
    model=SmallUNet(len(checkpoint['classes'])); model.load_state_dict(checkpoint['model']); model.to(device)
    return model,checkpoint

def device_for(value):
    if value=='cpu': return torch.device('cpu')
    if value not in ('cuda:0','cuda:1'): raise ValueError('Допустимы cpu, cuda:0, cuda:1.')
    if not torch.cuda.is_available(): raise ValueError('CUDA недоступна.')
    return torch.device(value)

def predict_image(model,image,size,device):
    tensor,_,(left,top,nw,nh)=letterbox(image,None,size)
    with torch.inference_mode():
        logits=model(tensor.unsqueeze(0).to(device))[:,:,top:top+nh,left:left+nw]
        logits=F.interpolate(logits,size=(image.height,image.width),mode='bilinear',align_corners=False)
        return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

def evaluate(model,root,rows,classes,size,device):
    matrix=np.zeros((len(classes),len(classes)),dtype=np.int64); model.eval()
    for i,r in enumerate(rows):
        with Image.open(Path(root)/r['prepared_image']) as im: image=im.convert('RGB')
        with Image.open(Path(root)/r['prepared_annotation']) as im: target=np.array(im)
        prediction=predict_image(model,image,size,device); keep=target!=255
        matrix+=np.bincount(target[keep].astype(np.int64)*len(classes)+prediction[keep],minlength=len(classes)**2).reshape(matrix.shape)
        if i%25==0: print('Оценка',i+1,'/',len(rows),flush=True)
    return metrics(matrix,classes)

def save_checkpoint(path,value):
    path=Path(path); temporary=path.with_suffix('.tmp'); torch.save(value,temporary); temporary.replace(path)

def train(root,settings,run,pilot=False,resume=False):
    root=Path(root); run=Path(run); meta=read(root/'dataset.json'); rows=read(root/'manifest.json')
    torch.set_num_threads(2); seed=int(settings['seed']); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=device_for(settings['device']); size=int(settings['image_size'])
    if size<32 or size>2048 or size%4: raise ValueError('Размер должен быть кратным 4, от 32 до 2048.')
    classes=meta['classes']; train_rows=[r for r in rows if r['split']=='train']; val_rows=[r for r in rows if r['split']=='val']
    if pilot: train_rows=train_rows[:16]; val_rows=val_rows[:16]
    start=0; best=-1.0; source=settings.get('checkpoint','')
    if resume: source=str(run/'last.pt')
    if source:
        model,checkpoint=load_checkpoint(source,device)
        if checkpoint['classes']!=classes: raise ValueError('Классы checkpoint не совпадают с датасетом.')
        if resume:
            if checkpoint['dataset_signature']!=dataset_signature(root): raise ValueError('Датасет отличается от сохранённого запуска.')
            if checkpoint['settings']!=settings: raise ValueError('Настройки изменены; используйте новый запуск с checkpoint для дообучения.')
            start=int(checkpoint['epoch'])+1; best=float(checkpoint['best_iou'])
    else: model=SmallUNet(len(classes)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(settings['lr']),weight_decay=0.0001)
    if resume: optimizer.load_state_dict(checkpoint['optimizer'])
    epochs=1 if pilot else int(settings['epochs'])
    if start>=epochs: raise ValueError('Этот запуск уже достиг указанного количества эпох.')
    loader=DataLoader(SegData(root,train_rows,size),batch_size=int(settings['batch']),shuffle=True,num_workers=0)
    criterion=nn.CrossEntropyLoss(ignore_index=255); run.mkdir(parents=True,exist_ok=resume)
    dump(run/'run.json',{'settings':settings,'classes':classes,'pilot':pilot,'dataset':str(root.resolve()),
                         'initialization':'fine_tune' if source and not resume else 'resume' if resume else 'from_scratch',
                         'warning':'Small UNet is a baseline, not a pretrained railway model.'})
    log=run/'epochs.csv'
    with log.open('a' if resume else 'w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        if not resume: writer.writerow(['epoch','train_loss','val_mean_iou','val_pixel_accuracy'])
        for epoch in range(start,epochs):
            # Makes epoch-boundary resume reproducible without storing arbitrary RNG pickle objects.
            random.seed(seed+epoch); torch.manual_seed(seed+epoch)
            model.train(); losses=[]
            for x,y in loader:
                x=x.to(device); y=y.to(device)
                if not (y!=255).any(): continue
                optimizer.zero_grad(set_to_none=True); loss=criterion(model(x),y)
                if not torch.isfinite(loss): raise ValueError('Loss не является конечным числом.')
                loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            if not losses: raise ValueError('После масштабирования не осталось размеченных пикселей.')
            measured=evaluate(model,root,val_rows,classes,size,device); score=measured['mean_iou']
            if score is None: raise ValueError('Нет пикселей для оценки.')
            improved=score>best; best=max(best,score)
            saved={'format':FORMAT,'classes':classes,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                   'epoch':epoch,'best_iou':best,'settings':settings,'dataset_signature':dataset_signature(root),'metrics':measured}
            save_checkpoint(run/'last.pt',saved)
            if improved: save_checkpoint(run/'best.pt',saved)
            writer.writerow([epoch+1,float(np.mean(losses)),score,measured['pixel_accuracy']]); stream.flush()
            print(f'Эпоха {epoch+1}/{epochs}: loss={np.mean(losses):.5f}; val mIoU={score:.4f}',flush=True)
    dump(run/'completed.json',{'best_checkpoint':str((run/'best.pt').resolve()),'best_validation_mean_iou':best,
                               'pilot':pilot,'independent_test':False,'application_weights_replaced':False})
    return read(run/'completed.json')

def dataset_signature(root):
    import hashlib
    return hashlib.sha256((Path(root)/'manifest.json').read_bytes()+(Path(root)/'dataset.json').read_bytes()).hexdigest()

def compare(root,first,second,device_name='cpu',output=None):
    verify_prepared(root); meta=read(Path(root)/'dataset.json'); rows=[r for r in read(Path(root)/'manifest.json') if r['split']=='val']
    device=device_for(device_name); torch.set_num_threads(2); results=[]
    for path in (first,second):
        model,ck=load_checkpoint(path,device)
        if ck['classes']!=meta['classes']: raise ValueError('Классы моделей и набора должны совпадать.')
        results.append({'checkpoint':str(path),'image_size':ck['settings']['image_size'],
                        'metrics':evaluate(model,root,rows,meta['classes'],ck['settings']['image_size'],device)})
        del model
    result={'split':'val','same_images':len(rows),'models':results,'independent_test':False,
            'mean_iou_delta':results[1]['metrics']['mean_iou']-results[0]['metrics']['mean_iou'],
            'note':'Inference sizes come from each checkpoint and are reported; compare accuracy and resource cost together.'}
    if output:
        if Path(output).exists(): raise ValueError('Отчёт уже существует.')
        dump(output,result)
    return result

def export_predictions(checkpoint,images,output,device_name='cpu'):
    from core import files, IMAGE_EXTS
    root=Path(images).resolve(); output=Path(output).resolve()
    if output.exists() or Path(str(output)+'.zip').exists(): raise ValueError('Экспорт требует новой папки и нового имени ZIP.')
    if output.is_relative_to(root): raise ValueError('Экспорт не должен находиться внутри входных изображений.')
    paths=files(root,IMAGE_EXTS)
    if not paths: raise ValueError('Нет изображений для экспорта.')
    names=[p.relative_to(root).with_suffix('.png').as_posix().casefold() for p in paths]
    if len(names)!=len(set(names)): raise ValueError('Имена масок совпадут после замены расширения на PNG.')
    device=device_for(device_name); torch.set_num_threads(2); model,ck=load_checkpoint(checkpoint,device); model.eval()
    output.mkdir(parents=True); dump(output/'status.json',{'ready':False})
    rows=[]
    for i,p in enumerate(paths):
        with Image.open(p) as im:
            if im.getexif().get(274,1)!=1: raise ValueError('EXIF-поворот требует обработки до экспорта: '+str(p))
            image=im.convert('RGB')
        prediction=predict_image(model,image,ck['settings']['image_size'],device)
        target=output/'masks'/p.relative_to(root).with_suffix('.png'); target.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(prediction).save(target)
        rows.append({'image':p.relative_to(root).as_posix(),'mask':target.relative_to(output).as_posix(),
                     'width':image.width,'height':image.height})
        print('Экспорт',i+1,'/',len(paths),flush=True)
    dump(output/'manifest.json',rows); dump(output/'classes.json',{'classes':ck['classes'],'encoding':'class-index uint8 PNG, model class IDs; not source RGB colors'})
    with (output/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['image','mask','width','height']); writer.writeheader(); writer.writerows(rows)
    dump(output/'status.json',{'ready':True,'images':len(rows)})
    shutil_archive(output)
    return {'output':str(output),'images':len(rows),'archive':str(output)+'.zip'}

def shutil_archive(output):
    import shutil
    archive=Path(str(output)+'.zip')
    if archive.exists(): raise ValueError('ZIP с таким именем уже существует; он не перезаписан.')
    shutil.make_archive(str(archive)[:-4],'zip',root_dir=output)
