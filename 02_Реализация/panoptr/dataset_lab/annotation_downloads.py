"""Explicit downloads to a separate folder; no training environment upgrades."""
from pathlib import Path
import json,hashlib,urllib.request,zipfile
from core import dump

PUBLIC_WEIGHTS=['mobile_sam.pt','sam2.1_t.pt','FastSAM-s.pt','yoloe-26s-seg.pt','yolov8s-worldv2.pt','yoloe-26s-seg-pf.pt','mobileclip2_b.ts']

def get(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Panoptr-Dataset-Lab'}),timeout=90)

def download_public(output,names=None):
    root=Path(output).resolve(); root.mkdir(parents=True,exist_ok=True); names=names or PUBLIC_WEIGHTS
    if any(n not in PUBLIC_WEIGHTS for n in names): raise ValueError('Неизвестное публичное имя весов.')
    with get('https://api.github.com/repos/ultralytics/assets/releases/tags/v8.4.0') as r: release=json.load(r)
    assets={a['name']:a for a in release['assets']}; result=[]
    # Some older model families remain in the official v8.3.0 asset release.
    missing=[n for n in names if n not in assets]
    if missing:
        with get('https://api.github.com/repos/ultralytics/assets/releases/tags/v8.3.0') as r: old=json.load(r)
        assets.update({a['name']:a for a in old['assets'] if a['name'] in missing})
    for name in names:
        a=assets.get(name)
        if not a: result.append({'name':name,'downloaded':False,'reason':'Not in official asset releases checked.'}); continue
        target=root/name
        if not target.exists():
            part=root/(name+'.part')
            with get(a['browser_download_url']) as response,part.open('wb') as stream:
                while True:
                    block=response.read(1024*1024)
                    if not block: break
                    stream.write(block)
            if part.stat().st_size!=a['size']: raise ValueError('Размер скачанных весов не совпадает с GitHub: '+name)
            part.replace(target)
        if target.stat().st_size!=a['size']: raise ValueError('Существующий файл имеет неверный размер: '+name)
        sha=hashlib.sha256(target.read_bytes()).hexdigest(); upstream=a.get('digest')
        if upstream and upstream.startswith('sha256:') and sha!=upstream.split(':',1)[1]: raise ValueError('SHA256 GitHub не совпал: '+name)
        result.append({'name':name,'downloaded':True,'bytes':target.stat().st_size,'sha256':sha,'url':a['browser_download_url'],
                       'upstream_digest':upstream,'license':'Upstream Ultralytics model/code conditions apply; not relicensed.'})
        print('Сохранено',name,flush=True)
    old=json.loads((root/'weights_provenance.json').read_text(encoding='utf-8')) if (root/'weights_provenance.json').exists() else []
    merged={r['name']:r for r in old}; merged.update({r['name']:r for r in result})
    dump(root/'weights_provenance.json',list(merged.values())); return result

def download_clip(output):
    from bootstrap import setup
    setup()
    import clip
    url=clip.clip._MODELS['ViT-B/32']; expected=url.split('/')[-2]
    root=Path(output).resolve()/'clip'; root.mkdir(parents=True,exist_ok=True); target=root/'ViT-B-32.pt'
    if not target.exists():
        part=root/'ViT-B-32.pt.part'
        with get(url) as response,part.open('wb') as stream:
            while True:
                block=response.read(1024*1024)
                if not block: break
                stream.write(block)
        sha=hashlib.sha256(part.read_bytes()).hexdigest()
        if sha!=expected: raise ValueError('SHA256 CLIP не совпал с опубликованным хэшем.')
        part.replace(target)
    sha=hashlib.sha256(target.read_bytes()).hexdigest()
    if sha!=expected: raise ValueError('Локальный CLIP повреждён.')
    report={'file':str(target),'url':url,'sha256':sha,'bytes':target.stat().st_size,'upstream_sha256_verified':True}
    dump(root/'provenance.json',report); return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--names',nargs='*'); p.add_argument('--clip',action='store_true'); a=p.parse_args()
    print(json.dumps(download_clip(a.output) if a.clip else download_public(a.output,a.names),ensure_ascii=False,indent=2))
