from dataset_lab.bootstrap import setup
setup()
import json,sys
from pathlib import Path
import numpy as np
from final_core import read_image,save_json
from path_segmenter import PathSegmenter
from path_geometry import serializable_geometry
from path_benchmark import compare
if __name__=='__main__':
    config=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    if config['action']=='segment':
        import torch
        torch.set_num_threads(2); frame=read_image(config['image']); g=PathSegmenter().infer(frame,config['model'],1,1,0.,config.get('source_id',config['image']),config.get('confidence',.35),config.get('imgsz',640))
        out=Path(config['output']); out.mkdir(parents=True,exist_ok=False)
        arrays={f'semantic_{k}':v for k,v in g.get('semantic_masks',{}).items()}; arrays.update({f"candidate_{c['id']}":c['mask'] for c in g['candidates']}); np.savez_compressed(out/'masks.npz',**arrays)
        save_json(out/'geometry.json',serializable_geometry(g)); print(json.dumps({'status':g['status'],'device':'cpu','output':str(out)},ensure_ascii=False))
    elif config['action']=='compare':
        r=compare(json.loads(Path(config['input']).read_text(encoding='utf-8')),config['output']); print(json.dumps(r['metrics'],ensure_ascii=False))
    else: raise ValueError('Неизвестный инструмент.')
