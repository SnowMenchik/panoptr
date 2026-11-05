"""Rehearsal fixture: good data and deliberately broken separate input."""
from pathlib import Path
import csv
import shutil
import numpy as np
from PIL import Image
from core import dump

def create_demo(output):
    root=Path(output).resolve()
    if root.exists(): raise ValueError('Для репетиции нужна новая папка.')
    for folder in ['good/images','good/masks']: (root/folder).mkdir(parents=True)
    rng=np.random.default_rng(20261012); pairs=[]
    for scene in range(6):
        for frame in range(2):
            name=f'scene{scene}_frame{frame}'; image=rng.integers(20,100,size=(48,64,3),dtype=np.uint8)
            mask=np.zeros((48,64),dtype=np.uint8); left=6+scene*3; mask[15:34,left:left+12]=1
            image[mask==1]=[200,100+frame*20,50]
            Image.fromarray(image).save(root/'good/images'/f'{name}.png')
            Image.fromarray(mask).save(root/'good/masks'/f'{name}_mask.png')
            pairs.append({'image':name+'.png','annotation':name+'_mask.png','group':'scene'+str(scene)})
    with (root/'good/pairs.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['image','annotation','group']); writer.writeheader(); writer.writerows(pairs)
    cfg={'format':'semantic_masks','images':str(root/'good/images'),'annotations':str(root/'good/masks'),
         'pairs_csv':str(root/'good/pairs.csv'),'mask_suffix':'_mask','classes':['background','rock'],
         'mask_mapping':{'0':0,'1':1},'independent_images':False,'group_regex':'','val_fraction':.33,'seed':20261012}
    dump(root/'good_config.json',cfg)
    shutil.copytree(root/'good',root/'broken')
    (root/'broken/masks/scene0_frame0_mask.png').unlink()
    Image.fromarray(np.zeros((24,32),dtype=np.uint8)).save(root/'broken/masks/scene1_frame0_mask.png')
    mask=np.full((48,64),17,dtype=np.uint8); Image.fromarray(mask).save(root/'broken/masks/scene2_frame0_mask.png')
    shutil.copy2(root/'broken/images/scene3_frame0.png',root/'broken/images/duplicate.png')
    Image.fromarray(np.zeros((48,64),dtype=np.uint8)).save(root/'broken/masks/duplicate_mask.png')
    bad={**cfg,'images':str(root/'broken/images'),'annotations':str(root/'broken/masks'),'pairs_csv':'',
         'group_regex':r'^(scene\d+|duplicate)'}
    dump(root/'broken_config.json',bad)
    return {'good_config':str(root/'good_config.json'),'broken_config':str(root/'broken_config.json'),
            'expected_problems':['missing mask','wrong dimensions','unknown class value 17','duplicate image with conflicting mask']}
