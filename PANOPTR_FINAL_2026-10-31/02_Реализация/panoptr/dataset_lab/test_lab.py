"""Meaningful CPU checks with temporary, deliberately broken fixture data."""
from bootstrap import setup
setup()
from pathlib import Path
import tempfile
import unittest
import json
import os
import numpy as np
from PIL import Image
from core import audit,prepare,read,dump,decode_mask,verify_prepared,digest
from demo import create_demo

def temp_dir():
    base=(Path(__file__).resolve().parent/'.test_tmp').resolve()
    assert base.is_relative_to(Path(__file__).resolve().parent)
    base.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base)

class DataChecks(unittest.TestCase):
    def setUp(self):
        self.temp=temp_dir(); self.root=Path(self.temp.name)
        self.fixture=create_demo(self.root/'fixture'); self.cfg=read(self.fixture['good_config'])
    def tearDown(self): self.temp.cleanup()
    def test_rehearsal_detects_real_errors(self):
        good=audit(self.cfg); self.assertTrue(good['ready'],good['issues'])
        bad=audit(read(self.fixture['broken_config'])); self.assertFalse(bad['ready'])
        combined='\n'.join(bad['issues'])
        for text in ('Нет однозначной пары','Размер маски','без явного соответствия','разную разметку'):
            self.assertIn(text,combined)
    def test_unknown_group_and_missing_class_mapping_block_preparation(self):
        cfg={**self.cfg,'pairs_csv':'','group_regex':'','independent_images':False}
        self.assertFalse(audit(cfg)['ready'])
        cfg={**self.cfg,'mask_mapping':{'0':0}}
        with self.assertRaises(ValueError): prepare(cfg,self.root/'must_not_exist')
        self.assertFalse((self.root/'must_not_exist').exists())
    def test_split_groups_and_immutable_sources(self):
        before={p:digest(p) for p in Path(self.cfg['images']).glob('*.png')}
        output=self.root/'prepared'; prepare(self.cfg,output); self.assertTrue(verify_prepared(output)['pass'])
        rows=read(output/'manifest.json'); train={r['group'] for r in rows if r['split']=='train'}; val={r['group'] for r in rows if r['split']=='val'}
        self.assertFalse(train&val)
        self.assertTrue(all(digest(p)==h for p,h in before.items()))
        with self.assertRaises(ValueError): prepare(self.cfg,output)
        target=output/rows[0]['prepared_annotation']; arr=np.array(Image.open(target)); arr[0,0]=1-arr[0,0]; Image.fromarray(arr).save(target)
        with self.assertRaises(ValueError): verify_prepared(output)
    def test_rgb_palette_and_ignore_values(self):
        rgb=np.array([[[0,0,0],[255,0,0],[0,255,0]]],dtype=np.uint8); p=self.root/'rgb.png'; Image.fromarray(rgb).save(p)
        arr=decode_mask(p,{'0,0,0':0,'255,0,0':1,'0,255,0':-1},2)
        self.assertEqual(arr.tolist(),[[0,1,255]])
        pal=Image.fromarray(np.array([[0,1]],dtype=np.uint8)).convert('P'); pal.putpalette([0,0,0,255,0,0]+[0]*762); pal.save(self.root/'palette.png')
        self.assertEqual(decode_mask(self.root/'palette.png',{'0':0,'1':1},2).tolist(),[[0,1]])
    def test_path_traversal_in_csv_is_rejected(self):
        csv=Path(self.cfg['pairs_csv']); csv.write_text('image,annotation,group\n../outside.png,mask.png,s1\n',encoding='utf-8')
        with self.assertRaises(ValueError): audit(self.cfg)
    def test_coco_ids_are_explicitly_remapped(self):
        root=self.root/'coco_images'; root.mkdir()
        for i in range(4): Image.new('RGB',(32,24),(i*20,50,80)).save(root/f'{i}.png')
        annotations=self.root/'coco.json'
        dump(annotations,{'categories':[{'id':7,'name':'rock'}],
             'images':[{'id':i,'file_name':f'{i}.png','width':32,'height':24} for i in range(4)],
             'annotations':[{'id':i,'image_id':i,'category_id':7,'bbox':[2,3,8,6]} for i in range(4)]})
        cfg={'format':'coco_bbox','images':str(root),'annotations':str(annotations),'classes':['rock'],'independent_images':True}
        output=self.root/'bbox'; prepare(cfg,output); self.assertTrue(verify_prepared(output)['pass'])
        self.assertTrue(all(r['boxes'][0][0]==0 for r in read(output/'manifest.json')))
    def test_yolo_polygons_do_not_become_boxes(self):
        labels=self.root/'labels'; labels.mkdir()
        for p in Path(self.cfg['images']).glob('*.png'): (labels/(p.stem+'.txt')).write_text('0 0.1 0.1 0.9 0.1 0.9 0.9\n')
        cfg={**self.cfg,'format':'yolo_bbox','annotations':str(labels),'classes':['rock'],'pairs_csv':'','mask_suffix':'','independent_images':True}
        report=audit(cfg); self.assertFalse(report['ready']); self.assertIn('полигоны','\n'.join(report['issues']))
    def test_video_frame_indices_and_single_scene(self):
        import cv2
        from cli import extract_frames
        video=self.root/'input.avi'; writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
        self.assertTrue(writer.isOpened())
        for i in range(8): writer.write(np.full((48,64,3),i*20,dtype=np.uint8))
        writer.release(); output=self.root/'frames'; extract_frames(video,output,every=2,limit=3)
        meta=read(output/'frames.json'); self.assertEqual([r['frame_index'] for r in meta['frames']],[0,2,4])
        self.assertEqual(len({r['group'] for r in meta['frames']}),1); self.assertFalse(meta['masks_provided'])
    def test_shared_gpu_lock_refuses_concurrent_launch_without_touching_gpu(self):
        from cli import training_lock
        lock=self.root/'training.lock'
        with training_lock('cuda:0',lock_path=lock):
            with self.assertRaises(ValueError):
                with training_lock('cuda:0',lock_path=lock): pass
        with training_lock('cuda:0',lock_path=lock): pass

class SemanticFlow(unittest.TestCase):
    def test_resume_restores_completed_epoch_after_interruption(self):
        from unittest.mock import patch
        import semantic
        with temp_dir() as tmp:
            root=Path(tmp); fixture=create_demo(root/'fixture'); dataset=root/'prepared'; prepare(read(fixture['good_config']),dataset)
            settings={**read(dataset/'training.json'),'epochs':2,'image_size':32,'batch':2}; run=dataset/'runs/train'
            real=semantic.evaluate; counter=[0]
            def interrupted(*args,**kwargs):
                counter[0]+=1
                if counter[0]==2: raise RuntimeError('simulated power interruption')
                return real(*args,**kwargs)
            with patch('semantic.evaluate',side_effect=interrupted):
                with self.assertRaises(RuntimeError): semantic.train(dataset,settings,run)
            _,checkpoint=semantic.load_checkpoint(run/'last.pt'); self.assertEqual(checkpoint['epoch'],0)
            semantic.train(dataset,settings,run,resume=True)
            _,checkpoint=semantic.load_checkpoint(run/'last.pt'); self.assertEqual(checkpoint['epoch'],1)
            self.assertEqual(len((run/'epochs.csv').read_text().splitlines()),3)

    def test_cpu_training_resume_compare_export_original_geometry(self):
        from semantic import train,compare,export_predictions,load_checkpoint
        with temp_dir() as tmp:
            root=Path(tmp); fixture=create_demo(root/'fixture'); cfg=read(fixture['good_config']); dataset=root/'prepared'; prepare(cfg,dataset)
            settings={**read(dataset/'training.json'),'epochs':1,'image_size':32,'batch':2}; run=dataset/'runs/train'
            train(dataset,settings,run); self.assertTrue((run/'best.pt').is_file())
            first=compare(dataset,run/'best.pt',run/'best.pt'); self.assertAlmostEqual(first['mean_iou_delta'],0)
            # A changed setting is a new experiment, never silent resume.
            with self.assertRaises(ValueError): train(dataset,{**settings,'lr':.002},run,resume=True)
            # Explicit checkpoint transfer for a new fine-tuning run is supported.
            fine=dataset/'runs/finetune'; train(dataset,{**settings,'checkpoint':str(run/'best.pt')},fine)
            result=export_predictions(run/'best.pt',cfg['images'],root/'export')
            self.assertEqual(result['images'],12)
            with Image.open(root/'export/masks/scene0_frame0.png') as image:
                self.assertEqual(image.size,(64,48)); self.assertEqual(image.mode,'L')
            self.assertTrue((root/'export.zip').is_file())
            with self.assertRaises(ValueError): export_predictions(run/'best.pt',cfg['images'],root/'export')

if __name__=='__main__': unittest.main(verbosity=2)
