"""Annotation integrity, reviewed-only exports and inference contracts, CPU only."""
from bootstrap import setup
setup()
from pathlib import Path
import copy,json,unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from test_lab import temp_dir
from core import read,digest,prepare,verify_prepared
import annotation as a
import annotation_models as m

class AnnotationChecks(unittest.TestCase):
    def test_incomplete_clip_has_actionable_error_instead_of_attribute_error(self):
        import sys
        from types import SimpleNamespace
        with patch.dict(sys.modules,{'clip':SimpleNamespace()}):
            with self.assertRaisesRegex(ValueError,'CLIP'):
                m.offline_clip({})

    def setUp(self):
        self.temp=temp_dir(); self.root=Path(self.temp.name); self.images=self.root/'images'; self.images.mkdir()
        for i in range(3): Image.new('RGB',(32,24),(i*50,50,80)).save(self.images/f'{i}.png')
        self.project=self.root/'project'; a.create_project(self.images,self.project,['person','rock'],independent=True)
        self.ids=a.project(self.project)['frame_ids']; self.before={p:digest(p) for p in self.images.iterdir()}
    def tearDown(self):
        self.assertTrue(all(digest(p)==sha for p,sha in self.before.items())); self.temp.cleanup()
    def mask(self):
        mask=np.zeros((24,32),bool); mask[4:16,6:20]=True; return mask
    def confirmed(self,index=0,mask=True):
        row=a.frame(self.project,self.ids[index]); a.add_object(self.project,row,0,mask=self.mask()) if mask else a.add_object(self.project,row,0,bbox=[6,4,20,16])
        row['status']='confirmed'; a.save_frame(self.project,row); return row
    def test_review_required_and_unassigned_class_block(self):
        with self.assertRaises(ValueError): a.export_project(self.project,self.root/'export','yolo_bbox')
        self.assertFalse((self.root/'export').exists())
        row=a.frame(self.project,self.ids[0]); a.add_object(self.project,row,None,mask=self.mask(),source='model'); row['status']='confirmed'
        with self.assertRaises(ValueError): a.save_frame(self.project,row)
        self.assertEqual(a.frame(self.project,self.ids[0])['revision'],0)
    def test_revision_and_provenance_guard(self):
        row=a.frame(self.project,self.ids[0]); stale=copy.deepcopy(row); a.save_frame(self.project,row)
        with self.assertRaises(ValueError): a.save_frame(self.project,stale)
        row['image_sha256']='changed'
        with self.assertRaises(ValueError): a.save_frame(self.project,row)
    def test_exact_rle_holes_and_disconnected_parts(self):
        mask=self.mask(); mask[7:10,9:13]=False; mask[20:23,28:31]=True
        self.assertTrue(np.array_equal(a.rle_decode(a.rle_encode(mask)),mask))
        row=a.frame(self.project,self.ids[0]); a.add_object(self.project,row,0,mask=mask); row['status']='confirmed'; a.save_frame(self.project,row)
        target=self.root/'poly'
        with self.assertRaises(ValueError): a.export_project(self.project,target,'yolo_segment')
        self.assertFalse(target.exists()); a.export_project(self.project,self.root/'coco','coco')
        ann=read(self.root/'coco/annotations.json')['annotations'][0]
        self.assertTrue(np.array_equal(a.rle_decode(ann['segmentation']),mask))
    def test_bbox_never_invented_as_mask(self):
        self.confirmed(mask=False)
        for kind in ('semantic','yolo_segment'):
            with self.assertRaises(ValueError): a.export_project(self.project,self.root/kind,kind)
            self.assertFalse((self.root/kind).exists())
    def test_semantic_overlap_explicit_ignore_or_last(self):
        row=self.confirmed(); second=self.mask(); second[:7,:9]=True; a.add_object(self.project,row,1,mask=second); row['status']='confirmed'; a.save_frame(self.project,row)
        with self.assertRaises(ValueError): a.export_project(self.project,self.root/'bad','semantic')
        a.export_project(self.project,self.root/'ignore','semantic','ignore'); arr=np.array(Image.open(self.root/'ignore/masks/0.png.png'))
        self.assertEqual(int(arr[8,10]),255); self.assertEqual(int(arr[1,1]),2)
        a.export_project(self.project,self.root/'last','semantic','last'); arr=np.array(Image.open(self.root/'last/masks/0.png.png')); self.assertEqual(int(arr[8,10]),2)
    def test_four_exports_prepare_verify_and_keep_groups(self):
        self.confirmed(0); self.confirmed(1)
        for kind in ('yolo_bbox','yolo_segment','coco','semantic'):
            with self.subTest(kind=kind):
                target=self.root/kind; report=a.export_project(self.project,target,kind); self.assertEqual(report['images'],2)
                cfg=read(target/'import_config.json'); prepared=self.root/(kind+'_prepared'); prepare(cfg,prepared)
                self.assertTrue(verify_prepared(prepared)['pass']); rows=read(prepared/'manifest.json')
                self.assertEqual({r['group'] for r in rows},{a.frame(self.project,self.ids[i])['group'] for i in (0,1)})
                self.assertFalse({r['group'] for r in rows if r['split']=='train'}&{r['group'] for r in rows if r['split']=='val'})
    def test_model_batch_protects_reviewed_frames_and_empty_is_unreviewed(self):
        self.confirmed(0); row=a.frame(self.project,self.ids[1]); row['status']='edited'; a.save_frame(self.project,row)
        class Fake:
            def __init__(self,*args): self.provenance={'test_only':True}
            def predict(self,*args): return []
        with patch.object(m,'Adapter',Fake): report=m.run_batch(self.project,{'mode':'yolo_bbox'})
        self.assertEqual(report['processed'],1); self.assertEqual(a.frame(self.project,self.ids[0])['status'],'confirmed')
        last=a.frame(self.project,self.ids[2]); self.assertEqual(last['status'],'unreviewed'); self.assertTrue(last['inference_complete'])
        with self.assertRaises(ValueError): m.apply(self.project,self.ids[0],[],{})
    def test_interactive_refinement_preserves_other_objects_requires_review(self):
        row=self.confirmed(); other=a.add_object(self.project,row,1,bbox=[1,1,4,4]); row['status']='confirmed'; a.save_frame(self.project,row)
        target=row['objects'][0]['id']
        class Fake:
            def __init__(self,*args): self.provenance={'test_only':True}
            def predict(self,*args): return [{'class_id':0,'mask':np.ones((24,32),bool)}]
        with patch.object(m,'Adapter',Fake): m.run_batch(self.project,{'mode':'sam_prompt'},identifier=row['id'],prompt={'replace_object':target})
        updated=a.frame(self.project,row['id']); self.assertEqual(updated['status'],'edited'); self.assertIn(other['id'],[o['id'] for o in updated['objects']]); self.assertEqual(len(updated['objects']),2)
    def test_mask_edit_keeps_old_mask_and_empty_erase_refused(self):
        row=self.confirmed(); obj=row['objects'][0]; old=obj['mask']; sha=digest(self.project/old)
        a.edit_mask(self.project,row,obj,[[2,2]],2); self.assertNotEqual(obj['mask'],old); self.assertEqual(digest(self.project/old),sha)
        new=obj['mask']
        with self.assertRaises(ValueError): a.edit_mask(self.project,row,obj,[[16,12]],100,erase=True)
        self.assertEqual(obj['mask'],new)
    def test_person_filter_does_not_merge_other_classes(self):
        import torch
        from ultralytics.engine.results import Results
        result=Results(np.zeros((24,32,3),np.uint8),'fake',{0:'person',1:'car'},boxes=torch.tensor([[1,1,5,6,.8,0],[7,2,12,8,.9,1]]))
        proposals=m.convert_result(result,{'mode':'yolo_bbox'},['person']); self.assertEqual(len(proposals),1); self.assertEqual(proposals[0]['class_id'],0)
        proposals=m.convert_result(result,{'mode':'yolo_bbox','keep_unknown':True},['person']); self.assertIsNone(proposals[1]['class_id'])
        result.names=['person','car']
        proposals=m.convert_result(result,{'mode':'sam3_text'},['person']); self.assertEqual(len(proposals),1)
    def test_invalid_modes_missing_sam3_weights_and_confidence(self):
        for mode in m.MODES:
            with self.subTest(mode=mode), self.assertRaises(ValueError): m.preflight({'mode':mode,'model':str(self.root/'missing.pt')})
        row=self.confirmed(); row['objects'][0]['confidence']=float('nan')
        with self.assertRaises(ValueError): a.save_frame(self.project,row)

if __name__=='__main__': unittest.main(verbosity=2)
