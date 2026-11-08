from dataset_lab.bootstrap import setup
setup()
import copy,json,tempfile,unittest
from pathlib import Path
from event_benchmark import *

def fixture():
    p={'format':'panoptr_event_gt_v1','protocol':{**PROTOCOL,'max_keyframe_gap_s':2},
       'independence':{'status':'synthetic_software_qa'},'scenes':[{
       'scene_id':'s','source_group':'g','source':'s.avi','source_sha256':'placeholder',
       'duration_s':10.,'fps':10.,'width':100,'height':100,'timing_kind':'synthetic',
       'reviewer':'QA generator','reviewed':[[0,10]],'unusable':[[6,8]],'condition':'original',
       'events':[{'gt_id':'one','class_name':'person','location':'inside','start_s':1.,'end_s':3.,
         'keyframes':[{'t':1.,'box':[10,10,30,50]},{'t':3.,'box':[10,10,30,50]}]},
         {'gt_id':'two','class_name':'person','location':'near','start_s':1.,'end_s':3.,
         'keyframes':[{'t':1.,'box':[60,10,80,50]},{'t':3.,'box':[60,10,80,50]}]}]}]}
    p=validate(p,True); lock={'dataset_signature':signature(p)}
    pred={'format':'panoptr_event_predictions_v1','dataset_signature':lock['dataset_signature'],
          'system':copy.deepcopy(p['system']),'scenes':[{'scene_id':'s','source_sha256':'placeholder',
          'assessed':[[0,6],[8,10]],'events':[]}],'timing_definition':'synthetic QA'}
    return p,pred,lock

def card(identifier,box,t=1.5,alarm=1.7,kind='person',valid=True):
    return {'event_id':identifier,'class_name':kind,'observations':[{'t':t,'alarm_t':alarm,'box':box,'valid':valid}]}

class EventTests(unittest.TestCase):
    def test_one_to_one_neighbours(self):
        p,r,l=fixture(); r['scenes'][0]['events']=[card('p',[10,10,30,50])]
        s=evaluate(p,r,l)['summary']; self.assertEqual(s['recall'],.5); self.assertEqual(s['missed'],1)
    def test_duplicate_separate_from_false_event(self):
        p,r,l=fixture(); r['scenes'][0]['events']=[card('a',[10,10,30,50]),card('dup',[10,10,30,50],2,2.2),card('fp',[40,50,50,70])]
        s=evaluate(p,r,l)['summary']; self.assertEqual(s['duplicates'],1); self.assertEqual(s['false_events'],1); self.assertAlmostEqual(s['false_events_per_minute'],6.)
    def test_no_cards_is_missed_not_perfect(self):
        p,r,l=fixture(); self.assertEqual(evaluate(p,r,l)['summary']['recall'],0)
    def test_missing_scene_blocked(self):
        p,r,l=fixture(); r['scenes']=[]
        with self.assertRaises(ValueError): evaluate(p,r,l)
    def test_wrong_class_false_and_miss(self):
        p,r,l=fixture(); r['scenes'][0]['events']=[card('a',[10,10,30,50],kind='rock')]
        s=evaluate(p,r,l)['summary']; self.assertEqual(s['false_events'],1); self.assertEqual(s['detected'],0)
    def test_invalid_observation_not_alarm(self):
        p,r,l=fixture(); r['scenes'][0]['events']=[card('a',[10,10,30,50],valid=False)]
        s=evaluate(p,r,l)['summary']; self.assertEqual(s['detected'],0); self.assertEqual(s['nonqualifying_cards'],1)
    def test_hit_threshold_and_first_matching_delay(self):
        p,r,l=fixture(); p['protocol']['min_valid_hits']=2
        c=card('a',[10,10,30,50]); c['observations'].append({'t':2.,'alarm_t':2.3,'box':[10,10,30,50],'valid':True}); r['scenes'][0]['events']=[c]
        s=evaluate(p,r,l)['summary']; self.assertAlmostEqual(s['delay_median_s'],1.3); self.assertEqual(s['deadline_recall'],0.)
    def test_source_time_vs_alarm_time(self):
        p,r,l=fixture(); r['scenes'][0]['events']=[card('a',[10,10,30,50],t=2.9,alarm=4.)]
        s=evaluate(p,r,l)['summary']; self.assertEqual(s['detected'],1); self.assertEqual(s['delay_median_s'],3.)
    def test_temporal_mismatch_not_match(self):
        p,r,l=fixture(); r['scenes'][0]['events']=[card('a',[10,10,30,50],t=5,alarm=5.1)]
        self.assertEqual(evaluate(p,r,l)['summary']['false_events'],1)
    def test_false_first_card_cannot_be_redeemed_by_later_overlap(self):
        p,r,l=fixture(); c=card('a',[40,50,50,70],t=.5,alarm=.6)
        c['observations'].append({'t':2.,'alarm_t':2.2,'box':[10,10,30,50],'valid':True}); r['scenes'][0]['events']=[c]
        s=evaluate(p,r,l)['summary']; self.assertEqual(s['false_events'],1); self.assertEqual(s['detected'],0)
    def test_bad_gt_and_availability_independent(self):
        p,r,l=fixture(); s=evaluate(p,r,l)['summary']; self.assertAlmostEqual(s['unassessed_fraction'],.2)
        r['scenes'][0]['assessed']=[[0,10]]; self.assertEqual(evaluate(p,r,l)['summary']['incorrectly_assessed_bad_s'],2)
    def test_empty_availability_not_filled_from_cards(self):
        p,r,l=fixture(); r['scenes'][0]['assessed']=[]; self.assertEqual(evaluate(p,r,l)['summary']['unassessed_fraction'],1.)
    def test_union_availability_no_double_count(self):
        p,r,l=fixture(); r['scenes'][0]['assessed']=[[0,5],[4,10]]; self.assertEqual(evaluate(p,r,l)['summary']['unassessed_fraction'],0.)
    def test_partial_review_rejected(self):
        p,r,l=fixture(); p['scenes'][0]['reviewed']=[[0,8]]
        with self.assertRaises(ValueError): evaluate(p,r,l)
    def test_empty_gt_recall_na(self):
        p,r,l=fixture(); p['scenes'][0]['events']=[]; s=evaluate(p,r,l)['summary']; self.assertIsNone(s['recall']); self.assertIsNone(s['delay_p95_s'])
    def test_nonfinite_and_reverse_time_rejected(self):
        p,r,l=fixture(); r['scenes'][0]['events']=[card('a',[10,10,30,50],alarm=1.0)]
        with self.assertRaises(ValueError): evaluate(p,r,l)
        r['scenes'][0]['events'][0]['observations'][0]['alarm_t']=float('nan')
        with self.assertRaises(ValueError): evaluate(p,r,l)
    def test_sparse_gt_and_duplicate_time_rejected(self):
        p,r,l=fixture(); p['protocol']['max_keyframe_gap_s']=.5
        with self.assertRaises(ValueError): validate(p,True)
        p['protocol']['max_keyframe_gap_s']=2; p['scenes'][0]['events'][0]['keyframes']*=2
        with self.assertRaises(ValueError): validate(p,True)
    def test_unfinished_draft_can_load(self):
        p,r,l=fixture(); p['scenes'][0]['events'][0]['keyframes']=[]; validate(p)
        with self.assertRaises(ValueError): validate(p,True)
    def test_parameter_mismatch_blocked(self):
        p,r,l=fixture(); r['system']['confidence']=.9
        with self.assertRaises(ValueError): evaluate(p,r,l)
    def test_hash_mismatch_blocked(self):
        p,r,l=fixture(); r['scenes'][0]['source_sha256']='wrong'
        with self.assertRaises(ValueError): evaluate(p,r,l)
    def test_maximum_cardinality(self):
        edges=[(0,0,(0,.9,None)),(0,1,(0,.8,None)),(1,0,(0,.7,None))]
        self.assertEqual(len(max_matching(edges)),2)
    def test_interpolation(self):
        e={'keyframes':[{'t':0,'box':[0,0,10,10]},{'t':1,'box':[10,0,20,10]}]}
        self.assertEqual(gt_box(e,.5,1),[5,0,15,10]); self.assertIsNone(gt_box(e,.5,.1))
    def test_normal_lookalike_explained(self):
        p,r,l=fixture(); p['scenes'][0]['events'][0]['kind']='normal'; r['scenes'][0]['events']=[card('a',[10,10,30,50])]
        result=evaluate(p,r,l); self.assertEqual(result['summary']['false_events'],1); self.assertEqual(result['predicted_event_results'][0]['normal_lookalike_ids'],['one'])
    def test_fully_bad_event_visible_in_recall(self):
        p,r,l=fixture(); p['scenes'][0]['unusable']=[[1,3]]
        report=evaluate(p,r,l); self.assertEqual(report['summary']['recall'],0.); self.assertIsNone(report['summary']['recall_with_usable_observation'])
    def test_freeze_mutation_and_report_compare(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root=Path(tmp); p,r,l=fixture(); video=root/'s.avi'
            import cv2,numpy as np
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(100,100)); self.assertTrue(writer.isOpened())
            for _ in range(100): writer.write(np.zeros((100,100,3),np.uint8))
            writer.release()
            p['scenes'][0]['source']=str(video); p['scenes'][0]['source_sha256']=sha256(video)
            save_json(root/'draft.json',p); freeze(root/'draft.json',root/'frozen'); pp,ll=load_frozen(root/'frozen')
            r['dataset_signature']=ll['dataset_signature']; r['scenes'][0]['source_sha256']=pp['scenes'][0]['source_sha256']
            save_json(root/'pred.json',r); score(root/'frozen',root/'pred.json',root/'report')
            compare_reports(root/'report/report.json',root/'report/report.json',root/'compare')
            target=root/'frozen/assets/0.avi'; target.write_bytes(b'changed')
            with self.assertRaises(ValueError): load_frozen(root/'frozen')

if __name__=='__main__': unittest.main()
