from dataset_lab.bootstrap import setup
setup()
import unittest,tempfile,copy,json,zipfile,os,math
from pathlib import Path
import numpy as np
from finale import TrialSeries,profile,configuration,write
from rail_reference import catalogue,advice,validate_review,valid_codes
from final_store import EventStore
from final_core import EventAssociator
from stand_geometry import scene
TEST_TMP=Path(__file__).resolve().parent/'finale_test_tmp'
TEST_TMP.mkdir(exist_ok=True)

class Clock:
    def __init__(self):self.t=100.
    def __call__(self):return self.t
    def tick(self,n=1):self.t+=n
def event(key='a',name='rail_break',qualified=True):
    return {'event_id':key*32,'channel':0,'class_name':name,'observation_state':'observed_repeatedly' if qualified else 'candidate','evidence_saved':True}

class Profiles(unittest.TestCase):
    def test_profiles_do_not_calibrate(self):
        for key,mm in [('field750',750),('track1520',1520),('model',45)]:
            p=profile(key,mm);self.assertEqual(p['gauge_mm'],mm);self.assertFalse(p['metric_calibration_valid'])
    def test_bad_profiles(self):
        for value in [0,9,2001,float('nan'),True,None]:
            with self.assertRaises(ValueError):profile('model',value)
        with self.assertRaises(ValueError):profile('field750',1520)
    def test_configuration_requires_local_source_and_weight(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as t:
            root=Path(t);(root/'models').mkdir();(root/'models/rail9_finetuned_2026.pt').write_bytes(b'test fixture only');(root/'clip.avi').write_bytes(b'video fixture')
            c=configuration(root,'rail',profile('field750'),'file',root/'clip.avi');self.assertEqual(len(c['model_sha256']),64)
            with self.assertRaises(ValueError):configuration(root,'rail',profile('field750'),'file',root/'no.avi')
            with self.assertRaises(ValueError):configuration(root,'person',profile('field750'),'camera',0)
    def test_actual_inner_gauge_and_model_scale(self):
        for mm in [750.,1520.,45.]:
            rails=sorted([m for m in scene(gauge_mm=mm) if m.key=='rail'],key=lambda m:m.vertices[:,0].mean());gap=rails[1].vertices[:,0].min()-rails[0].vertices[:,0].max();self.assertAlmostEqual(gap,mm/1000)
            for m in scene(gauge_mm=mm):self.assertTrue(np.isfinite(m.vertices).all())

class Catalogue(unittest.TestCase):
    def test_source_and_no_automatic_mapping(self):
        self.assertGreaterEqual(len(catalogue()['entries']),40);self.assertIn('20.1',valid_codes());self.assertIsNone(advice('rail_crack')['normative_code']);self.assertFalse(advice('rail_crack')['automatic_diagnosis'])
    def test_out_of_scope_classes(self):
        for name in ['person','drone','rock','fastener_missing','pantograph']:self.assertFalse(advice(name)['relevant'])
    def test_cannot_set_code_from_acceptance_alone(self):
        with self.assertRaises(ValueError):validate_review('rail_crack',{'status':'visual_only','code':'20.1'})
        with self.assertRaises(ValueError):validate_review('rail_crack',{'status':'expert_recorded','code':'20.1'})
    def test_requires_expert_details_and_applicable_code(self):
        v={'status':'expert_recorded','code':'20.1','reviewer':'Испытатель','method':'УЗК','note':'Учебная запись обследования, не результат модели'}
        self.assertFalse(validate_review('rail_crack',v)['automatic_diagnosis'])
        with self.assertRaises(ValueError):validate_review('person',v)
        with self.assertRaises(ValueError):validate_review('rail_crack',dict(v,code='99.9'))

class Trials(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(dir=TEST_TMP);self.root=Path(self.tmp.name);self.clock=Clock();self.s=TrialSeries(self.root/'series',{'channel':0,'profile':profile('field750')},'Испытатель',clock=self.clock)
    def tearDown(self):self.tmp.cleanup()
    def end(self):self.clock.tick();self.s.finish();return self.s.data['trials'][-1]['id']
    def test_truth_not_automatically_inferred(self):
        self.s.begin('target','rail_break','');self.clock.tick();self.s.observe([event()],True);self.end();self.assertIsNone(self.s.summary()['event_recall']);self.assertEqual(self.s.summary()['pending_review_trials'],1)
    def test_true_detection_and_delay(self):
        key=self.s.begin('target','rail_break','');self.clock.tick(.4);self.s.observe([event()],True);self.end();self.s.judge(key,'a'*32,'match','Цель предъявлена, совпадает');self.s.review(key);r=self.s.summary();self.assertEqual(r['event_recall'],1);self.assertAlmostEqual(r['first_card_delay_s'][0],.4)
    def test_miss_without_any_cards_is_preserved(self):
        key=self.s.begin('target','rail_break','');self.clock.tick();self.s.observe([],False);self.end();self.s.review(key);r=self.s.summary();self.assertEqual(r['event_recall'],0);self.assertEqual(r['missed_events'],1);self.assertEqual(r['unassessed_fraction'],1)
    def test_false_events_use_reviewed_duration(self):
        key=self.s.begin('negative','','');self.clock.tick(30);self.s.observe([event()],True);self.clock.tick(30);self.s.finish();self.s.judge(key,'a'*32,'false','На стенде нормальный элемент');self.s.review(key);self.assertAlmostEqual(self.s.summary()['false_events_per_minute'],1)
    def test_unreviewed_event_cannot_enter_metrics(self):
        key=self.s.begin('negative','','');self.clock.tick();self.s.observe([event()],True);self.end()
        with self.assertRaises(ValueError):self.s.review(key)
    def test_single_target_and_duplicates_separate(self):
        key=self.s.begin('target','rail_break','');self.clock.tick();self.s.observe([event(),event('b')],True);self.end();self.s.judge(key,'a'*32,'match','Первое событие')
        with self.assertRaises(ValueError):self.s.judge(key,'b'*32,'match','Вторая карточка')
        self.s.judge(key,'b'*32,'duplicate','Та же физическая цель');self.s.review(key);self.assertEqual(self.s.summary()['detected_events'],1);self.assertEqual(self.s.summary()['duplicate_cards'],1)
    def test_wrong_class_and_candidate_are_not_matches(self):
        key=self.s.begin('target','rail_break','');self.clock.tick();self.s.observe([event(name='person'),event('b',qualified=False)],True);self.end()
        for eid in ['a'*32,'b'*32]:
            with self.assertRaises(ValueError):self.s.judge(key,eid,'match','Проверка')
    def test_old_channel_and_old_events_excluded(self):
        base=event();self.s.begin('negative','','',[base]);self.clock.tick();self.s.observe([base,dict(event('b'),channel=1),dict(event('c'),first_seen=1.)],True);self.assertEqual(self.s.active['events'],{})
    def test_configuration_change_invalidates_trial(self):
        key=self.s.begin('target','rail_break','');self.s.action('configuration_changed','Изменён порог');self.end();self.s.review(key);r=self.s.summary();self.assertEqual(r['changed_configuration_trials'],1);self.assertIsNone(r['event_recall'])
    def test_assistance_counts_only_active_trial(self):
        self.s.action('start','Подготовка');self.s.begin('negative','','');self.s.action('help','Наставник поправил источник');self.end();self.assertEqual(self.s.summary()['assistance_count'],1);self.assertEqual(self.s.summary()['all_assistance_actions'],2)
    def test_export_active_or_inside_source_rejected(self):
        self.s.begin('negative','','')
        with self.assertRaises(ValueError):self.s.export(self.root/'report')
        self.end()
        with self.assertRaises(ValueError):self.s.export(self.s.root/'report')
    def test_reopen_running_trial_marks_interrupted(self):
        self.s.begin('target','rail_break','');restored=TrialSeries.open(self.s.root,clock=self.clock);self.assertIsNone(restored.active);self.assertEqual(restored.summary()['interrupted_trials'],1);self.assertIsNone(restored.summary()['event_recall'])
    def test_bad_clock_is_rejected(self):
        self.s.begin('negative','','');self.clock.t=99
        with self.assertRaises(ValueError):self.s.finish()
        self.clock.t=float('nan')
        with self.assertRaises(ValueError):self.s.observe([],True)
    def test_health_poll_gaps_are_unassessed(self):
        key=self.s.begin('negative','','');self.s.observe([],True);self.clock.tick(10);self.s.finish();self.s.review(key);self.assertAlmostEqual(self.s.summary()['unassessed_host_poll_s'],9)
    def test_export_is_self_contained_and_escaped(self):
        store=self.root/'store';ev=store/'events'/('a'*32);ev.mkdir(parents=True);(ev/'evidence.png').write_bytes(b'png fixture');(ev/'context.avi').write_bytes(b'avi fixture');self.s.data['event_store']=str(store)
        key=self.s.begin('target','rail_break','<script>');self.clock.tick();self.s.observe([event()],True);self.end();self.s.judge(key,'a'*32,'match','Проверено');self.s.review(key);out=self.s.export(self.root/'report')
        with zipfile.ZipFile(str(out)+'.zip') as z:self.assertIn('evidence/'+('a'*32)+'/evidence.png',z.namelist());self.assertIsNone(z.testzip())

class ReviewPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=TEST_TMP);self.root=Path(self.tmp.name);self.store=EventStore(self.root/'events');self.image=np.full((120,160,3),120,np.uint8);assoc=EventAssociator()
        self.event=assoc.update(0,1,1,1.,[[10,10,60,60,.9,0]],{0:'rail_break'},self.image.shape,{'confirm_hits':1},{'assessed':True})[0]
        self.store.submit(self.event,[(1.,self.image)],{'captured_at_monotonic':1.});self.store.flush()
    def tearDown(self):self.store.close();self.tmp.cleanup()
    def test_review_survives_next_observation_and_is_exported(self):
        eid=self.event['event_id'];self.store.save_review(eid,{'status':'needs_measurement','note':'Нужен обмер'});self.store.submit(self.event,[],{});self.store.flush();self.assertIn('technical_review',self.store.events()[0]);out=self.store.export(self.root/'export')
        with zipfile.ZipFile(str(out)+'.zip') as z:self.assertIn('evidence/'+eid+'/technical_review.json',z.namelist())
    def test_bad_review_does_not_hide_event(self):
        (self.store.root/'events'/self.event['event_id']/'technical_review.json').write_text('broken');self.assertEqual(len(self.store.events()),1);self.assertIn('technical_review_error',self.store.events()[0])
    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):self.store.save_review('../other',{})

class NewWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])
    def test_reference_event_does_not_shadow_qt_handler(self):
        from rail_reference_ui import RailReferenceDialog
        window=RailReferenceDialog(event={'event_id':'a'*32,'class_name':'rail_break'})
        try:
            window.show();self.app.processEvents()
            self.assertTrue(callable(window.event));self.assertFalse(window.grab().isNull())
            self.assertEqual(window.status.currentData(),'visual_only')
            self.assertEqual(window.code.currentData(),'')
        finally:window.close()

if __name__=='__main__':unittest.main()
