from dataset_lab.bootstrap import setup
setup()
import os,unittest,json,tempfile,time,copy
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np,cv2
from path_ui import demo_geometry,PathUnderstandingDialog
from path_geometry import *
from path_tracking import *
from path_benchmark import compare,synthetic_tracking_fixture
from final_core import validate_policy,EventAssociator
from PySide6.QtWidgets import QApplication
from unittest.mock import Mock,patch

def evaluated(branches=1):
    frame,g=demo_geometry(branches); return frame,g,RouteState().evaluate(g,0,1,1,g['source_id'])
def calibration_data():
    return {'shape':[360,640],'source_id':'synthetic_demonstration','reviewer':'QA calibration fixture','max_check_error_m':.01,
        'fit_points':[[100,100,1,1],[300,100,3,1],[300,350,3,3.5],[100,350,1,3.5]],
        'check_points':[[150,200,1.5,2],[250,200,2.5,2],[200,300,2,3]]}

class GeometryTests(unittest.TestCase):
    def test_model_change_revokes_route_confirmation(self):
        _,g,_=evaluated(); g['model_sha256']='old'; r=RouteState(); r.confirm(g,'route_1'); g['model_sha256']='new'; g['sequence']=2
        self.assertEqual(r.evaluate(g,0,2,1,g['source_id'])['status'],'ambiguous'); self.assertIsNone(r.selection)
    def test_single_pair_estimated_not_certified(self): _,g,state=evaluated(); self.assertEqual(state['status'],'estimated'); self.assertFalse(state['scores_are_probabilities']); self.assertFalse(state['metres_verified'])
    def test_never_extrapolates_above_visible_rails(self): _,g,state=evaluated(); self.assertFalse(state['corridor_mask'][:98].any())
    def test_multiple_tracks_do_not_select_intertrack_gap(self): _,g,s=evaluated(2); self.assertEqual(len(g['candidates']),2); self.assertEqual(s['status'],'ambiguous'); self.assertIsNone(s['corridor_mask'])
    def test_manual_switch_is_required_for_one_candidate(self): _,g,_=evaluated(); r=RouteState(); s=r.evaluate(g,0,1,1,g['source_id'],switch=True); self.assertEqual(s['status'],'ambiguous'); r.confirm(g,'route_1'); self.assertEqual(r.evaluate(g,0,1,1,g['source_id'],switch=True)['status'],'confirmed')
    def test_route_confirmation_is_bound_and_latches_on_topology_loss(self):
        _,g,_=evaluated(2); r=RouteState(); r.confirm(g,'route_1'); self.assertEqual(r.evaluate(g,0,1,1,g['source_id'])['status'],'confirmed')
        _,one,_=evaluated(); one['sequence']=2; self.assertEqual(r.evaluate(one,0,2,1,one['source_id'])['status'],'ambiguous'); one['sequence']=3; self.assertEqual(r.evaluate(one,0,3,1,one['source_id'])['status'],'ambiguous')
    def test_stale_or_other_source_returns_unknown(self):
        _,g,_=evaluated()
        for kwargs in [{'now':1,'source':g['source_id']},{'now':0,'source':'other'}]: self.assertEqual(RouteState().evaluate(g,kwargs['now'],1,1,kwargs['source'])['status'],'unknown')
    def test_generation_and_sequence_mismatch(self):
        _,g,_=evaluated(); self.assertEqual(RouteState().evaluate(g,0,2,1,g['source_id'])['status'],'unknown'); self.assertEqual(RouteState().evaluate(g,0,1,2,g['source_id'])['status'],'unknown')
    def test_bad_frame_cannot_confirm_geometry(self): _,g,_=evaluated(); self.assertEqual(RouteState().evaluate(g,0,1,1,g['source_id'],health_assessed=False)['status'],'unknown')
    def test_trackbed_only_does_not_invent_rails(self):
        m=np.ones((100,200),bool); g=propose_geometry([m],[[0,0,200,100,.9,0]],{0:'trackbed'},m.shape); self.assertEqual(g['status'],'unknown')
    def test_one_rail_cannot_make_a_corridor(self):
        m=np.zeros((100,200),bool); m[30:,50:56]=1; g=propose_geometry([m],[[0,0,200,100,.9,0]],{0:'rail_left'},m.shape); self.assertEqual(g['status'],'unknown')
    def test_low_confidence_masks_are_not_geometry(self):
        frame,g=demo_geometry(); masks=list(g['semantic_masks'].values()); boxes=[[0,0,640,360,.1,i] for i in range(len(masks))]; bad=propose_geometry(masks,boxes,dict(enumerate(g['semantic_masks'])),frame.shape); self.assertEqual(bad['status'],'unknown')
    def test_mask_resolution_mismatch_rejected(self):
        with self.assertRaises(ValueError): propose_geometry([np.ones((10,10))],[[0,0,200,100,.9,0]],{0:'rail'},(100,200))
    def test_top_overlap_does_not_imply_bottom_in_path(self):
        _,g,s=evaluated(); where=object_context([100,100,140,300],(360,640),s); self.assertNotEqual(where['location'],'inside')
    def test_unknown_geometry_does_not_return_metres(self):
        _,g,s=evaluated(2); c=GroundCalibration(calibration_data()); where=object_context([180,180,240,300],(360,640),s,calibration=c); self.assertEqual(where['metric_status'],'geometry_unknown'); self.assertIsNone(where['metric_ground_xy_m'])

class CalibrationTests(unittest.TestCase):
    def test_verified_points_project_only_in_hull(self): c=GroundCalibration(calibration_data()); self.assertTrue(c.valid); np.testing.assert_allclose(c.project([200,200],'synthetic_demonstration',[360,640])['metric_ground_xy_m'],[2.,2.]); self.assertEqual(c.project([600,200],'synthetic_demonstration',[360,640])['metric_status'],'outside_validated_image_region')
    def test_independent_check_error_blocks_metres(self): d=calibration_data(); d['check_points'][0][2]+=1; c=GroundCalibration(d); self.assertFalse(c.valid); self.assertEqual(c.project([200,200],d['source_id'],d['shape'])['metric_status'],'calibration_check_failed')
    def test_fit_points_cannot_be_reused_as_validation(self): d=calibration_data(); d['check_points'][0]=d['fit_points'][0]; self.assertRaises(ValueError,GroundCalibration,d)
    def test_source_or_resolution_change_blocks_projection(self): c=GroundCalibration(calibration_data()); self.assertEqual(c.project([200,200],'other',[360,640])['metric_status'],'calibration_source_or_resolution_mismatch'); self.assertEqual(c.project([200,200],'synthetic_demonstration',[720,1280])['metric_status'],'calibration_source_or_resolution_mismatch')
    def test_degenerate_or_missing_calibration_rejected(self): d=calibration_data(); d['fit_points']=[[0,0,0,0]]*4; self.assertRaises(ValueError,GroundCalibration,d); self.assertRaises(ValueError,GroundCalibration,{})

class ContextTests(unittest.TestCase):
    def test_outside_stone_is_context_and_not_event(self):
        f,g,s=evaluated(); p=PathEventPipeline(); events=p.update(0,1,1,0,[[450,150,500,300,.9,0]],{0:'rock'},f.shape,{}, {'assessed':True},geometry=s); self.assertEqual(events,[]); self.assertEqual(p.context[0]['risk'],'outside_context')
    def test_inside_rock_on_ballast_is_review_not_collision_alarm(self):
        f,g,s=evaluated(); p=PathEventPipeline(); events=p.update(0,1,1,0,[[180,180,240,300,.9,0]],{0:'rock'},f.shape,{}, {'assessed':True},geometry=s); self.assertEqual(events[0]['context_state'],'ballast_overlap_review'); self.assertFalse(events[0]['automatic_collision_alarm'])
    def test_ballast_never_becomes_obstacle_by_role_override(self):
        f,g,s=evaluated(); p=PathEventPipeline(); events=p.update(0,1,1,0,[[180,180,240,300,.9,0]],{0:'ballast'},f.shape,{'roles':{'ballast':'obstacle'}}, {'assessed':True},geometry=s); self.assertEqual(events,[]); self.assertEqual(p.context[0]['risk'],'background_ballast')
    def test_unknown_is_review_not_suppressed_as_clear(self):
        f,g,s=evaluated(2); p=PathEventPipeline(); events=p.update(0,1,1,0,[[180,180,240,300,.9,0]],{0:'rock'},f.shape,{}, {'assessed':True},geometry=s); self.assertEqual(events[0]['context_state'],'geometry_unknown_review')
    def test_non_ballast_inside_rock_still_requires_review(self):
        f,g,s=evaluated(); s['semantic_masks']['ballast'][:]=False; p=PathEventPipeline(); e=p.update(0,1,1,0,[[180,180,240,300,.9,0]],{0:'rock'},f.shape,{}, {'assessed':True},geometry=s); self.assertEqual(e[0]['context_state'],'path_candidate'); self.assertTrue(e[0]['requires_review'])

class TrackerTests(unittest.TestCase):
    def test_telemetry_template_preserves_detections_without_inventing_GT(self):
        from export_path_tracking_template import make_template
        r={'kind':'path_context','channel':0,'generation':1,'source_ref':{'path':'movie'},'shape':[100,200],'names':{'0':'person'},'object_model_sha256':'sha','sequence':1,'source_time_s':.5,'captured_at':20.,'boxes':[[1,2,20,40,.8,0]],'frame_sha256':'pixels','image_assessed':True}
        second={**r,'sequence':2,'source_time_s':.6,'captured_at':22.}
        d=make_template([r,second],0,1); self.assertAlmostEqual(d['frames'][1]['time_s'],.1); self.assertEqual(d['frames'][0]['detections'],r['boxes']); self.assertFalse(d['frames'][0]['reviewed']); self.assertRaises(ValueError,compare,d)
        self.assertRaises(ValueError,make_template,[r,{**second,'object_model_sha256':'changed'}],0,1)
    def test_benchmark_counts_actual_duplicate_cards(self):
        r=compare(synthetic_tracking_fixture())
        for m in r['metrics']: self.assertEqual(m['event_cards_without_path_filter'],1+m['repeat_event_cards']); self.assertEqual(m['false_event_cards'],0)
    def test_class_and_channel_isolation(self):
        for mode in ('iou','bytetrack'):
            t=TrackingBackend(mode); rows=np.array([[10,10,30,40,.9,0],[10,10,30,40,.9,1]]); first=t.update(0,1,1,0,rows,(100,100)); self.assertNotEqual(first[0],first[1]); other=t.update(1,1,1,0,rows,(100,100)); self.assertNotEqual(first[0],other[0])
    def test_repeated_frame_does_not_double_confirm(self):
        p=PathEventPipeline('bytetrack'); box=[[10,10,30,40,.9,0]]; first=p.update(0,1,1,0,box,{0:'person'},(100,100),{}, {'assessed':True}); second=p.update(0,1,1,.01,box,{0:'person'},(100,100),{}, {'assessed':True}); self.assertEqual(second,[]); self.assertEqual(first[0]['valid_hits'],1)
    def test_generation_time_gap_reset_identity(self):
        for mode in ('iou','bytetrack'):
            t=TrackingBackend(mode); rows=[[10,10,30,40,.9,0]]; a=t.update(0,1,1,0,rows,(100,100)); b=t.update(0,2,1,.1,rows,(100,100)); self.assertNotEqual(a,b); c=t.update(0,2,2,2,rows,(100,100)); self.assertNotEqual(b,c)
    def test_byte_low_confidence_recovery(self):
        t=TrackingBackend('bytetrack'); a=t.update(0,1,1,0,[[10,10,40,40,.9,0]],(100,100)); b=t.update(0,1,2,.05,[[11,10,41,40,.12,0]],(100,100)); self.assertEqual(a,b)
    def test_no_prediction_box_is_emitted_without_detection(self): t=TrackingBackend('bytetrack'); t.update(0,1,1,0,[[10,10,40,40,.9,0]],(100,100)); self.assertEqual(t.update(0,1,2,.05,[],(100,100)),[]); self.assertEqual(t.lost_episodes,1)
    def test_legacy_iou_partition_matches_existing_associator(self):
        tracker=TrackingBackend(); assoc=EventAssociator(); rows=[[10,10,30,40,.9,0]]; associations=[]; ids=[]
        for i in range(5):
            boxes=np.array(rows)+np.array([[i*3,0,i*3,0,0,0]]); ids.append(tracker.update(0,1,i+1,i*.05,boxes,(100,100))[0]); associations.append(assoc.update(0,1,i+1,i*.05,boxes,{0:'person'},(100,100),{}, {'assessed':True})[0]['event_id'])
        self.assertEqual(len(set(ids)),len(set(associations)))
    def test_benchmark_requires_reviewed_gt(self): d=synthetic_tracking_fixture(); d['frames'][0]['reviewed']=False; self.assertRaises(ValueError,compare,d)
    def test_benchmark_reports_tradeoff_not_fake_superiority(self): r=compare(synthetic_tracking_fixture()); self.assertEqual(len(r['metrics']),2); self.assertTrue(r['same_input_detections']); self.assertGreater(r['metrics'][1]['missed_visible_observations'],0)

class IntegrationTests(unittest.TestCase):
    def test_gui_does_not_apply_result_after_settings_change(self):
        w=PathUnderstandingDialog(); w.action='segment'; w.output=Path('unused'); w.pending_frame_hash=frame_digest(w.frame)
        w.pending_settings=(w.model.text(),w.conf.value(),w.size.value(),w.raw['source_id']); original=w.raw
        w.conf.setValue(.4)
        with patch('path_ui.read_geometry',return_value=copy.deepcopy(original)): w.finished(0,None)
        self.assertIs(w.raw,original); self.assertIn('настройки изменились',w.log.toPlainText()); w.close()
    def test_live_geometry_stale_and_wrong_frame_are_unknown(self):
        from main import CameraPanel,DetectionPacket,InferenceScheduler
        from PySide6.QtGui import QPixmap
        scheduler=InferenceScheduler(); defaults={'camera_index':0,'model':'local.pt','device':'CPU','capture_width':640,'capture_height':360,'capture_fps':15,'inference_fps':8,'image_size':320,'confidence':.2}
        p=CameraPanel(0,defaults,scheduler,[]); p.running=True; p.event_store=Mock(); p.policy=validate_policy({'path_enabled':True,'min_blur':0,'max_age_ms':5000,'geometry_max_age_ms':700})
        f,g=demo_geometry(); now=time.monotonic(); g.update(captured_at=now,generation=p.generation,source_id=source_key(p.source_ref))
        capture=Mock(); capture.error=None; capture.finished=False; capture.actual_fps=15.; capture.captured_at=now; capture.latest.return_value=(1,f); capture.frame_age.return_value=0.; p.capture=capture; p.last_submit_time=now
        packet=DetectionPacket(np.empty((0,6)),{},sequence=1,generation=p.generation,captured_at=now,options=p.options(),source_frame=f,geometry=g)
        try:
            p.set_result(packet,10,'CPU'); self.assertEqual(p.path_geometry['status'],'estimated'); p.route_state.confirm(g,'route_1')
            from export_path_tracking_template import make_template
            sample=p.event_store.telemetry.call_args.args[0]; self.assertEqual(sample['kind'],'path_context'); json.dumps(sample,allow_nan=False)
            template=make_template([sample],p.channel,p.generation); self.assertFalse(template['frames'][0]['reviewed'])
            p.last_ring_sequence=1; p.capture_health={'assessed':True}; p.last_submitted_sequence=1
            with patch('main.time.monotonic',return_value=now+.8),patch('main.frame_to_pixmap',return_value=QPixmap()): p.refresh()
            self.assertEqual(p.path_geometry['status'],'unknown'); self.assertIsNone(p.path_geometry['corridor_mask']); self.assertTrue(p.route_state.require_confirmation)
            g2=copy.deepcopy(g); g2.update(sequence=2,captured_at=time.monotonic(),frame_sha256='wrong')
            from dataclasses import replace
            p.set_result(replace(packet,sequence=2,captured_at=g2['captured_at'],geometry=g2),10,'CPU')
            self.assertEqual(p.path_geometry['status'],'unknown'); self.assertEqual(p.path_raw['candidates'],[])
        finally: p.running=False; scheduler.close(); p.close()
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])
    def test_gui_switch_confirmation_and_ballast_probe(self):
        w=PathUnderstandingDialog(); w.show(); self.app.processEvents(); w.demo_switch(); self.assertEqual(w.geometry['status'],'ambiguous'); w.confirm(); self.assertEqual(w.geometry['status'],'confirmed'); w.object_class.setCurrentText('ballast'); w.probe(); self.assertIn('background_ballast',w.details.toPlainText()); w.close()
    def test_main_packet_missing_segmenter_retains_detector_hash_and_unknown(self):
        from main import InferenceScheduler,InferenceOptions,InferenceJob,DetectionPacket
        scheduler=object.__new__(InferenceScheduler); scheduler._last_model_sha='objectsha'; scheduler._last_load_ms=2.; f=np.zeros((100,200,3),np.uint8)
        job=InferenceJob(0,1,1,time.monotonic(),f,InferenceOptions('model','CPU',320,.5,True,'missing.pt'))
        packet=scheduler._stamp(DetectionPacket(np.empty((0,6)),{}),job); self.assertEqual(packet.model_sha256,'objectsha'); self.assertEqual(packet.geometry['status'],'unknown')
    def test_path_profile_rejects_invalid_config(self): self.assertRaises(ValueError,validate_policy,{'tracker':'fake'}); self.assertRaises(ValueError,validate_policy,{'geometry_max_age_ms':float('nan')})

if __name__=='__main__': unittest.main()
