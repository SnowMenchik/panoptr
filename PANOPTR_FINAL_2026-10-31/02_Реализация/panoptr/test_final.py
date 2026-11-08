"""CPU-only integrity, failover, event and data-review checks; no real camera or GPU."""
from dataset_lab.bootstrap import setup
setup()
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
import tempfile,unittest,time,threading,json,multiprocessing as mp
from unittest.mock import patch,Mock
from dataclasses import replace
import numpy as np
import cv2
from PySide6.QtWidgets import QApplication
from final_core import EventAssociator,frame_health,zone_location,validate_policy,sha256,save_json
from final_store import EventStore,ArchiveStore
from final_tools import duplicate_review,review_predictions,training_registry,readiness
from final_execution import gpu_guard

def writer(buffer,sequence,stamp,done):
    array=np.frombuffer(buffer,np.uint8).reshape(48,64,3)
    for n in range(1,90):
        with sequence.get_lock():
            array[:24]=n; time.sleep(.001); array[24:]=n; stamp.value=float(n); sequence.value=n
        time.sleep(.001)
    done.set()

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        base=Path(__file__).resolve().parent/'.final_test_tmp'; base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base); self.root=Path(self.temp.name)
        self.frame=np.random.default_rng(1).integers(0,255,(48,64,3),np.uint8); self.good=frame_health(self.frame,1.,1.01,15,{})
    def tearDown(self): self.temp.cleanup()
    def test_shared_buffer_is_coherent_under_partial_writes(self):
        from main import CameraCapture
        capture=CameraCapture(); capture.width=64; capture.height=48; ctx=mp.get_context('spawn')
        capture._buffer=ctx.RawArray('B',48*64*3); capture._sequence=ctx.Value('L',0); capture._last_frame_at=ctx.Value('d',0,lock=False)
        done=ctx.Event(); process=ctx.Process(target=writer,args=(capture._buffer,capture._sequence,capture._last_frame_at,done)); process.start(); seen=0; deadline=time.monotonic()+15
        try:
            while time.monotonic()<deadline and not done.is_set():
                sequence,image=capture.latest()
                if image is not None:
                    self.assertTrue((image==sequence).all()); self.assertEqual(capture.captured_at,float(sequence)); seen+=1
                time.sleep(.001)
            self.assertTrue(done.is_set()); self.assertGreater(seen,5)
        finally:
            process.join(3)
            if process.is_alive(): process.terminate(); process.join()
    def test_dead_writer_lock_does_not_freeze_reader(self):
        from main import CameraCapture
        capture=CameraCapture(); capture.width=64; capture.height=48; ctx=mp.get_context('spawn')
        capture._buffer=ctx.RawArray('B',48*64*3); capture._sequence=ctx.Value('L',0)
        entered=threading.Event(); release=threading.Event()
        def hold():
            with capture._sequence.get_lock(): entered.set(); release.wait(2)
        thread=threading.Thread(target=hold); thread.start(); self.assertTrue(entered.wait(1)); started=time.monotonic()
        try: self.assertIsNone(capture.latest()[1]); self.assertLess(time.monotonic()-started,.2)
        finally: release.set(); thread.join()
    def test_dark_stale_blur_and_temporal_coverage(self):
        self.assertFalse(frame_health(np.zeros_like(self.frame),1,1.01,15,{})['assessed'])
        self.assertIn('устаревший кадр',frame_health(self.frame,1,3,15,{})['reasons'])
        health=frame_health(self.frame,1,1.01,8,{'speed_kmh':400,'visible_length_m':1})
        self.assertEqual(health['coverage_status'],'insufficient'); self.assertAlmostEqual(health['coverage']['sample_spacing_m'],13.8888889,places=5)
        self.assertEqual(self.good['coverage_status'],'not_calibrated'); self.assertFalse(self.good['blur_threshold_calibrated'])
        with self.assertRaises(ValueError): validate_policy({'max_age_ms':float('nan')})
        blurred=cv2.GaussianBlur(self.frame,(21,21),8); sharp_score=self.good['blur_variance']; blur_score=frame_health(blurred,1,1.01,15,{})['blur_variance']
        threshold=(sharp_score+blur_score)/2
        self.assertTrue(frame_health(self.frame,1,1.01,15,{'min_blur':threshold})['assessed']); self.assertFalse(frame_health(blurred,1,1.01,15,{'min_blur':threshold})['assessed'])
    def test_zone_unset_inside_outside_and_mask_shape(self):
        self.assertEqual(zone_location([20,20,40,40],self.frame.shape,{})['location'],'zone_not_set')
        policy={'zone':[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],'near_pixels':1}
        self.assertEqual(zone_location([20,20,40,30],self.frame.shape,policy)['location'],'inside')
        self.assertEqual(zone_location([0,0,3,3],self.frame.shape,policy)['location'],'outside')
        with self.assertRaises(ValueError): zone_location([1,1,2,2],self.frame.shape,policy,np.ones((2,2),bool))
    def test_components_do_not_create_defects_and_one_object_one_event(self):
        tracker=EventAssociator(); boxes=np.array([[1,1,12,15,.8,0],[20,2,32,15,.9,1]])
        first=tracker.update(0,0,1,0,boxes,{0:'normal',1:'missing'},self.frame.shape,{},self.good)
        second=tracker.update(0,0,2,.1,boxes,{0:'normal',1:'missing'},self.frame.shape,{},self.good)
        self.assertEqual(len(first),1); self.assertEqual(first[0]['event_id'],second[0]['event_id']); self.assertEqual(second[0]['hits'],2)
        self.assertEqual(second[0]['operator_status'],'unreviewed'); self.assertEqual(second[0]['observation_state'],'observed_repeatedly')
        self.assertEqual(tracker.update(0,0,2,.1,boxes,{0:'normal',1:'missing'},self.frame.shape,{},self.good),[])
    def test_nearby_objects_channels_and_generations_stay_separate(self):
        tracker=EventAssociator(); boxes=np.array([[1,1,10,15,.8,0],[11,1,20,15,.8,0]])
        a=tracker.update(0,0,1,0,boxes,{0:'person'},self.frame.shape,{},self.good)
        b=tracker.update(0,0,2,.1,boxes,{0:'person'},self.frame.shape,{},self.good)
        self.assertEqual(len({r['event_id'] for r in a+b}),2)
        c=tracker.update(1,0,1,0,boxes[:1],{0:'person'},self.frame.shape,{},self.good); self.assertNotIn(c[0]['event_id'],[r['event_id'] for r in a])
        tracker.reset(0); d=tracker.update(0,1,1,.2,boxes[:1],{0:'person'},self.frame.shape,{},self.good); self.assertNotIn(d[0]['event_id'],[r['event_id'] for r in a])
    def test_bad_observations_do_not_confirm_good_hits(self):
        tracker=EventAssociator(); boxes=np.array([[1,1,10,15,.8,0]])
        bad={**self.good,'assessed':False,'reasons':['dark']}
        a=tracker.update(0,0,1,0,boxes,{0:'person'},self.frame.shape,{},bad)[0]
        b=tracker.update(0,0,2,.1,boxes,{0:'person'},self.frame.shape,{},self.good)[0]
        self.assertEqual(a['observation_state'],'unassessed_candidate'); self.assertEqual(b['valid_hits'],1); self.assertEqual(b['observation_state'],'candidate')
    def event(self): return EventAssociator().update(0,0,1,1,np.array([[1,1,10,15,.8,0]]),{0:'person'},self.frame.shape,{},self.good)[0]
    def test_evidence_decision_history_clip_export_and_provenance(self):
        store=EventStore(self.root/'session')
        try:
            event=self.event(); store.submit(event,[(.9,self.frame),(1.,self.frame)],{'sequence':1,'model_sha256':'original'}); store.flush()
            folder=store.root/'events'/event['event_id']; self.assertTrue((folder/'context.avi').is_file()); sha=sha256(folder/'evidence.png')
            store.decide(event['event_id'],'accepted','reviewed'); event['sequence']=2; event['box']=[2,2,11,16]
            store.submit(event,[],{'sequence':2}); store.flush(); row=store.events()[0]
            self.assertEqual(row['operator_status'],'accepted'); self.assertEqual(row['evidence_observation']['sequence'],1); self.assertEqual(sha256(folder/'evidence.png'),sha)
            store.telemetry({'kind':'accepted_result','end_to_end_ms':10}); store.flush(); store.export(self.root/'export')
            self.assertTrue((self.root/'export.zip').is_file()); report=json.loads((self.root/'export/summary.json').read_text()); self.assertIsNone(report['recall'])
            archive=ArchiveStore(store.root); self.assertEqual(len(archive.events()),1)
        finally: store.close()
    def test_disk_failure_and_queue_pressure_are_visible(self):
        store=EventStore(self.root/'disk')
        try:
            with patch('final_store.Image.Image.save',side_effect=OSError('disk error')): store.submit(self.event(),[(1,self.frame)],{}); store.flush()
            self.assertIn('Ошибка записи',store.error)
            self.assertFalse(store.events())
            with patch.object(store.queue,'put_nowait',side_effect=__import__('queue').Full): self.assertFalse(store.submit(self.event(),[],{})); self.assertGreater(store.dropped,0)
        finally: store.close()
    def test_gpu_lock_matches_training_and_refuses_without_gpu(self):
        lock=self.root/'lock'
        with gpu_guard('0',lock):
            with self.assertRaises(RuntimeError):
                with gpu_guard('0',lock): pass
        with gpu_guard('cpu',lock): pass
    def test_camera_shutdown_does_not_block_gui_on_driver_join(self):
        from main import CameraCapture
        capture=CameraCapture(); process=Mock(); process.join.side_effect=lambda **kwargs:time.sleep(.25); process.is_alive.return_value=False
        capture._process=process; started=time.monotonic(); capture.stop(); self.assertLess(time.monotonic()-started,.15)
        capture.wait_closed(); process.join.assert_called_once()
    def test_duplicates_are_candidates_and_sources_unchanged(self):
        from PIL import Image
        a=self.root/'a'; b=self.root/'b'; a.mkdir(); b.mkdir(); Image.fromarray(self.frame).save(a/'one.png'); Image.fromarray(self.frame).save(b/'copy.png')
        before=sha256(a/'one.png'); report=duplicate_review(a,b); self.assertTrue(report['complete_scan']); self.assertTrue(report['candidates'][0]['exact_duplicate']); self.assertEqual(sha256(a/'one.png'),before)
    def test_missing_gt_is_not_treated_as_background(self):
        p=self.root/'p.json'; g=self.root/'g.json'; s=self.root/'s.json'
        save_json(p,[{'image':'one.png','width':64,'height':48,'detections':[{'xyxy':[1,1,10,15],'class_id':0,'confidence':.2}]}]); save_json(g,[]); save_json(s,[{'image':'one.png','detections':[]}])
        row=review_predictions(p,g,s)['queue'][0]; self.assertIn('ground_truth_missing',row['reasons']); self.assertNotIn('unmatched_prediction',row['reasons']); self.assertIn('model_disagreement',row['reasons'])
    def test_registry_reads_only_and_does_not_rank_different_runs(self):
        path=self.root/'08_Обучение/runs/run/results.csv'; path.parent.mkdir(parents=True); path.write_text('epoch,metrics/mAP50(B)\n1,0.5\n2,0.6\n')
        sha=sha256(path); report=training_registry(self.root); self.assertEqual(report['runs'][0]['epochs_recorded'],2); self.assertFalse(report['ranking_performed']); self.assertEqual(sha256(path),sha)
    def test_fixed_comparison_copies_gt_and_ignores_download_code(self):
        import yaml
        from PIL import Image
        from final_compare import prepare_validation
        images=self.root/'dataset/images/val'; labels=self.root/'dataset/labels/val'; images.mkdir(parents=True); labels.mkdir(parents=True)
        Image.fromarray(self.frame).save(images/'one.png'); (labels/'one.txt').write_text('0 .5 .5 .2 .3\n'); before=sha256(images/'one.png')
        data=self.root/'input.yaml'; data.write_text(yaml.safe_dump({'path':str(self.root/'dataset'),'val':'images/val','names':['person'],'download':'raise RuntimeError("must never execute")'}))
        local,metadata=prepare_validation(data,'val',self.root/'evaluation','detect',{0:'person'})
        copied=yaml.safe_load(local.read_text()); self.assertNotIn('download',copied); self.assertEqual(metadata['gt_instances_per_class'],[1]); self.assertEqual(sha256(images/'one.png'),before)
        self.assertTrue(Path(metadata['source_manifest'][0]['copied_image']).is_file()); self.assertFalse(metadata['independent_test_verified'])
    def test_comparison_missing_gt_and_class_mismatch_block_before_copy(self):
        import yaml
        from PIL import Image
        from final_compare import prepare_validation
        images=self.root/'dataset/images/val'; images.mkdir(parents=True); Image.fromarray(self.frame).save(images/'one.png')
        data=self.root/'input.yaml'; data.write_text(yaml.safe_dump({'path':str(self.root/'dataset'),'val':'images/val','names':['person']}))
        for names in ({0:'person'},{0:'rock'}):
            with self.assertRaises(ValueError): prepare_validation(data,'val',self.root/'must_not_exist','detect',names)
        self.assertFalse((self.root/'must_not_exist').exists())
    def test_offline_readiness_and_file_playback(self):
        from final_capture import FileCapture
        weights=self.root/'model.pt'; weights.write_bytes(b'local'); report=readiness(self.root,[str(weights)]); self.assertTrue(report['ready']); self.assertFalse(report['gpu_used'])
        video=self.root/'test.avi'; writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),5,(64,48)); self.assertTrue(writer.isOpened()); writer.write(self.frame); writer.write(self.frame); writer.release()
        source=FileCapture(); source.start(video); seq,im=source.latest(); self.assertEqual(seq,1); self.assertEqual(im.shape,self.frame.shape); self.assertEqual(source.source_time,0); source.stop()
    def test_stale_generation_settings_and_out_of_order_results_are_rejected(self):
        from main import MainWindow,DetectionPacket
        defaults={'camera_index':0,'model':str(self.root/'local.pt'),'device':'CPU','capture_width':64,'capture_height':48,'capture_fps':15,'inference_fps':8,'image_size':320,'confidence':.2}
        with patch('main.discover_camera_devices',return_value=[]): window=MainWindow({'window':{'width':1280,'height':900},'cameras':[defaults,defaults],'event_session':str(self.root/'session')})
        window.timer.stop(); p=window.panels[0]; p.running=True
        try:
            packet=DetectionPacket(np.array([[1,1,10,15,.8,0]]),{0:'person'},sequence=1,generation=p.generation,captured_at=time.monotonic(),options=p.options(),source_frame=self.frame)
            p.set_result(packet,10,'CPU'); self.assertIs(p.detections,packet)
            p.confidence.setValue(.4); p.set_result(packet,10,'CPU'); self.assertIsNone(p.detections); self.assertEqual(p.rejected_results,1)
            p.set_failure({'generation':packet.generation,'message':'old failure'}); self.assertNotIn('old failure',p.status.text())
            new=replace(packet,generation=p.generation,options=p.options(),captured_at=time.monotonic()); p.set_result(new,10,'CPU'); self.assertIs(p.detections,new)
            p.set_result(replace(new,sequence=0),10,'CPU'); self.assertIs(p.detections,new)
            p.set_result(replace(new,sequence=2,captured_at=time.monotonic()-10),10,'CPU'); self.assertIs(p.detections,new)
            p.stop(); p.set_result(replace(new,sequence=2),10,'CPU'); self.assertIsNone(p.detections)
        finally: window.close()
    def test_file_event_time_comes_from_analysed_frame(self):
        from main import CameraPanel,DetectionPacket,InferenceScheduler
        scheduler=InferenceScheduler(); defaults={'camera_index':0,'model':'local.pt','device':'CPU','capture_width':64,'capture_height':48,'capture_fps':15,'inference_fps':8,'image_size':320,'confidence':.2}
        panel=CameraPanel(0,defaults,scheduler,[]); panel.running=True; panel.source_kind='file'
        capture=Mock(); capture.actual_fps=10.; capture.source_time=9.9; capture.error=None; panel.capture=capture
        panel.event_store=Mock(); panel.source_ref={'kind':'file'}
        packet=DetectionPacket(np.array([[1,1,10,15,.8,0]]),{0:'person'},sequence=3,generation=panel.generation,captured_at=time.monotonic(),options=panel.options(),source_frame=self.frame)
        try:
            panel.set_result(packet,10,'CPU')
            self.assertEqual(panel.event_store.submit.call_args.args[2]['source_time_s'],.2)
            self.assertEqual(panel.event_store.telemetry.call_args.args[0]['source_time_s'],.2)
        finally: panel.running=False; scheduler.close()
    def test_boxes_are_drawn_on_evidence_frame_not_latest_frame(self):
        from main import CameraPanel,DetectionPacket,InferenceScheduler
        from PySide6.QtGui import QPixmap
        scheduler=InferenceScheduler(); defaults={'camera_index':0,'model':'local.pt','device':'CPU','capture_width':64,'capture_height':48,'capture_fps':15,'inference_fps':8,'image_size':320,'confidence':.2}
        panel=CameraPanel(0,defaults,scheduler,[]); source=np.full_like(self.frame,[20,100,200]); panel.running=True
        panel.detections=DetectionPacket(np.array([[1,1,5,5,.8,0]]),{0:'person'},sequence=1,generation=panel.generation,captured_at=time.monotonic(),options=panel.options(),source_frame=source)
        capture=Mock(); capture.latest.return_value=(99,np.full_like(source,90)); capture.frame_age.return_value=0.; capture.captured_at=time.monotonic(); capture.error=None; capture.finished=False; panel.capture=capture; panel.last_submit_time=time.monotonic()
        rendered=[]
        try:
            with patch('main.frame_to_pixmap',side_effect=lambda image,size:(rendered.append(image.copy()),QPixmap())[1]): panel.refresh()
            self.assertTrue(np.array_equal(rendered[0][40,50],source[40,50]))
            capture.latest.return_value=(100,np.zeros_like(source)); capture.captured_at=time.monotonic()
            with patch('main.frame_to_pixmap',side_effect=lambda image,size:(rendered.append(image.copy()),QPixmap())[1]): panel.refresh()
            self.assertIn('текущий кадр источника',panel.health_label.text()); self.assertFalse(rendered[-1].any())
        finally: panel.stop(); scheduler.close()
    def test_segmentation_removes_letterbox_padding_in_original_coordinates(self):
        import torch
        from ultralytics.engine.results import Results
        from main import InferenceScheduler
        masks=torch.zeros((1,200,200)); masks[0,75:100,20:40]=1
        result=Results(np.zeros((100,200,3),np.uint8),'fixture',{0:'rail_left'},boxes=torch.tensor([[20,25,40,50,.8,0]]),masks=masks)
        packet=InferenceScheduler._make_packet(result,(100,200,3)); self.assertEqual(packet.masks[0].shape,(100,200)); expected=np.zeros((100,200),bool); expected[25:50,20:40]=True
        self.assertTrue(np.array_equal(packet.masks[0],expected))
    def test_shared_model_cache_uses_one_instance_for_two_channels(self):
        from main import InferenceScheduler
        path=self.root/'local.pt'; path.write_bytes(b'weights'); scheduler=InferenceScheduler()
        try:
            with patch('ultralytics.YOLO',return_value=object()) as loader:
                a=scheduler._get_model(0,str(path),'cpu'); b=scheduler._get_model(1,str(path),'cpu'); self.assertIs(a,b); self.assertEqual(loader.call_count,1)
        finally: scheduler.close()
    def test_scheduler_exception_recovers_and_has_frame_provenance(self):
        import torch
        from ultralytics.engine.results import Results
        from main import InferenceScheduler,InferenceOptions
        path=self.root/'local.pt'; path.write_bytes(b'weights'); model=Mock(); result=Results(self.frame,'fixture',{0:'person'},boxes=torch.tensor([[1,1,10,15,.8,0]]))
        model.predict.side_effect=[RuntimeError('injected CPU error'),[result]]; scheduler=InferenceScheduler(); failed=[]; complete=[]
        scheduler.signals.failed.connect(lambda c,error:failed.append(error)); scheduler.signals.completed.connect(lambda c,packet,elapsed,device:complete.append(packet))
        def until(predicate):
            end=time.monotonic()+5
            while not predicate() and time.monotonic()<end: self.app.processEvents(); time.sleep(.01)
            self.assertTrue(predicate())
        try:
            with patch('ultralytics.YOLO',return_value=model):
                options=InferenceOptions(str(path),'CPU',320,.2); scheduler.submit(0,1,self.frame,options,0); until(lambda:bool(failed))
                scheduler.submit(0,2,self.frame,options,0); until(lambda:bool(complete)); self.assertEqual(complete[0].sequence,2); self.assertEqual(complete[0].model_sha256,sha256(path)); self.assertEqual(complete[0].generation,0)
                scheduler.invalidate(0,1); scheduler.submit(0,3,self.frame,options,0); self.assertFalse(scheduler._pending)
        finally: scheduler.close()
    def test_auto_fallback_reports_degraded_device_without_using_gpu(self):
        from contextlib import nullcontext
        import torch
        from ultralytics.engine.results import Results
        from main import InferenceScheduler,InferenceOptions
        path=self.root/'local.pt'; path.write_bytes(b'weights'); model=Mock(); result=Results(self.frame,'fixture',{0:'person'},boxes=torch.tensor([[1,1,10,15,.8,0]]))
        def predict(**kwargs):
            if kwargs['device']!='cpu': raise RuntimeError('injected unavailable CUDA')
            return [result]
        model.predict.side_effect=predict; scheduler=InferenceScheduler(); complete=[]
        scheduler.signals.completed.connect(lambda c,packet,elapsed,device:complete.append((packet,device)))
        try:
            with patch('ultralytics.YOLO',return_value=model),patch.object(scheduler,'_resolve_device',return_value=('0',True)),patch('final_execution.gpu_guard',return_value=nullcontext()):
                scheduler.submit(0,1,self.frame,InferenceOptions(str(path),'Auto',320,.2),0); end=time.monotonic()+5
                while not complete and time.monotonic()<end: self.app.processEvents(); time.sleep(.01)
                self.assertTrue(complete); self.assertEqual(complete[0][1],'CPU fallback'); self.assertEqual(model.predict.call_count,2)
        finally: scheduler.close()
    def test_shared_camera_capture_and_ownership_transfer(self):
        import queue
        from main import CameraCapture,_ACTIVE_CAMERAS
        cap1=CameraCapture(); cap2=CameraCapture()
        fake_proc=Mock(); fake_proc.is_alive.return_value=True; fake_proc.exitcode=None
        ctx1=Mock()
        buf=mp.get_context('spawn').RawArray('B',4*4*3)
        np.frombuffer(buf,dtype=np.uint8).reshape(4,4,3)[:]=77
        seq=mp.get_context('spawn').Value('L',5)
        last_at=mp.get_context('spawn').Value('d',time.monotonic(),lock=False)
        fps_val=mp.get_context('spawn').Value('d',25.0,lock=False)
        fake_queue=Mock(); fake_queue.get_nowait.side_effect=queue.Empty
        ctx1.RawArray.return_value=buf; ctx1.Value.side_effect=[seq,last_at,fps_val]; ctx1.Event.return_value=Mock(); ctx1.Queue.return_value=fake_queue; ctx1.Process.return_value=fake_proc
        cap1._context=ctx1
        try:
            cap1.start(0,4,4,30)
            self.assertIs(_ACTIVE_CAMERAS.get(0),cap1)
            self.assertEqual(ctx1.Process.call_count,1)
            # Second panel starts on same camera index 0 -> attaches without spawning a second process
            cap2.start(0,2,2,30)
            self.assertIs(cap2._shared_primary,cap1)
            s1,f1=cap1.latest(); s2,f2=cap2.latest()
            self.assertEqual(s1,5); self.assertEqual(s2,5)
            self.assertEqual(f1.shape,(4,4,3)); self.assertEqual(f2.shape,(2,2,3))
            self.assertTrue(np.all(f2==77))
            # Stopping cap1 transfers process ownership to cap2 without stopping the camera process
            cap1.stop()
            fake_proc.terminate.assert_not_called()
            self.assertIs(_ACTIVE_CAMERAS.get(0),cap2)
            self.assertIsNone(cap2._shared_primary)
            seq.value=6
            s2_next,f2_next=cap2.latest()
            self.assertEqual(s2_next,6); self.assertEqual(f2_next.shape,(2,2,3))
        finally:
            cap1.stop(); cap2.stop()
            cap1.wait_closed(); cap2.wait_closed()
    def test_open_camera_capture_backend_and_resolution_fallback(self):
        from main import _open_camera_capture
        attempts=[]
        good_frame=np.full((240,320,3),128,dtype=np.uint8)
        class FakeCap:
            def __init__(self,idx,backend):
                self.idx=idx; self.backend=backend; attempts.append(backend)
            def isOpened(self): return len(attempts)>=2
            def set(self,*_args): return True
            def read(self): return (True,good_frame.copy()) if len(attempts)>=2 else (False,None)
            def release(self): pass
        with patch('cv2.VideoCapture',side_effect=FakeCap):
            cap,first=_open_camera_capture(0,640,480,30)
            self.assertIsNotNone(cap); self.assertIsNotNone(first)
            self.assertGreaterEqual(len(attempts),2)
    def test_file_capture_looping(self):
        from final_capture import FileCapture
        video=self.root/'loop.avi'; writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),25,(64,48))
        writer.write(self.frame); writer.release()
        looped=FileCapture(loop=True); looped.start(video)
        try:
            for expected_seq in (1,2,3):
                looped.next_at=0.
                seq,im=looped.latest()
                self.assertEqual(seq,expected_seq)
                self.assertIsNotNone(im)
                self.assertFalse(looped.finished)
        finally:
            looped.stop()
    def test_start_all_with_single_physical_camera_starts_both_panels(self):
        from main import MainWindow,CameraDevice
        defaults={'camera_index':0,'model':str(self.root/'local.pt'),'device':'CPU','capture_width':64,'capture_height':48,'capture_fps':15,'inference_fps':8,'image_size':320,'confidence':.2}
        with patch('main.discover_camera_devices',return_value=[CameraDevice(0,'USB Camera'),CameraDevice(1,'OBS Virtual Camera')]):
            window=MainWindow({'window':{'width':1280,'height':900},'cameras':[defaults,{**defaults,'camera_index':1}],'event_session':str(self.root/'session_dual')})
        window.timer.stop()
        try:
            # Both panels should auto-select the physical camera 0 rather than the inactive virtual camera 1
            self.assertEqual(window.panels[0].selected_camera_index(),0)
            self.assertEqual(window.panels[1].selected_camera_index(),0)
            with patch.object(window.panels[0],'toggle_camera') as t0,patch.object(window.panels[1],'toggle_camera') as t1:
                window._start_all()
                t0.assert_called_once(); t1.assert_called_once()
        finally:
            window.close()

if __name__=='__main__': mp.freeze_support(); unittest.main(verbosity=2)
