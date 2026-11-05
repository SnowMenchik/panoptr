from dataset_lab.bootstrap import setup
setup()
import os,unittest,tempfile,json,math
from pathlib import Path
import numpy as np
TEST_TMP=Path(__file__).resolve().parent/'stand_test_tmp'
TEST_TMP.mkdir(exist_ok=True)
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from stand_physics import engineering,thermal,scale_similarity
from industrial_tiles import LineAssembler,rail_tiles,map_box
from stand_geometry import scene,camera_module,export_meshes,project,box
from stand_renderer import raster
from virtual_stand import SyntheticBench,rehearsal,VirtualStandDialog
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont,QFontDatabase
from PySide6.QtCore import Qt

class EnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.d=engineering()
    def test_line_rate_from_400kmh(self): self.assertAlmostEqual(self.d['line_rate_required_hz'],400/3.6/.0006); self.assertTrue(self.d['line_rate_pass'])
    def test_overspeed_is_rejected_by_envelope(self): self.assertFalse(engineering(450)['line_rate_pass'])
    def test_payload_from_pixel_rate(self): self.assertAlmostEqual(self.d['raw_capture_MB_s'],4096*400/3.6/.0006/1e6); self.assertLess(self.d['camera_link_payload_Gbps'],6.8)
    def test_smear_and_independent_feature_samples(self): self.assertAlmostEqual(self.d['smear_along_px'],5/9); self.assertAlmostEqual(self.d['defect_independent_px_along'],10/.6)
    def test_minimum_exposure_and_period(self): self.assertFalse(engineering(exposure_us=2)['exposure_period_pass']); self.assertFalse(engineering(exposure_us=6)['exposure_period_pass'])
    def test_gpu_capacity_is_requirement_not_claim(self): self.assertFalse(self.d['neural_throughput_verified']); self.assertAlmostEqual(self.d['two_rail_tiles_per_second'],2*self.d['line_rate_required_hz']/448)
    def test_roi_preserves_pixels_but_small_rocks_fail(self): self.assertGreater(self.d['forward_roi_to1280_short_px_at300m'],16); self.assertLess(self.d['forward_full_to1280_short_px_at300m'],16); self.assertLess(self.d['forward_10cm_rock_short_px_at300m'],16)
    def test_finer_stride_raises_gpu_requirement(self): self.assertGreater(engineering(stride_rows=300)['two_rail_tiles_per_second'],self.d['two_rail_tiles_per_second'])
    def test_invalid_geometry(self):
        for args in [{'speed_kmh':0},{'line_pitch_mm':float('nan')},{'tile_rows':0},{'stride_rows':513}]:
            with self.assertRaises(ValueError): engineering(**args)
    def test_thermal_loss_and_warmup_lower_bound(self): t=thermal(); self.assertAlmostEqual(t['estimated_steady_loss_W'],45*.16*4); self.assertGreater(t['no_loss_warmup_lower_bound_s'],1700)
    def test_insufficient_heater(self): self.assertFalse(thermal(heater_w=1)['positive_steady_margin'])
    def test_scale_pixel_and_time_similarity(self):
        a=scale_similarity(.05,100,.01,2,100000); b=scale_similarity(1,100,.01,2,100000)
        self.assertEqual(a['lines_during_object_unchanged'],b['lines_during_object_unchanged']); self.assertEqual(a['pixels_ratio_unchanged'],b['pixels_ratio_unchanged']); self.assertEqual(a['model_speed_m_s'],5)

class StripTests(unittest.TestCase):
    def test_overlap_and_counter_extents(self):
        a=LineAssembler(4,4,3); s=a.push(np.arange(32,dtype=np.uint8).reshape(8,4),10,np.arange(8)/100)
        self.assertEqual([(p.first_line,p.last_line) for p in s],[(10,13),(13,16)]); np.testing.assert_equal(s[0].image[-1],s[1].image[0]); self.assertEqual(s[0].distance_kind,'nominal_pitch_not_measured')
    def test_packet_mutation_does_not_change_pending(self):
        a=LineAssembler(4,4,3); data=np.ones((2,4),np.uint8); a.push(data,0,[0,.01]); data[:]=0; s=a.push(data,2,[.02,.03]); self.assertEqual(int(s[0].image[:2].sum()),8)
    def test_counter_gap_resets_strip(self):
        a=LineAssembler(4,4,3); a.push(np.ones((2,4),np.uint8),0,[0,.01]); s=a.push(np.ones((4,4),np.uint8),7,[.1,.11,.12,.13]); self.assertEqual(s[0].first_line,7); self.assertEqual(a.gaps,1)
    def test_time_reversal_resets_strip(self):
        a=LineAssembler(4,4,3); a.push(np.ones((2,4),np.uint8),0,[1,2]); s=a.push(np.ones((4,4),np.uint8),2,[0,.1,.2,.3]); self.assertEqual(s[0].first_line,2); self.assertEqual(a.gaps,1)
    def test_quality_change_resets(self):
        a=LineAssembler(4,4,3); a.push(np.ones((2,4),np.uint8),0,[0,.01]); s=a.push(np.ones((4,4),np.uint8),2,[.02,.03,.04,.05],True); self.assertEqual(s[0].first_line,2); self.assertEqual(a.gaps,1)
    def test_invalid_packet(self):
        for image,times in [(np.zeros((2,4),float),[0,1]),(np.zeros((2,3),np.uint8),[0,1]),(np.zeros((2,4),np.uint8),[1,1])]:
            with self.assertRaises(ValueError): LineAssembler(4,4,3).push(image,0,times)
    def test_nonfinite_pitch_rejected(self):
        with self.assertRaises(ValueError): LineAssembler(pitch_mm=float('nan'))
    def test_roi_original_coordinate_mapping(self):
        s=LineAssembler(4096,4,3).push(np.zeros((4,4096),np.uint8),0,[0,1,2,3])[0]; tiles=rail_tiles(s); self.assertEqual(map_box([0,0,20,4],tiles[1]),[3334,0,3354,4])
        with self.assertRaises(ValueError): map_box([0,0,1000,4],tiles[0])

class GeometryTests(unittest.TestCase):
    def test_depth_buffer_is_independent_of_scene_order(self):
        objects=[box('far','far',[0,0,0],[2,2,2],(200,20,20)),box('near','near',[0,2,1],[1,1,1],(20,200,20))]
        a=raster(objects,[0,6,3],[0,0,0],160,120); b=raster(list(reversed(objects)),[0,6,3],[0,0,0],160,120); np.testing.assert_equal(a,b); self.assertGreater(a[:,:,1].max(),100)
    def test_all_scene_vertices_finite_and_faces_valid(self):
        for mode in ('bench','platform'):
            for m in scene(mode,explode=1):
                self.assertTrue(np.isfinite(m.vertices).all()); self.assertTrue(all(0<=i<len(m.vertices) for f in m.faces for i in f))
    def test_camera_dimensions_match_selected_body(self):
        for front,size in [(False,[.062,.0369,.062]),(True,[.074,.0788,.074])]:
            body=next(m for m in camera_module('camera',[0,0,0],0,0,front) if m.key=='camera'); np.testing.assert_allclose(np.ptp(body.vertices,axis=0),size)
    def test_gauge_1520(self): rails=sorted([m for m in scene() if m.key=='rail'],key=lambda m:m.vertices[:,0].mean()); self.assertAlmostEqual(rails[1].vertices[:,0].min()-rails[0].vertices[:,0].max(),1.52)
    def test_explode_changes_panels_not_sensor(self):
        a=camera_module('c',[0,0,0],0,0); b=camera_module('c',[0,0,0],0,1)
        np.testing.assert_equal(next(m.vertices for m in a if m.key=='c'),next(m.vertices for m in b if m.key=='c')); self.assertFalse(np.array_equal(a[2].vertices,b[2].vertices))
    def test_export_units_and_nonmanufacturing_status(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as t:
            export_meshes(camera_module('c',[0,0,0]),t,'test'); p=Path(t); data=json.loads((p/'test_parts.json').read_text(encoding='utf-8')); self.assertEqual(data['stl_units'],'millimetres'); self.assertFalse(data['manufacturing_ready']); self.assertGreater((p/'test.obj').stat().st_size,1000)
    def test_projection_center_and_behind(self):
        pixels,depth=project(np.array([[0,0,0.]]),[0,5,2],[0,0,0],800,600); np.testing.assert_allclose(pixels[0],[400,300]); self.assertGreater(depth[0],0)
        self.assertIsNone(project(np.array([[0,10,4.]]),[0,5,2],[0,0,0],800,600)[0])

class ScenarioTests(unittest.TestCase):
    def test_two_independent_synthetic_hits_make_one_card(self):
        b=SyntheticBench(); _,a,_=b.step('person',1.2,0); _,c,_=b.step('person',1.19,.05); self.assertEqual(a[0]['event_id'],c[0]['event_id']); self.assertEqual(c[0]['observation_state'],'observed_repeatedly')
    def test_normal_does_not_emit_alarm(self): self.assertEqual(SyntheticBench().step('normal',1.2,0)[1],[])
    def test_outside_visibility_does_not_emit(self): self.assertEqual(SyntheticBench().step('rock',0,0)[1],[])
    def test_shutter_marks_unobserved_without_fake_clear(self): _,events,row=SyntheticBench().step('person',1.2,0,100,True); self.assertEqual(events,[]); self.assertFalse(row['assessed'])
    def test_low_light_marks_unobserved(self): self.assertFalse(SyntheticBench().step('missing',1.2,0,14)[2]['assessed'])
    def test_rehearsal_records_evidence_and_deterministic_protocol(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as t:
            one=rehearsal(Path(t)/'a'); two=rehearsal(Path(t)/'b'); self.assertEqual(one,two); self.assertEqual([r['event_cards'] for r in one],[0,1,1,1,0,0]); self.assertTrue(list((Path(t)/'a').glob('*/events/*/evidence.png')))
            a=json.loads((Path(t)/'a/protocol.json').read_text()); b=json.loads((Path(t)/'b/protocol.json').read_text()); self.assertEqual(a,b); self.assertFalse(a['real_performance_verified'])

class QtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([]); QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf'); cls.app.setFont(QFont('Segoe UI',9))
    def test_offscreen_3d_controls_and_report(self):
        w=VirtualStandDialog(); w.timer.stop(); w.show(); self.app.processEvents(); self.assertGreater(len(w.canvas.meshes),40)
        w.mode.setCurrentIndex(1); w.kind.setCurrentText('Камень'); w.explode.setValue(100); w.shutter.setChecked(True); self.app.processEvents(); self.assertTrue(any(m.key=='shutter' for m in w.canvas.meshes)); self.assertIn('не измерен',w.report.toPlainText().lower().replace('не измерено','не измерен')); w.close()
    def test_pending_event_indicator_and_reset_finish_session(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as t:
            with patch('virtual_stand.OUT',Path(t)):
                w=VirtualStandDialog(); w.timer.stop(); w.kind.setCurrentText('Человек'); w.position=1.2; w.refresh(observe=True); w.t=.05; w.refresh(observe=True)
                self.assertEqual(len(w.saved),1); self.assertIn('event',w.status.text()); old=w.store.root; w.reset(); self.assertEqual(len(w.saved),0)
                self.assertTrue(json.loads((old/'status.json').read_text())['closed']); self.assertEqual(len(json.loads((old/'virtual_protocol.json').read_text())['rows']),2); w.close()

if __name__=='__main__': unittest.main()
