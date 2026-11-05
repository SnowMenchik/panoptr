from dataset_lab.bootstrap import setup
setup()
import copy,json,math,unittest
import numpy as np
from camera_design import *

class CameraTests(unittest.TestCase):
    def test_near_projection_matches_ground_scale(self):
        p=preset('near'); p.update({'width_px':2000,'height_px':1000,'hfov_deg':90,'model_size_px':1000,'camera_height_m':2.,'object_width_m':.02,'object_extent_m':.04})
        r=calculate(p); self.assertAlmostEqual(r['target']['sensor_width_px'],10); self.assertAlmostEqual(r['target']['sensor_height_px'],20)
        self.assertAlmostEqual(r['target']['model_width_px'],5)
    def test_near_motion_blur_matches_closed_form(self):
        p=preset('near'); k=intrinsics(p); r=calculate(p)
        expected=k['fy_px']*(p['speed_kmh']/3.6)*(p['exposure_ms']/1000)/p['camera_height_m']
        self.assertAlmostEqual(r['target']['sensor_blur_px'],expected,places=8)
    def test_ground_full_visibility_interval(self):
        p=preset('near'); k=intrinsics(p); geo=geometric_interval(p,k)
        expected=p['camera_height_m']*p['height_px']/k['fy_px']-p['object_extent_m']
        self.assertAlmostEqual(geo['length_m'],expected,places=8)
    def test_exposure_shrinks_visibility_window(self):
        p=preset('near'); r=calculate(p)
        expected=r['coverage']['geometric_window']['length_m']-r['motion']['distance_during_exposure_m']
        self.assertAlmostEqual(r['coverage']['readable_length_m'],expected,places=6)
    def test_near_default_not_enough_frames_at_360(self):
        r=calculate(preset('near')); self.assertFalse(r['checks']['observation_count']); self.assertLess(r['coverage']['expected_observations'],1)
        self.assertFalse(r['all_design_checks_met'])
    def test_near_slow_inspection_satisfies_targets(self):
        p=preset('near'); p['speed_kmh']=20; r=calculate(p); self.assertTrue(r['all_design_checks_met'])
    def test_near_hypothetical_fast_pipeline_satisfies_math_only(self):
        p=preset('near'); p.update({'capture_fps':1000,'analysis_fps':1000,'predict_ms':.5,'pipeline_ms':.2}); r=calculate(p)
        self.assertTrue(r['all_design_checks_met']); self.assertGreater(r['data_rate']['raw_RGB8_MB_s'],6000)
    def test_forward_centre_no_translational_optical_flow(self):
        p=preset('forward'); p['pitch_deg']=0.; k=intrinsics(p)
        points=np.array([[0.,100.,p['camera_height_m']]])
        trail=trajectory(p,k,points,[-.005,.005]); self.assertAlmostEqual(trail_span(trail),0.)
    def test_forward_corner_expansion_not_transverse_formula(self):
        p=preset('forward'); p.update({'width_px':2000,'height_px':1000,'model_size_px':2000,'hfov_deg':90,'camera_height_m':1.,'pitch_deg':0.,'distance_m':100.,'object_width_m':1.,'object_extent_m':2.,'exposure_ms':10.})
        r=calculate(p); expected=math.sqrt(.5**2+1**2)*1000*(1/99.5-1/100.5)
        self.assertAlmostEqual(r['target']['sensor_blur_px'],expected,places=8)
        self.assertLess(r['target']['sensor_blur_px'],.12)
    def test_pitch_rate_stationary_motion(self):
        p=preset('forward'); p.update({'speed_kmh':0,'pitch_deg':0,'pitch_rate_deg_s':100,'exposure_ms':1,'hfov_deg':90,'width_px':2000})
        k=intrinsics(p); points=np.array([[0.,100.,p['camera_height_m']]])
        span=trail_span(trajectory(p,k,points,[-.0005,.0005])); expected=2*k['fy_px']*math.tan(math.radians(.05))
        self.assertAlmostEqual(span,expected)
    def test_downsampling_loses_pixels(self):
        p=preset('forward'); r=calculate(p); self.assertAlmostEqual(r['target']['model_min_px'],r['target']['sensor_min_px']/3)
    def test_upsampling_does_not_create_detail(self):
        p=preset('forward'); p['model_size_px']=8192; r=calculate(p)
        self.assertEqual(r['intrinsics']['model_scale'],1.); self.assertEqual(r['target']['model_min_px'],r['target']['sensor_min_px']); self.assertTrue(r['warnings'])
    def test_forward_300m_resolution_and_lead_fail(self):
        r=calculate(preset('forward')); self.assertFalse(r['checks']['target_resolution']); self.assertFalse(r['checks']['requested_lead_time'])
        self.assertGreater(r['requirements']['sensor_width_px_at_same_aspect_and_fov'],3840)
    def test_narrow_forward_optics_can_meet_example_goals(self):
        p=preset('forward'); p.update({'hfov_deg':7.5,'pitch_deg':math.degrees(math.atan2(2.5-.85,300))}); r=calculate(p)
        self.assertTrue(r['checks']['target_resolution']); self.assertTrue(r['checks']['requested_lead_time']); self.assertTrue(r['all_design_checks_met'])
    def test_serial_predict_caps_throughput(self):
        p=preset('near'); p.update({'capture_fps':1000,'analysis_fps':1000,'predict_ms':20.}); r=calculate(p)
        self.assertEqual(r['coverage']['effective_analysis_fps'],50.)
    def test_capture_rate_caps_analysis(self):
        p=preset('near'); p.update({'capture_fps':30,'analysis_fps':1000,'predict_ms':1.}); r=calculate(p)
        self.assertEqual(r['coverage']['effective_analysis_fps'],30.)
    def test_exposure_period_caps_independent_frames(self):
        p=preset('near'); p.update({'capture_fps':240,'analysis_fps':240,'predict_ms':0,'exposure_ms':10}); r=calculate(p)
        self.assertEqual(r['coverage']['effective_capture_fps_estimate'],100)
    def test_delay_distance_units_and_sampling_wait(self):
        p=preset('forward'); p.update({'predict_ms':100,'pipeline_ms':0,'analysis_fps':30}); r=calculate(p)
        self.assertEqual(r['motion']['speed_m_s'],100); self.assertEqual(r['motion']['distance_during_predict_m'],10)
        self.assertAlmostEqual(r['motion']['distance_before_first_result_m_estimate'],20.02)
    def test_exposure_requirement_closed_form(self):
        p=preset('near'); r=calculate(p); expected=p['max_blur_px']/r['intrinsics']['fy_px']/r['intrinsics']['model_scale']/(p['speed_kmh']/3.6)*p['camera_height_m']*1000
        self.assertAlmostEqual(r['requirements']['max_exposure_ms_at_design_target'],expected,places=7)
    def test_stationary_no_finite_pass_count(self):
        p=preset('near'); p['speed_kmh']=0; r=calculate(p)
        self.assertIsNone(r['coverage']['expected_observations']); self.assertIsNone(r['requirements']['max_exposure_ms_at_design_target']); json.dumps(r,allow_nan=False)
    def test_rolling_readout_separate_from_exposure(self):
        p=preset('near'); p.update({'shutter':'rolling','readout_ms':10}); r=calculate(p)
        self.assertGreater(r['motion']['rolling_full_readout_motion_envelope_model_px'],1000)
        self.assertLess(r['target']['model_blur_px'],2)
        self.assertFalse(r['checks']['rolling_readout_motion_budget'])
    def test_unknown_rolling_readout_cannot_pass_all_checks(self):
        p=preset('near'); p.update({'speed_kmh':20,'shutter':'rolling','readout_ms':0}); r=calculate(p)
        self.assertFalse(r['checks']['rolling_readout_motion_budget']); self.assertFalse(r['all_design_checks_met'])
    def test_invalid_nonfinite_and_negative(self):
        for key,value in [('speed_kmh',-1),('exposure_ms',float('nan')),('camera_height_m',0),('width_px',1.5),('hfov_deg',180)]:
            p=preset('near'); p[key]=value
            with self.assertRaises(ValueError): calculate(p)
    def test_target_offscreen_no_readable_window(self):
        p=preset('near'); p['lateral_offset_m']=10; r=calculate(p)
        self.assertFalse(r['checks']['target_inside_frame']); self.assertEqual(r['coverage']['readable_length_m'],0.)
    def test_calibrated_projection_uses_asymmetric_principal_point(self):
        p=preset('near'); p.update({'intrinsics_mode':'calibrated','fx_px':1500.,'fy_px':1800.,'cx_px':800.,'cy_px':500.})
        k=intrinsics(p); pt=trajectory(p,k,np.array([[0,0,0]],float),[0])[0,0]
        self.assertAlmostEqual(pt[0],800); self.assertAlmostEqual(pt[1],500)
    def test_behind_camera_and_singular_exposure_no_nan(self):
        p=preset('forward'); p.update({'pitch_deg':0,'distance_m':.01,'exposure_ms':10}); r=calculate(p)
        self.assertIsNone(r['target']['sensor_blur_px']); self.assertFalse(r['checks']['target_exposure_blur']); json.dumps(r,allow_nan=False)

if __name__=='__main__': unittest.main()
