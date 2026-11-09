"""Unit tests for the automated 5-scenario VSM stress matrix runner."""
from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from run_vsm_stress_matrix import build_stress_scenarios, run_stress_matrix


class TestVsmStressMatrix(unittest.TestCase):
    def test_five_scenarios_defined(self) -> None:
        scenarios = build_stress_scenarios()
        self.assertEqual(len(scenarios), 5)
        speeds = [cfg.speed_kmh for _, _, cfg in scenarios]
        self.assertEqual(speeds, [100, 300, 350, 400, 400])

    def test_run_stress_matrix_generates_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "vsm_stress_matrix"
            summary = run_stress_matrix(out_dir)
            self.assertEqual(summary["total_scenarios"], 5)
            self.assertTrue((out_dir / "stress_matrix_summary.json").is_file())
            self.assertTrue((out_dir / "stress_matrix_summary.csv").is_file())
            self.assertTrue((out_dir / "ground_truth_stress_matrix.json").is_file())

            loaded = json.loads((out_dir / "stress_matrix_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(loaded["gt_verified_runs"]), 5)
            self.assertEqual(len(loaded["hardware_gpu_field_video_runs"]), 5)

            scen4 = loaded["gt_verified_runs"][3]
            self.assertEqual(scen4["scenario_id"], "04_speed_400kmh_fault_recovery")
            self.assertGreater(scen4["camera_off_frames"], 0)
            self.assertIsNotNone(scen4["camera_recovery_sec"])
            self.assertIsNotNone(scen4["gnss_recovery_sec"])
            self.assertGreater(scen4["true_positives"], 0)


if __name__ == "__main__":
    unittest.main()
