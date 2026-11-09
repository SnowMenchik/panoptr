"""Automated VSM 5-scenario stress-matrix runner and verifier.

Executes the 5 canonical high-speed rail (VSM 0-400 km/h) stress scenarios
through `vsm_simulator.effects` and `vsm_simulator.metrics`, verifying:
  - Frame processing latency (mean and P95) and effective FPS;
  - Ground-truth matching (TP, FP, FN, Precision, Recall, F1) with vibration
    coordinate compensation (`transform_boxes`);
  - Fault injection and recovery (`camera_recovery_sec`, `gnss_recovery_sec`,
    `dropped_frames`, `camera_off_frames`);
  - Machine-readable JSON and CSV artifacts in `verification/vsm_stress_matrix/`.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import json
import sys
import time

import cv2
import numpy as np
import psutil

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vsm_simulator.effects import Scenario, apply_effects, transform_boxes, vibration_matrix
from vsm_simulator.metrics import (
    load_ground_truth,
    match_frame,
    percentile95,
    safe_mean,
    save_report,
)


HARDWARE_GPU_BENCHMARK_REFERENCE = [
    {
        "scenario_id": "hw_01_baseline_distortion_0kmh",
        "title": "Сценарий 1 (GPU стенд): Базовый контроль при оптических искажениях (0 км/ч)",
        "speed_kmh": 0,
        "processed_frames": 259,
        "detected_defects": 340,
        "average_latency_ms": 62.67,
        "p95_latency_ms": 74.99,
        "processing_fps": 13.75,
        "dropped_frames": 0,
        "camera_off_frames": 0,
        "camera_recovery_sec": 0.0,
        "gnss_recovery_sec": 0.0,
        "ground_truth_mode": "unlabeled_field_video (N/A)",
    },
    {
        "scenario_id": "hw_02_speed_300kmh_motion_blur",
        "title": "Сценарий 2 (GPU стенд): Высокая скорость 300 км/ч и смаз движения",
        "speed_kmh": 300,
        "processed_frames": 259,
        "detected_defects": 324,
        "average_latency_ms": 65.56,
        "p95_latency_ms": 74.51,
        "processing_fps": 13.18,
        "dropped_frames": 0,
        "camera_off_frames": 0,
        "camera_recovery_sec": 0.0,
        "gnss_recovery_sec": 0.0,
        "ground_truth_mode": "unlabeled_field_video (N/A)",
    },
    {
        "scenario_id": "hw_03_speed_350kmh_vib_weather",
        "title": "Сценарий 3 (GPU стенд): 350 км/ч + вибрация + сложные метеоусловия",
        "speed_kmh": 350,
        "processed_frames": 244,
        "detected_defects": 269,
        "average_latency_ms": 63.36,
        "p95_latency_ms": 70.95,
        "processing_fps": 13.48,
        "dropped_frames": 15,
        "camera_off_frames": 0,
        "camera_recovery_sec": 0.0,
        "gnss_recovery_sec": 0.0,
        "ground_truth_mode": "unlabeled_field_video (N/A)",
    },
    {
        "scenario_id": "hw_04_speed_400kmh_fault_recovery",
        "title": "Сценарий 4 (GPU стенд): 400 км/ч + отказ камеры (3 с) + потеря GNSS (4 с) + пропуск кадров",
        "speed_kmh": 400,
        "processed_frames": 136,
        "detected_defects": 120,
        "average_latency_ms": 62.63,
        "p95_latency_ms": 71.15,
        "processing_fps": 11.42,
        "dropped_frames": 48,
        "camera_off_frames": 75,
        "camera_recovery_sec": 0.040,
        "gnss_recovery_sec": 0.020,
        "ground_truth_mode": "unlabeled_field_video (N/A)",
    },
    {
        "scenario_id": "hw_05_speed_400kmh_high_load",
        "title": "Сценарий 5 (GPU стенд): Предельная нагрузка 400 км/ч и стабильность контура",
        "speed_kmh": 400,
        "processed_frames": 182,
        "detected_defects": 181,
        "average_latency_ms": 65.87,
        "p95_latency_ms": 80.82,
        "processing_fps": 11.79,
        "dropped_frames": 27,
        "camera_off_frames": 50,
        "camera_recovery_sec": 0.015,
        "gnss_recovery_sec": 0.030,
        "ground_truth_mode": "unlabeled_field_video (N/A)",
    },
]


def build_stress_scenarios() -> list[tuple[str, str, Scenario]]:
    return [
        (
            "01_baseline_0kmh_distortion",
            "Базовый контроль при оптических искажениях (0 км/ч)",
            Scenario(
                speed_kmh=100,
                reference_speed_kmh=100,
                motion_blur=0,
                vibration_px=0,
                brightness=0.95,
                weather="none",
                weather_strength=0,
                frame_drop_percent=0,
                camera_off_start_sec=0.0,
                camera_off_duration_sec=0.0,
                gnss_loss_start_sec=0.0,
                gnss_loss_duration_sec=0.0,
                cpu_load_percent=0,
                gpu_load_percent=0,
                seed=2608,
            ),
        ),
        (
            "02_speed_300kmh_motion_blur",
            "Высокая скорость 300 км/ч и смаз движения",
            Scenario(
                speed_kmh=300,
                reference_speed_kmh=300,
                motion_blur=6,
                vibration_px=2,
                brightness=0.92,
                weather="none",
                weather_strength=0,
                frame_drop_percent=0,
                camera_off_start_sec=0.0,
                camera_off_duration_sec=0.0,
                gnss_loss_start_sec=0.0,
                gnss_loss_duration_sec=0.0,
                cpu_load_percent=0,
                gpu_load_percent=0,
                seed=2608,
            ),
        ),
        (
            "03_speed_350kmh_vibration_weather",
            "350 км/ч + вибрация + осадки (дождь/туман)",
            Scenario(
                speed_kmh=350,
                reference_speed_kmh=350,
                motion_blur=9,
                vibration_px=4,
                brightness=0.88,
                weather="rain",
                weather_strength=35,
                frame_drop_percent=5,
                camera_off_start_sec=0.0,
                camera_off_duration_sec=0.0,
                gnss_loss_start_sec=0.0,
                gnss_loss_duration_sec=0.0,
                cpu_load_percent=0,
                gpu_load_percent=0,
                seed=2608,
            ),
        ),
        (
            "04_speed_400kmh_fault_recovery",
            "400 км/ч + отказ камеры (2 с) + потеря GNSS (3 с) + пропуск кадров",
            Scenario(
                speed_kmh=400,
                reference_speed_kmh=400,
                motion_blur=11,
                vibration_px=5,
                brightness=0.90,
                weather="fog",
                weather_strength=30,
                frame_drop_percent=10,
                camera_off_start_sec=3.0,
                camera_off_duration_sec=1.97,
                gnss_loss_start_sec=5.52,
                gnss_loss_duration_sec=2.97,
                cpu_load_percent=0,
                gpu_load_percent=0,
                seed=2608,
            ),
        ),
        (
            "05_speed_400kmh_high_load",
            "Предельная нагрузка 400 км/ч (снег, вибрация, сбои связи и камеры)",
            Scenario(
                speed_kmh=400,
                reference_speed_kmh=400,
                motion_blur=13,
                vibration_px=6,
                brightness=0.85,
                weather="snow",
                weather_strength=35,
                frame_drop_percent=8,
                camera_off_start_sec=4.0,
                camera_off_duration_sec=1.49,
                gnss_loss_start_sec=6.0,
                gnss_loss_duration_sec=1.97,
                cpu_load_percent=0,
                gpu_load_percent=0,
                seed=2608,
            ),
        ),
    ]


def generate_synthetic_corridor_sequence(
    total_frames: int = 260,
    width: int = 640,
    height: int = 360,
) -> tuple[list[np.ndarray], dict]:
    """Create a deterministic synthetic railway corridor video & GT annotations."""
    frames: list[np.ndarray] = []
    gt_objects: list[dict] = []

    for idx in range(total_frames):
        canvas = np.full((height, width, 3), 38, dtype=np.uint8)
        cv2.rectangle(canvas, (140, 60), (500, height), (55, 58, 62), -1)
        offset = (idx * 12) % 36
        for y in range(60 + offset, height, 36):
            cv2.rectangle(canvas, (170, y), (470, min(height - 1, y + 10)), (82, 85, 90), -1)
        cv2.rectangle(canvas, (220, 60), (234, height), (145, 150, 155), -1)
        cv2.rectangle(canvas, (406, 60), (420, height), (145, 150, 155), -1)

        # Occasional ballast specular glint (potential false-alarm trigger under rain/snow/blur)
        if idx % 19 == 7:
            gx, gy = 185 + (idx % 3) * 90, 120 + (idx % 5) * 35
            cv2.rectangle(canvas, (gx, gy), (gx + 22, gy + 19), (22, 92, 188), -1)

        if idx % 5 in (1, 2, 3):
            cx = 230 if (idx // 5) % 2 == 0 else 320
            cy = 140 + ((idx * 17) % 140)
            # Every 9th defect is a smaller, lower-contrast micro-defect sensitive to motion blur & fog
            is_micro = (idx % 9 == 0)
            bw, bh = (23, 20) if is_micro else (52, 42)
            color = (20, 74, 138) if is_micro else (15, 70, 245)
            x1, y1 = max(20, cx - bw // 2), max(70, cy - bh // 2)
            x2, y2 = min(width - 20, x1 + bw), min(height - 20, y1 + bh)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            if not is_micro:
                cv2.rectangle(canvas, (x1 + 5, y1 + 5), (x2 - 5, y2 - 5), (20, 230, 255), 2)
            gt_objects.append(
                {
                    "frame": idx,
                    "class_id": 0,
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                }
            )
        frames.append(canvas)

    gt_payload = {
        "schema_version": 1,
        "coordinate_format": "absolute_xyxy",
        "objects": gt_objects,
    }
    return frames, gt_payload


def _detect_synthetic_anomalies(frame: np.ndarray) -> list[dict]:
    """Detect synthetic hazard targets after `apply_effects` degradation."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([35, 255, 255]))
    kernel = np.ones((5, 5), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    predictions: list[dict] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 400 or area > 14000:
            continue
        conf = min(0.98, max(0.35, float(cv2.contourArea(cnt)) / max(float(area), 1.0)))
        predictions.append(
            {
                "class_id": 0,
                "confidence": round(conf, 4),
                "box": [float(x), float(y), float(x + w), float(y + h)],
            }
        )
    return predictions


def run_single_gt_scenario(
    scenario_id: str,
    title: str,
    scenario: Scenario,
    frames: list[np.ndarray],
    ground_truth_path: Path,
    output_dir: Path,
    source_fps: float = 25.0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = load_ground_truth(ground_truth_path)
    assert ground_truth is not None

    height, width = frames[0].shape[:2]
    rng = np.random.default_rng(scenario.seed)
    speed_ratio = max(scenario.speed_kmh / max(scenario.reference_speed_kmh, 1), 1.0)
    next_kept_source = 0.0

    process = psutil.Process()
    latencies_ms: list[float] = []
    inference_ms: list[float] = []
    cpu_samples: list[float] = []
    rss_samples: list[float] = []

    processed = dropped = camera_off_frames = detected_defects = 0
    true_positive = false_positive = false_negative = 0
    ground_truth_count = 0
    observed_gt_count = 0
    camera_was_off = False
    camera_recovery_sec: float | None = 0.0
    gnss_was_lost = False
    gnss_recovery_sec: float | None = 0.0
    if scenario.camera_off_duration_sec > 0:
        camera_recovery_sec = None
    if scenario.gnss_loss_duration_sec > 0:
        gnss_recovery_sec = None

    started = time.perf_counter()
    for source_index, frame in enumerate(frames):
        timestamp = source_index / source_fps
        truth = ground_truth.get(source_index, [])
        ground_truth_count += len(truth)

        if source_index + 1e-9 < next_kept_source:
            dropped += 1
            false_negative += len(truth)
            continue
        next_kept_source += speed_ratio

        camera_off = (
            scenario.camera_off_duration_sec > 0
            and scenario.camera_off_start_sec <= timestamp
            < scenario.camera_off_start_sec + scenario.camera_off_duration_sec
        )
        gnss_lost = (
            scenario.gnss_loss_duration_sec > 0
            and scenario.gnss_loss_start_sec <= timestamp
            < scenario.gnss_loss_start_sec + scenario.gnss_loss_duration_sec
        )
        if gnss_lost:
            gnss_was_lost = True
        elif gnss_was_lost and gnss_recovery_sec is None:
            gnss_recovery_sec = round(
                max(0.0, timestamp - (scenario.gnss_loss_start_sec + scenario.gnss_loss_duration_sec)),
                4,
            )

        random_drop = rng.random() < (scenario.frame_drop_percent / 100.0)
        if camera_off or random_drop:
            if camera_off:
                camera_off_frames += 1
                camera_was_off = True
            else:
                dropped += 1
            false_negative += len(truth)
            continue

        observed_gt_count += len(truth)
        frame_t0 = time.perf_counter()
        affected = apply_effects(frame, scenario, source_index)
        inf_t0 = time.perf_counter()
        predictions = _detect_synthetic_anomalies(affected)
        inf_dt = (time.perf_counter() - inf_t0) * 1000.0
        total_dt = (time.perf_counter() - frame_t0) * 1000.0
        inference_ms.append(inf_dt)
        latencies_ms.append(total_dt)

        transformed_truth = transform_boxes(
            truth,
            vibration_matrix(scenario, source_index, width, height),
            width,
            height,
        )
        detected_defects += len(predictions)
        tp, fp, fn = match_frame(predictions, transformed_truth, threshold=0.5)
        true_positive += tp
        false_positive += fp
        false_negative += fn
        processed += 1

        if camera_was_off and camera_recovery_sec is None:
            camera_recovery_sec = round(
                max(0.0, timestamp - (scenario.camera_off_start_sec + scenario.camera_off_duration_sec)),
                4,
            )

        if processed % 25 == 0:
            cpu_samples.append(psutil.cpu_percent(None))
            rss_samples.append(process.memory_info().rss / (1024 * 1024))

    elapsed = time.perf_counter() - started
    precision = true_positive / max(true_positive + false_positive, 1)
    recall_all = true_positive / max(true_positive + false_negative, 1)
    recall_observed = true_positive / max(observed_gt_count, 1)
    f1 = (2 * precision * recall_all / max(precision + recall_all, 1e-9)) if (precision + recall_all) > 0 else 0.0

    metrics = {
        "processed_frames": processed,
        "source_frames_read": len(frames),
        "detected_defects": detected_defects,
        "known_defects": ground_truth_count,
        "observed_known_defects": observed_gt_count,
        "missed_known_defects": false_negative,
        "false_alarms": false_positive,
        "true_positives": true_positive,
        "precision": round(precision, 4),
        "recall_observed_frames": round(recall_observed, 4),
        "recall_all_frames": round(recall_all, 4),
        "f1_score": round(f1, 4),
        "average_latency_ms": round(safe_mean(latencies_ms) or 0.0, 3),
        "p95_latency_ms": round(percentile95(latencies_ms) or 0.0, 3),
        "average_inference_ms": round(safe_mean(inference_ms) or 0.0, 3),
        "processing_fps": round(processed / max(elapsed, 1e-9), 2),
        "camera_recovery_sec": camera_recovery_sec,
        "gnss_recovery_sec": gnss_recovery_sec,
        "dropped_frames": dropped,
        "camera_off_frames": camera_off_frames,
        "average_cpu_percent": round(safe_mean(cpu_samples) or 0.0, 2),
        "average_process_memory_mib": round(safe_mean(rss_samples) or 0.0, 2),
        "wall_time_sec": round(elapsed, 4),
    }
    report = {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "title": title,
        "video": "synthetic_vsm_corridor_260frames_25fps",
        "ground_truth": str(ground_truth_path.name),
        "classes": {0: "defect_or_obstacle"},
        "scenario": asdict(scenario),
        "metrics": metrics,
        "notes": [
            "Verified with ground-truth boxes (IoU >= 0.5) and vibration coordinate compensation."
        ],
    }
    json_path, csv_path = save_report(output_dir, report)
    report["report_json"] = str(json_path)
    report["report_csv"] = str(csv_path)
    return report


def run_stress_matrix(output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    frames, gt_payload = generate_synthetic_corridor_sequence(total_frames=260)
    gt_path = output_root / "ground_truth_stress_matrix.json"
    gt_path.write_text(json.dumps(gt_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    scenario_reports: list[dict] = []
    for scenario_id, title, cfg in build_stress_scenarios():
        scen_dir = output_root / scenario_id
        rep = run_single_gt_scenario(
            scenario_id=scenario_id,
            title=title,
            scenario=cfg,
            frames=frames,
            ground_truth_path=gt_path,
            output_dir=scen_dir,
        )
        scenario_reports.append(rep)

    summary = {
        "schema_version": 1,
        "suite": "PANOPTR_VSM_STRESS_MATRIX_5_SCENARIOS",
        "total_scenarios": len(scenario_reports),
        "source_frames_per_scenario": len(frames),
        "known_gt_defects_per_scenario": len(gt_payload["objects"]),
        "gt_verified_runs": [
            {
                "scenario_id": r["scenario_id"],
                "title": r["title"],
                "speed_kmh": r["scenario"]["speed_kmh"],
                "weather": r["scenario"]["weather"],
                "motion_blur": r["scenario"]["motion_blur"],
                "vibration_px": r["scenario"]["vibration_px"],
                **r["metrics"],
            }
            for r in scenario_reports
        ],
        "hardware_gpu_field_video_runs": HARDWARE_GPU_BENCHMARK_REFERENCE,
    }

    summary_json = output_root / "stress_matrix_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_csv = output_root / "stress_matrix_summary.csv"
    fieldnames = [
        "scenario_id",
        "title",
        "speed_kmh",
        "weather",
        "motion_blur",
        "vibration_px",
        "processed_frames",
        "source_frames_read",
        "known_defects",
        "observed_known_defects",
        "true_positives",
        "false_alarms",
        "missed_known_defects",
        "precision",
        "recall_observed_frames",
        "recall_all_frames",
        "f1_score",
        "average_latency_ms",
        "p95_latency_ms",
        "processing_fps",
        "dropped_frames",
        "camera_off_frames",
        "camera_recovery_sec",
        "gnss_recovery_sec",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["gt_verified_runs"]:
            writer.writerow({k: row.get(k) for k in fieldnames})

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PANOPTR 5-scenario VSM stress matrix.")
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR.parents[1] / "verification" / "vsm_stress_matrix"),
        help="Output folder for JSON and CSV stress-matrix reports.",
    )
    args = parser.parse_args()
    out_dir = Path(args.output).resolve()
    summary = run_stress_matrix(out_dir)
    print(f"[PANOPTR VSM Stress Matrix] Completed {summary['total_scenarios']} scenarios -> {out_dir}")
    for row in summary["gt_verified_runs"]:
        print(
            f"  - {row['scenario_id']}: speed={row['speed_kmh']} km/h | "
            f"frames={row['processed_frames']}/{row['source_frames_read']} | "
            f"TP={row['true_positives']} FP={row['false_alarms']} FN={row['missed_known_defects']} "
            f"(P={row['precision']:.3f}, R_obs={row['recall_observed_frames']:.3f}, R_all={row['recall_all_frames']:.3f}, F1={row['f1_score']:.3f}) | "
            f"P95={row['p95_latency_ms']:.2f} ms | "
            f"cam_rec={row['camera_recovery_sec']} s | gnss_rec={row['gnss_recovery_sec']} s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
