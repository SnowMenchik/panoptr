from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import psutil

from effects import Scenario, apply_effects, transform_boxes, vibration_matrix
from metrics import load_ground_truth, match_frame, percentile95, safe_mean, save_report


ROOT = Path(__file__).resolve().parent


def _gpu_sample() -> tuple[float | None, float | None, float | None]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        values = [float(value.strip()) for value in output.splitlines()[0].split(",")]
        return values[0], values[1], values[2]
    except Exception:
        return None, None, None


def _terminate_tree(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        psutil.wait_procs(children, timeout=3)
        parent.terminate()
        parent.wait(timeout=3)
    except Exception:
        process.terminate()


def _start_load(kind: str, percent: int) -> subprocess.Popen | None:
    if percent <= 0:
        return None
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        [sys.executable, str(ROOT / "load_injector.py"), kind, str(percent)],
        cwd=ROOT,
        creationflags=flags,
    )


def _predictions(result) -> list[dict]:
    if result.boxes is None:
        return []
    rows = result.boxes.data.detach().cpu().numpy()
    return [
        {"box": [float(value) for value in row[:4]], "confidence": float(row[4]), "class_id": int(row[5])}
        for row in rows
    ]


def run_simulation(**arguments):
    from final_execution import gpu_guard
    path=Path(arguments['model_path']); path=path if path.is_absolute() else ROOT.parent/path
    if not path.is_file(): raise ValueError('Нужны локальные веса; автоскачивание отключено.')
    arguments['model_path']=str(path)
    with gpu_guard('cpu' if arguments['device_request']=='CPU' else '0'):
        return _run_simulation(**arguments)


def _run_simulation(
    *,
    video_path: str,
    model_path: str,
    ground_truth_path: str,
    output_folder: str,
    scenario: Scenario,
    device_request: str,
    image_size: int,
    confidence: float,
    stop_requested: Callable[[], bool],
    progress: Callable[[int, str], None],
    preview: Callable[[np.ndarray], None],
) -> dict:
    import torch
    from ultralytics import YOLO

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    source_fps = float(video.get(cv2.CAP_PROP_FPS)) or 25.0
    source_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("Видео не содержит корректного размера кадра")

    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output / "vsm_processed_video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Не удалось создать выходной MP4")

    ground_truth = load_ground_truth(ground_truth_path)
    if device_request == "CPU":
        device, half = "cpu", False
    elif torch.cuda.is_available():
        device, half = "0", True
    elif device_request == "GPU":
        raise RuntimeError("Выбран GPU, но NVIDIA CUDA недоступна")
    else:
        device, half = "cpu", False

    model = YOLO(model_path)
    cpu_injector = _start_load("cpu", scenario.cpu_load_percent)
    gpu_injector = _start_load("gpu", scenario.gpu_load_percent) if device != "cpu" else None
    process = psutil.Process()
    psutil.cpu_percent(None)
    process.cpu_percent(None)

    latencies_ms: list[float] = []
    inference_ms: list[float] = []
    cpu_samples: list[float] = []
    gpu_samples: list[float] = []
    gpu_memory_samples: list[float] = []
    gpu_temperature_samples: list[float] = []
    rss_samples: list[float] = []
    system_memory_samples: list[float] = []
    processed = dropped = camera_off_frames = detected_defects = 0
    true_positive = false_positive = false_negative = 0
    ground_truth_count = 0
    started = time.perf_counter()
    last_resource_sample = 0.0
    camera_recovery_sec: float | None = None
    camera_was_off = False
    gnss_was_lost = False
    gnss_recovery_sec: float | None = None
    rng = np.random.default_rng(scenario.seed)
    speed_ratio = max(scenario.speed_kmh / max(scenario.reference_speed_kmh, 1), 1.0)
    next_kept_source = 0.0
    names: dict[int, str] = {}

    try:
        source_index = -1
        while True:
            if stop_requested():
                raise InterruptedError("Остановлено пользователем; частичный отчёт не сформирован")
            ok, frame = video.read()
            if not ok:
                break
            source_index += 1
            timestamp = source_index / source_fps
            percent = int(min(100, (source_index + 1) * 100 / max(source_frames, 1)))
            progress(percent, f"Кадр {source_index + 1}/{source_frames}")
            truth = ground_truth.get(source_index, []) if ground_truth is not None else []
            ground_truth_count += len(truth)

            # Higher virtual speed covers more track per captured frame. Source
            # frames skipped here remain false negatives when GT is available.
            if source_index + 1e-9 < next_kept_source:
                dropped += 1
                if ground_truth is not None:
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
                gnss_recovery_sec = max(0.0, timestamp - (scenario.gnss_loss_start_sec + scenario.gnss_loss_duration_sec))

            random_drop = rng.random() < scenario.frame_drop_percent / 100.0
            if camera_off or random_drop:
                black = np.zeros_like(frame)
                text = "CAMERA OFF" if camera_off else "FRAME DROPPED"
                cv2.putText(black, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                writer.write(black)
                if camera_off:
                    camera_off_frames += 1
                    camera_was_off = True
                else:
                    dropped += 1
                if ground_truth is not None:
                    false_negative += len(truth)
                continue

            frame_started = time.perf_counter()
            affected = apply_effects(frame, scenario, source_index)
            inference_started = time.perf_counter()
            result = model.predict(
                source=affected,
                imgsz=image_size,
                conf=confidence,
                device=device,
                quantize=16 if half else None,
                verbose=False,
            )[0]
            inference_ms.append((time.perf_counter() - inference_started) * 1000.0)
            latencies_ms.append((time.perf_counter() - frame_started) * 1000.0)
            predictions = _predictions(result)
            transformed_truth = transform_boxes(
                truth,
                vibration_matrix(scenario, source_index, width, height),
                width,
                height,
            )
            names = {int(key): str(value) for key, value in result.names.items()}
            defect_predictions = [
                item
                for item in predictions
                if names.get(item["class_id"], "").casefold()
                not in {"normal", "good", "intact", "background"}
            ]
            detected_defects += len(defect_predictions)
            if ground_truth is not None:
                tp, fp, fn = match_frame(defect_predictions, transformed_truth)
                true_positive += tp
                false_positive += fp
                false_negative += fn
            processed += 1
            if camera_was_off and camera_recovery_sec is None:
                camera_recovery_sec = max(
                    0.0,
                    timestamp - (scenario.camera_off_start_sec + scenario.camera_off_duration_sec),
                )

            plotted = result.plot()
            if gnss_lost:
                cv2.putText(plotted, "GNSS LOST", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            writer.write(plotted)
            if processed % 5 == 0:
                preview(plotted.copy())
            now = time.monotonic()
            if now - last_resource_sample >= 1.0:
                cpu_samples.append(psutil.cpu_percent(None))
                rss_samples.append(process.memory_info().rss / 2**20)
                system_memory_samples.append(psutil.virtual_memory().percent)
                gpu_util, gpu_memory, gpu_temperature = _gpu_sample()
                if gpu_util is not None:
                    gpu_samples.append(gpu_util)
                if gpu_memory is not None:
                    gpu_memory_samples.append(gpu_memory)
                if gpu_temperature is not None:
                    gpu_temperature_samples.append(gpu_temperature)
                last_resource_sample = now
    finally:
        video.release()
        writer.release()
        _terminate_tree(cpu_injector)
        _terminate_tree(gpu_injector)

    elapsed = time.perf_counter() - started
    metrics = {
        "processed_frames": processed,
        "source_frames_read": source_index + 1,
        "detected_defects": detected_defects,
        "known_defects": ground_truth_count if ground_truth is not None else None,
        "missed_known_defects": false_negative if ground_truth is not None else None,
        "false_alarms": false_positive if ground_truth is not None else None,
        "true_positives": true_positive if ground_truth is not None else None,
        "average_latency_ms": safe_mean(latencies_ms),
        "p95_latency_ms": percentile95(latencies_ms),
        "average_inference_ms": safe_mean(inference_ms),
        "processing_fps": processed / max(elapsed, 1e-9),
        "camera_recovery_sec": camera_recovery_sec,
        "gnss_recovery_sec": gnss_recovery_sec,
        "dropped_frames": dropped,
        "camera_off_frames": camera_off_frames,
        "average_cpu_percent": safe_mean(cpu_samples),
        "average_gpu_percent": safe_mean(gpu_samples),
        "p95_gpu_percent": percentile95(gpu_samples),
        "peak_gpu_memory_mib": max(gpu_memory_samples) if gpu_memory_samples else None,
        "peak_gpu_temperature_c": max(gpu_temperature_samples) if gpu_temperature_samples else None,
        "average_process_memory_mib": safe_mean(rss_samples),
        "peak_process_memory_mib": max(rss_samples) if rss_samples else None,
        "average_system_memory_percent": safe_mean(system_memory_samples),
        "wall_time_sec": elapsed,
    }
    report = {
        "schema_version": 1,
        "video": str(Path(video_path).resolve()),
        "model": str(Path(model_path).resolve()),
        "ground_truth": str(Path(ground_truth_path).resolve()) if ground_truth_path else None,
        "device": "GPU" if device != "cpu" else "CPU",
        "image_size": image_size,
        "confidence": confidence,
        "classes": names,
        "scenario": asdict(scenario),
        "metrics": metrics,
        "notes": [] if ground_truth is not None else [
            "No ground truth was supplied; missed defects and false alarms are intentionally reported as N/A."
        ],
    }
    json_path, csv_path = save_report(output, report)
    report["report_json"] = str(json_path)
    report["report_csv"] = str(csv_path)
    return report
