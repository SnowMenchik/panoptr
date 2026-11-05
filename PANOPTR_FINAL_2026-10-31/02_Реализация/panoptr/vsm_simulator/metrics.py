from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


METRIC_LABELS = {
    "processed_frames": "Обработано кадров",
    "source_frames_read": "Прочитано исходных кадров",
    "detected_defects": "Найдено дефектов",
    "known_defects": "Известно дефектов в эталоне",
    "missed_known_defects": "Пропущено известных дефектов",
    "false_alarms": "Ложные тревоги",
    "true_positives": "Верные обнаружения",
    "average_latency_ms": "Средняя задержка, мс",
    "p95_latency_ms": "Задержка p95, мс",
    "average_inference_ms": "Среднее время inference, мс",
    "processing_fps": "FPS обработки",
    "camera_recovery_sec": "Восстановление камеры, с",
    "gnss_recovery_sec": "Восстановление GNSS, с",
    "dropped_frames": "Пропущено кадров",
    "camera_off_frames": "Кадров отключения камеры",
    "average_cpu_percent": "Средняя нагрузка CPU, %",
    "average_gpu_percent": "Средняя нагрузка GPU, %",
    "p95_gpu_percent": "Нагрузка GPU p95, %",
    "peak_gpu_memory_mib": "Пиковая память GPU, МиБ",
    "peak_gpu_temperature_c": "Пиковая температура GPU, °C",
    "average_process_memory_mib": "Средняя память процесса, МиБ",
    "peak_process_memory_mib": "Пиковая память процесса, МиБ",
    "average_system_memory_percent": "Средняя загрузка RAM, %",
    "wall_time_sec": "Время прогона, с",
}


def load_ground_truth(path: str) -> dict[int, list[dict]] | None:
    if not path:
        return None
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    frames: dict[int, list[dict]] = {}
    for item in document.get("objects", []):
        frame = int(item["frame"])
        box = [float(value) for value in item["bbox_xyxy"]]
        if len(box) != 4:
            raise ValueError("Each bbox_xyxy must contain four values")
        frames.setdefault(frame, []).append({"class_id": int(item["class_id"]), "box": box})
    return frames


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(area_left + area_right - intersection, 1e-9)


def match_frame(predictions: list[dict], truth: list[dict], threshold: float = 0.5) -> tuple[int, int, int]:
    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, target in enumerate(truth):
            if prediction["class_id"] == target["class_id"]:
                score = iou(prediction["box"], target["box"])
                if score >= threshold:
                    candidates.append((score, prediction_index, truth_index))
    matched_predictions, matched_truth = set(), set()
    for _, prediction_index, truth_index in sorted(candidates, reverse=True):
        if prediction_index not in matched_predictions and truth_index not in matched_truth:
            matched_predictions.add(prediction_index)
            matched_truth.add(truth_index)
    true_positive = len(matched_truth)
    return true_positive, len(predictions) - true_positive, len(truth) - true_positive


def percentile95(values: list[float]) -> float | None:
    return float(np.percentile(values, 95)) if values else None


def safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def save_report(folder: Path, report: dict) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / "vsm_simulation_report.json"
    csv_path = folder / "vsm_simulation_report.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["Показатель", "Значение"])
        for key, value in report["metrics"].items():
            writer.writerow([METRIC_LABELS.get(key, key), "N/A" if value is None else value])
    return json_path, csv_path
