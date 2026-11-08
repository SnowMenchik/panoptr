from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Scenario:
    speed_kmh: int = 300
    reference_speed_kmh: int = 300
    motion_blur: int = 0
    vibration_px: int = 0
    brightness: float = 1.0
    weather: str = "none"
    weather_strength: int = 0
    frame_drop_percent: int = 0
    camera_off_start_sec: float = 0.0
    camera_off_duration_sec: float = 0.0
    gnss_loss_start_sec: float = 0.0
    gnss_loss_duration_sec: float = 0.0
    cpu_load_percent: int = 0
    gpu_load_percent: int = 0
    seed: int = 2608


def _motion_blur(frame: np.ndarray, strength: int, speed_ratio: float) -> np.ndarray:
    length = int(round(strength * speed_ratio))
    if length < 2:
        return frame
    length = min(length, 41)
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length), dtype=np.float32)
    # Forward-facing rail footage has a dominant diagonal/radial component.
    cv2.line(kernel, (0, length - 1), (length - 1, 0), 1.0, 1)
    kernel /= max(kernel.sum(), 1.0)
    return cv2.filter2D(frame, -1, kernel)


def _weather(frame: np.ndarray, weather: str, strength: int, rng: np.random.Generator) -> np.ndarray:
    if weather == "none" or strength <= 0:
        return frame
    amount = np.clip(strength / 100.0, 0.0, 1.0)
    height, width = frame.shape[:2]
    output = frame.copy()
    if weather == "fog":
        fog = np.full_like(output, 215)
        return cv2.addWeighted(output, 1.0 - 0.65 * amount, fog, 0.65 * amount, 0)
    if weather == "rain":
        overlay = np.zeros_like(output)
        count = int((width * height / 6500) * (0.3 + amount * 1.7))
        for _ in range(count):
            x = int(rng.integers(0, width))
            y = int(rng.integers(0, height))
            length = int(rng.integers(8, 22))
            cv2.line(overlay, (x, y), (min(width - 1, x + 3), min(height - 1, y + length)), (190, 190, 190), 1)
        output = cv2.addWeighted(output, 1.0, overlay, 0.35 + 0.35 * amount, 0)
        return cv2.GaussianBlur(output, (3, 3), 0.5 * amount)
    if weather == "snow":
        overlay = np.zeros_like(output)
        count = int((width * height / 3500) * (0.3 + amount * 1.7))
        for _ in range(count):
            x = int(rng.integers(0, width))
            y = int(rng.integers(0, height))
            radius = int(rng.integers(1, 4))
            cv2.circle(overlay, (x, y), radius, (245, 245, 245), -1)
        return cv2.addWeighted(output, 1.0, overlay, 0.35 + 0.45 * amount, 0)
    return output


def vibration_matrix(
    scenario: Scenario, source_frame: int, width: int, height: int
) -> np.ndarray | None:
    if scenario.vibration_px <= 0:
        return None
    rng = np.random.default_rng(scenario.seed + source_frame * 104729)
    dx = float(rng.uniform(-scenario.vibration_px, scenario.vibration_px))
    dy = float(rng.uniform(-scenario.vibration_px, scenario.vibration_px))
    angle = float(rng.uniform(-0.15, 0.15) * scenario.vibration_px)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    matrix[:, 2] += (dx, dy)
    return matrix


def transform_boxes(boxes: list[dict], matrix: np.ndarray | None, width: int, height: int) -> list[dict]:
    if matrix is None:
        return boxes
    transformed = []
    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32)
        moved = corners @ matrix.T
        moved_x, moved_y = moved[:, 0], moved[:, 1]
        transformed.append(
            {
                "class_id": item["class_id"],
                "box": [
                    float(np.clip(moved_x.min(), 0, width - 1)),
                    float(np.clip(moved_y.min(), 0, height - 1)),
                    float(np.clip(moved_x.max(), 0, width - 1)),
                    float(np.clip(moved_y.max(), 0, height - 1)),
                ],
            }
        )
    return transformed


def apply_effects(frame: np.ndarray, scenario: Scenario, source_frame: int) -> np.ndarray:
    speed_ratio = scenario.speed_kmh / max(scenario.reference_speed_kmh, 1)
    output = _motion_blur(frame, scenario.motion_blur, speed_ratio)
    height, width = output.shape[:2]
    matrix = vibration_matrix(scenario, source_frame, width, height)
    if matrix is not None:
        output = cv2.warpAffine(output, matrix, (width, height), borderMode=cv2.BORDER_REFLECT101)
    output = cv2.convertScaleAbs(output, alpha=scenario.brightness, beta=0)
    rng = np.random.default_rng(scenario.seed + source_frame * 104729 + 1)
    return _weather(output, scenario.weather, scenario.weather_strength, rng)
