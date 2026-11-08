from __future__ import annotations

import json
import multiprocessing as mp
import queue
import sys
import threading
import time
from dataclasses import dataclass
from dataclasses import replace
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import psutil
from PySide6.QtCore import QObject, QProcess, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QDialog,QPlainTextEdit,QFormLayout,QScrollArea,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@dataclass(frozen=True)
class InferenceOptions:
    model: str
    device: str
    image_size: int
    confidence: float
    path_enabled: bool = False
    geometry_model: str = ''
    geometry_confidence: float = .35
    geometry_image_size: int = 640
    geometry_source_id: str = ''


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str

    @property
    def is_virtual(self) -> bool:
        normalized = self.name.casefold()
        markers = ("broadcast", "virtual", "obs", "manycam", "snap camera")
        return any(marker in normalized for marker in markers)


@dataclass(frozen=True)
class DetectionPacket:
    boxes: np.ndarray
    names: dict
    mask_overlay: np.ndarray | None = None
    keypoints: np.ndarray | None = None
    sequence: int = -1
    generation: int = -1
    captured_at: float = 0.
    completed_at: float = 0.
    options: InferenceOptions | None = None
    source_frame: np.ndarray | None = None
    masks: list | None = None
    model_sha256: str = ''
    model_load_ms: float = 0.
    geometry: dict | None = None

@dataclass(frozen=True)
class InferenceJob:
    channel: int
    sequence: int
    generation: int
    captured_at: float
    frame: np.ndarray
    options: InferenceOptions


def discover_camera_devices() -> list[CameraDevice]:
    """Returns user-facing camera names in the DirectShow/OpenCV index order."""
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph

            names = FilterGraph().get_input_devices()
            if names:
                return [CameraDevice(index, name) for index, name in enumerate(names)]
        except Exception:
            pass

    try:
        from PySide6.QtMultimedia import QMediaDevices

        qt_devices = QMediaDevices.videoInputs()
        if qt_devices:
            return [
                CameraDevice(index, device.description())
                for index, device in enumerate(qt_devices)
            ]
    except Exception:
        pass

    return []


def _log_camera_debug(msg: str) -> None:
    try:
        log_dir = ROOT.parents[1] / "data"
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "camera_debug.log").open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def _open_camera_capture(index, width: int, height: int, fps: int):
    """Opens a camera or stream with backend, codec (MJPG/native/default), and MSMF index mapping."""
    if sys.platform == "win32" and isinstance(index, int):
        msmf = getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)
        # Compute physical camera ordinal in case DirectShow has virtual cameras (e.g. OBS at ID 0)
        # that are skipped by Media Foundation (CAP_MSMF) or CAP_ANY.
        alt_indices = [index]
        try:
            devices = discover_camera_devices()
            physical = [d.index for d in devices if not d.is_virtual]
            if index in physical:
                phys_ord = physical.index(index)
                if phys_ord not in alt_indices:
                    alt_indices.append(phys_ord)
            for probe_idx in (0, 1, 2, 3):
                if probe_idx != index and probe_idx not in alt_indices and probe_idx not in [d.index for d in devices if d.is_virtual]:
                    pass
        except Exception:
            pass

        # (idx_to_open, backend, backend_name, mode)
        # mode: 'mjpg' = set width/height + MJPG, 'native' = set width/height only, 'raw' = no .set() calls at all
        attempts = [
            (index, cv2.CAP_DSHOW, "DSHOW", "mjpg"),
            (index, cv2.CAP_DSHOW, "DSHOW", "raw"),
            (index, cv2.CAP_DSHOW, "DSHOW", "native"),
            (index, msmf, "MSMF", "raw"),
            (index, msmf, "MSMF", "mjpg"),
        ]
        for alt_idx in alt_indices:
            if alt_idx != index:
                attempts.append((alt_idx, msmf, f"MSMF(phys={alt_idx})", "raw"))
                attempts.append((alt_idx, msmf, f"MSMF(phys={alt_idx})", "mjpg"))
        attempts.append((index, cv2.CAP_ANY, "ANY", "raw"))
    else:
        attempts = [
            (index, cv2.CAP_ANY, "ANY", "mjpg" if isinstance(index, int) else "raw"),
            (index, cv2.CAP_ANY, "ANY", "raw"),
        ]

    for open_idx, backend, bname, mode in attempts:
        t0 = time.perf_counter()
        capture = cv2.VideoCapture(open_idx, backend)
        if not capture.isOpened():
            _log_camera_debug(f"cam={index} try idx={open_idx} {bname}/{mode}: isOpened=False ({time.perf_counter()-t0:.2f}s)")
            capture.release()
            continue
        try:
            if mode in ("mjpg", "native") and isinstance(open_idx, int):
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                if mode == "mjpg":
                    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        for read_try in range(6):
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                _log_camera_debug(
                    f"cam={index} SUCCESS idx={open_idx} {bname}/{mode} shape={frame.shape} on read #{read_try+1} ({time.perf_counter()-t0:.2f}s)"
                )
                return capture, frame
            time.sleep(0.03)
        _log_camera_debug(f"cam={index} try idx={open_idx} {bname}/{mode}: opened but read() returned False ({time.perf_counter()-t0:.2f}s)")
        capture.release()
    return None, None


def camera_process_main(
    index: int,
    width: int,
    height: int,
    fps: int,
    shared_buffer,
    sequence,
    last_frame_at,
    actual_fps,
    stop_event,
    errors,
) -> None:
    """Owns one Windows camera and all of its driver calls in an isolated process."""
    source_label=f'Камера {index}' if isinstance(index,int) else 'Видеопоток'
    while not stop_event.is_set():
        capture, first_frame = _open_camera_capture(index, width, height, fps)
        if capture is None:
            errors.put(f"{source_label} не открылся (см. data/camera_debug.log); повтор подключения…")
            stop_event.wait(1.0)
            continue

        count = 0
        failures = 0
        measured_at = time.monotonic()
        pending_frame = first_frame
        try:
            while not stop_event.is_set():
                if pending_frame is not None:
                    ok, frame = True, pending_frame
                    pending_frame = None
                else:
                    ok, frame = capture.read()
                if not ok or frame is None or not frame.size:
                    failures += 1
                    if failures >= 20:
                        errors.put(f"{source_label}: потерян поток; переподключение…")
                        break
                    stop_event.wait(0.05)
                    continue
                failures = 0
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                target = np.frombuffer(shared_buffer, dtype=np.uint8).reshape(height, width, 3)
                with sequence.get_lock():
                    np.copyto(target, frame)
                    last_frame_at.value = time.monotonic()
                    sequence.value += 1
                count += 1
                now = time.monotonic()
                if now - measured_at >= 1.0:
                    actual_fps.value = count / (now - measured_at)
                    count = 0
                    measured_at = now
        finally:
            capture.release()
        if not stop_event.is_set():
            stop_event.wait(0.5)


_ACTIVE_CAMERAS_LOCK = threading.Lock()
_ACTIVE_CAMERAS: dict[object, "CameraCapture"] = {}


class CameraCapture:
    """Reads the newest frame from a dedicated per-camera child process (or shares an already-open camera across panels)."""

    def __init__(self) -> None:
        self._context = mp.get_context("spawn")
        self._process = None
        self._stop_event = None
        self._buffer = None
        self._sequence = None
        self._last_frame_at = None
        self._actual_fps = None
        self._errors = None
        self._cached_sequence = -1
        self._cached_frame: np.ndarray | None = None
        self.captured_at = 0.
        self.error: str | None = None
        self.started_at = 0.0
        self.width = 0
        self.height = 0
        self._buffer_width = 0
        self._buffer_height = 0
        self._cleanup_threads=[]
        self._camera_key = None
        self._shared_primary: CameraCapture | None = None
        self._subscribers: set[CameraCapture] = set()

    def start(self, index: int, width: int, height: int, fps: int) -> None:
        self.stop()
        self.width, self.height = width, height
        self._buffer_width, self._buffer_height = width, height
        self.started_at = time.monotonic()
        self.error = None
        self._cached_sequence = -1
        self._cached_frame = None
        self.captured_at = 0.
        self._camera_key = index
        with _ACTIVE_CAMERAS_LOCK:
            primary = _ACTIVE_CAMERAS.get(index)
            if (
                primary is not None
                and primary is not self
                and primary._process is not None
                and primary._process.is_alive()
                and primary._buffer is not None
                and primary._sequence is not None
            ):
                self._shared_primary = primary
                primary._subscribers.add(self)
                self._buffer = primary._buffer
                self._sequence = primary._sequence
                self._last_frame_at = primary._last_frame_at
                self._actual_fps = primary._actual_fps
                self._buffer_width = primary._buffer_width or primary.width
                self._buffer_height = primary._buffer_height or primary.height
                return
            self._shared_primary = None
            self._subscribers = {self}
            _ACTIVE_CAMERAS[index] = self
        self._buffer = self._context.RawArray("B", width * height * 3)
        self._sequence = self._context.Value("L", 0)
        self._last_frame_at = self._context.Value("d", self.started_at, lock=False)
        self._actual_fps = self._context.Value("d", 0.0, lock=False)
        self._stop_event = self._context.Event()
        self._errors = self._context.Queue()
        self._process = self._context.Process(
            target=camera_process_main,
            args=(
                index,
                width,
                height,
                fps,
                self._buffer,
                self._sequence,
                self._last_frame_at,
                self._actual_fps,
                self._stop_event,
                self._errors,
            ),
            daemon=True,
            name=f"camera-{index}",
        )
        self._process.start()

    def stop(self) -> None:
        with _ACTIVE_CAMERAS_LOCK:
            if self._shared_primary is not None:
                self._shared_primary._subscribers.discard(self)
                self._shared_primary = None
                self._camera_key = None
                self._process = None; self._errors = None; self._buffer = None; self._sequence = None; self._last_frame_at = None; self._actual_fps = None
                self._cached_frame = None; self._cached_sequence = -1; self.captured_at = 0.
                return
            other_subscribers = [sub for sub in self._subscribers if sub is not self]
            if other_subscribers and self._process is not None:
                new_primary = other_subscribers[0]
                new_primary._shared_primary = None
                new_primary._process = self._process
                new_primary._stop_event = self._stop_event
                new_primary._buffer = self._buffer
                new_primary._sequence = self._sequence
                new_primary._last_frame_at = self._last_frame_at
                new_primary._actual_fps = self._actual_fps
                new_primary._errors = self._errors
                new_primary._buffer_width = self._buffer_width or self.width
                new_primary._buffer_height = self._buffer_height or self.height
                new_primary._subscribers = set(other_subscribers)
                for sub in other_subscribers[1:]:
                    sub._shared_primary = new_primary
                if self._camera_key is not None:
                    _ACTIVE_CAMERAS[self._camera_key] = new_primary
                self._subscribers = set()
                self._camera_key = None
                self._process = None; self._stop_event = None; self._errors = None; self._buffer = None; self._sequence = None; self._last_frame_at = None; self._actual_fps = None
                self._cached_frame = None; self._cached_sequence = -1; self.captured_at = 0.
                return
            if self._camera_key is not None and _ACTIVE_CAMERAS.get(self._camera_key) is self:
                _ACTIVE_CAMERAS.pop(self._camera_key, None)
            self._subscribers = set()
            self._camera_key = None
        if self._stop_event is not None:
            self._stop_event.set()
        process,errors=self._process,self._errors
        self._process=None; self._errors=None; self._buffer=None; self._sequence=None; self._last_frame_at=None; self._actual_fps=None
        self._cached_frame=None; self._cached_sequence=-1; self.captured_at=0.
        if process is None and errors is None: return
        def cleanup():
            forced=False
            try:
                if process is not None:
                    process.join(timeout=1.)
                    if process.is_alive(): forced=True; process.terminate(); process.join(timeout=1.)
            finally:
                if errors is not None:
                    if forced: errors.cancel_join_thread()
                    errors.close()
                    if not forced: errors.join_thread()
        thread=threading.Thread(target=cleanup,name='capture-cleanup',daemon=True)
        self._cleanup_threads=[t for t in self._cleanup_threads if t.is_alive()]+[thread]; thread.start()

    def wait_closed(self,timeout=2.2):
        deadline=time.monotonic()+timeout
        for thread in self._cleanup_threads: thread.join(timeout=max(0,deadline-time.monotonic()))

    @property
    def actual_fps(self) -> float:
        return float(self._actual_fps.value) if self._actual_fps is not None else 0.0

    def latest(self) -> tuple[int, np.ndarray | None]:
        if self._errors is not None:
            try:
                while True:
                    self.error = self._errors.get_nowait()
            except queue.Empty:
                pass
        elif self._shared_primary is not None:
            self.error = self._shared_primary.error

        if self._sequence is None or self._buffer is None:
            return -1, None
        lock=self._sequence.get_lock()
        # A terminated driver process may have died while holding the lock.
        # Never freeze the GUI indefinitely; the watchdog will restart the capture.
        if not lock.acquire(timeout=.02):
            self.error='Буфер камеры недоступен'; return self._cached_sequence,None
        try:
            sequence = int(self._sequence.value)
            if sequence and sequence != self._cached_sequence:
                buf_w = self._buffer_width or self.width
                buf_h = self._buffer_height or self.height
                raw_frame = np.frombuffer(self._buffer, dtype=np.uint8).reshape(buf_h, buf_w, 3).copy()
                if (buf_w, buf_h) != (self.width, self.height) and self.width > 0 and self.height > 0:
                    raw_frame = cv2.resize(raw_frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                self._cached_frame = raw_frame
                self.captured_at=float(self._last_frame_at.value); self._cached_sequence=sequence; self.error=None
        finally: lock.release()
        proc = self._process or (self._shared_primary._process if self._shared_primary is not None else None)
        if (
            proc is not None
            and not proc.is_alive()
            and proc.exitcode not in (None, 0)
        ):
            self.error = f"Процесс камеры завершился с кодом {proc.exitcode}"
        return sequence, self._cached_frame

    def frame_age(self) -> float:
        if self._last_frame_at is None:
            return time.monotonic() - self.started_at
        return time.monotonic() - float(self._last_frame_at.value)


class InferenceSignals(QObject):
    completed = Signal(int, object, float, str)
    failed = Signal(int, object)


class InferenceScheduler:
    """Runs two independent models sequentially and drops obsolete frames."""

    def __init__(self) -> None:
        self.signals = InferenceSignals()
        self._condition = threading.Condition()
        self._pending = {}
        self._models = {}
        self._generations = {}
        self._weight_hashes = {}
        self._auto_cpu_channels: set[int] = set()
        self._last_channel = 1
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="yolo-scheduler")
        self._thread.start()

    def submit(
        self,
        channel: int,
        sequence: int,
        frame: np.ndarray,
        options: InferenceOptions,
        generation: int = 0,
        captured_at: float | None = None,
    ) -> None:
        with self._condition:
            if generation!=self._generations.get(channel,generation): return
            self._generations[channel]=generation
            self._pending[channel] = InferenceJob(channel,sequence,generation,captured_at if captured_at is not None else time.monotonic(),frame.copy(),options)
            self._condition.notify()

    def has_pending(self, channel: int) -> bool:
        with self._condition:
            return channel in self._pending

    def invalidate(self,channel,generation):
        with self._condition:
            self._generations[channel]=generation; self._pending.pop(channel,None); self._auto_cpu_channels.discard(channel)

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=3.0)

    @staticmethod
    def _resolve_device(requested: str, force_cpu: bool = False) -> tuple[str, bool]:
        if force_cpu:
            return "cpu", False
        if requested == "CPU":
            return "cpu", False
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False
        if requested == "GPU" and not cuda_available:
            return "cpu", False
        return ("0", True) if cuda_available else ("cpu", False)

    def _get_model(self, channel: int, model_path: str, device: str):
        path=Path(model_path); path=path if path.is_absolute() else ROOT/path
        if not path.is_file(): raise ValueError('Нет локальных весов: '+str(path)+'. Автоскачивание при запуске отключено.')
        stat=path.stat(); key=(str(path.resolve()),device,stat.st_size,stat.st_mtime_ns)
        hash_key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
        if key in self._models:
            self._last_model_sha=self._weight_hashes[hash_key]; return self._models[key]
        from ultralytics import YOLO
        from final_core import sha256
        model = YOLO(str(path)); sha=sha256(path); after=path.stat()
        if (after.st_size,after.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns): raise ValueError('Веса изменились при загрузке. Выберите завершённый checkpoint.')
        self._models[key]=model; self._weight_hashes[hash_key]=sha; self._last_model_sha=sha
        return model

    def _predict(self,job,device,use_half=False):
        from final_execution import gpu_guard
        with gpu_guard(device):
            loaded_at=time.perf_counter()
            model=self._get_model(job.channel,job.options.model,device)
            self._last_load_ms=(time.perf_counter()-loaded_at)*1000
            if self._last_load_ms > 50.0:
                with self._condition:
                    newer = self._pending.get(job.channel)
                    if newer is not None and newer.generation == job.generation and newer.options == job.options:
                        self._pending.pop(job.channel, None)
                        job.sequence = newer.sequence
                        job.captured_at = newer.captured_at
                        job.frame = newer.frame
            started=time.perf_counter()
            result=model.predict(source=job.frame,imgsz=job.options.image_size,conf=job.options.confidence,
                device=device,quantize=16 if use_half else None,verbose=False)[0]
            self._last_predict_ms=(time.perf_counter()-started)*1000; return result

    @staticmethod
    def _make_packet(result, frame_shape: tuple[int, ...]) -> DetectionPacket:
        height, width = frame_shape[:2]
        if result.boxes is not None:
            boxes = result.boxes.data.detach().cpu().numpy()
        else:
            boxes = np.empty((0, 6), dtype=np.float32)

        mask_overlay = None
        masks=None
        if result.masks is not None:
            from ultralytics.utils.ops import scale_masks
            data=result.masks.data
            if tuple(data.shape[-2:])!=(height,width): data=scale_masks(data[None].float(),(height,width))[0]
            masks=[m.detach().cpu().numpy()>.5 for m in data]
            mask_overlay = np.zeros((height, width, 3), dtype=np.uint8)
            for index, mask in enumerate(masks):
                active = mask
                class_id = int(boxes[index, 5]) if index < len(boxes) else index
                color = (
                    (53 * (class_id + 1)) % 200 + 40,
                    (97 * (class_id + 2)) % 200 + 40,
                    (151 * (class_id + 3)) % 200 + 40,
                )
                mask_overlay[active] = color

        keypoints = None
        if result.keypoints is not None:
            keypoints = result.keypoints.data.detach().cpu().numpy()

        return DetectionPacket(
            boxes=boxes,
            names=dict(result.names),
            mask_overlay=mask_overlay,
            keypoints=keypoints,
            masks=masks,
        )

    def _next(self) -> tuple[int, int, np.ndarray, InferenceOptions] | None:
        with self._condition:
            while not self._pending and not self._stop:
                self._condition.wait(timeout=0.5)
            if self._stop:
                return None
            channels = sorted(self._pending)
            channel = next((value for value in channels if value > self._last_channel), channels[0])
            self._last_channel = channel
            return self._pending.pop(channel)

    def _stamp(self,packet,job):
        geometry=None
        if job.options.path_enabled:
            from path_segmenter import PathSegmenter
            from path_geometry import unknown_geometry
            try:
                if not hasattr(self,'_path_segmenter'): self._path_segmenter=PathSegmenter()
                weights=Path(job.options.geometry_model); weights=weights if weights.is_absolute() else ROOT/weights
                geometry=self._path_segmenter.infer(job.frame,weights,job.sequence,job.generation,job.captured_at,job.options.geometry_source_id,
                    job.options.geometry_confidence,job.options.geometry_image_size)
            except Exception as exc:
                geometry=unknown_geometry(job.frame.shape,job.sequence,job.generation,job.captured_at,job.options.geometry_source_id,'Сегментация пути недоступна: '+str(exc))
        return replace(packet,sequence=job.sequence,generation=job.generation,captured_at=job.captured_at,completed_at=time.monotonic(),
            options=job.options,source_frame=job.frame,model_sha256=self._last_model_sha,model_load_ms=self._last_load_ms,geometry=geometry)

    def _run(self) -> None:
        while True:
            item = self._next()
            if item is None:
                return
            job=item; channel,frame,options=job.channel,job.frame,job.options
            with self._condition:
                if self._generations.get(channel)!=job.generation: continue
            device='cpu'
            try:
                device, use_half = self._resolve_device(
                    options.device,
                    options.device == "Auto" and channel in self._auto_cpu_channels,
                )
                started = time.perf_counter()
                result=self._predict(job,device,use_half)
                if self._stop: return
                elapsed_ms = self._last_predict_ms
                packet = self._stamp(self._make_packet(result, frame.shape),job)
                if options.device == "GPU" and device == "cpu":
                    actual_device = "CPU · NVIDIA CUDA недоступна"
                else:
                    actual_device = "GPU" if device != "cpu" else "CPU"
                self.signals.completed.emit(channel, packet, elapsed_ms, actual_device)
            except Exception as exc:
                # Auto must remain usable on machines whose GPU stack is incomplete.
                if options.device == "Auto" and device != "cpu":
                    try:
                        self._auto_cpu_channels.add(channel)
                        started = time.perf_counter()
                        result=self._predict(job,'cpu')
                        if self._stop: return
                        elapsed_ms = self._last_predict_ms
                        packet = self._stamp(self._make_packet(result, frame.shape),job)
                        self.signals.completed.emit(channel, packet, elapsed_ms, "CPU fallback")
                        continue
                    except Exception as fallback_exc:
                        exc = fallback_exc
                self.signals.failed.emit(channel, {'generation':job.generation,'message':str(exc)})


def frame_to_pixmap(frame: np.ndarray, size) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image).scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class CameraPanel(QGroupBox):
    @property
    def policy(self):
        return self._policy

    @policy.setter
    def policy(self, value):
        # Older saved/demo profiles may predate path tracking and new defaults.
        # Validate before assignment, so a bad import leaves the active policy intact.
        from final_core import validate_policy
        self._policy = validate_policy(value)

    def __init__(
        self,
        channel: int,
        defaults: dict,
        scheduler: InferenceScheduler,
        camera_devices: list[CameraDevice],
    ) -> None:
        super().__init__(f"Камера {channel + 1}")
        self.channel = channel
        self.defaults = defaults
        self.scheduler = scheduler
        self.capture = CameraCapture()
        self.running = False
        self.last_submitted_sequence = -1
        self.last_submit_time = 0.0
        self.detections: DetectionPacket | None = None
        self.last_error = ""
        self.capture_restart_count = 0
        self.last_capture_restart = 0.0
        self.capture_width = defaults["capture_width"]
        self.capture_height = defaults["capture_height"]
        self.capture_fps = defaults["capture_fps"]
        from final_core import validate_policy,EventAssociator
        self.policy=validate_policy(defaults.get('observation_policy',{})); self.associator=EventAssociator(); self.generation=0
        self.last_result_sequence=-1; self.rejected_results=0; self.analysis_times=deque(maxlen=50); self.frames_ring=deque(maxlen=24)
        self.last_ring_sequence=-1; self.health=None; self.event_store=None; self.evidence_frame=None; self.source_kind='camera'
        self._active_model_sha=''
        self._health_summary=''
        self.source_ref={'kind':'camera','index':defaults['camera_index']}
        self.capture_health=None
        from path_geometry import RouteState
        self.route_state=RouteState(); self.path_geometry=None; self.path_raw=None; self.path_pipeline=None

        self.video = QLabel("Камера остановлена")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(480, 300)
        self.video.setStyleSheet("background:#111; color:#aaa; border:1px solid #333")

        self.camera_selector = QComboBox()
        self.set_camera_devices(camera_devices, defaults["camera_index"])

        self.model_path = QLineEdit(defaults["model"])
        browse = QPushButton("Модель…")
        browse.clicked.connect(self._browse_model)

        self.model_preset = QComboBox()
        self.model_preset.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.model_preset.setMinimumContentsLength(18)
        self.model_preset.addItem("Детекция", "yolo26n.pt")
        self.model_preset.addItem("Сегментация — общая модель", "yolo26s-seg.pt")
        self.model_preset.addItem("Оценка позы", "yolo26n-pose.pt")
        trained_person_model = ROOT / "models" / "person_security_yolo26n.pt"
        if trained_person_model.exists():
            self.model_preset.addItem("Человек — исходная модель (откат)", str(trained_person_model))
        pantograph_presets = (
            ("Пантограф — локатор V2", "pantograph_locator_yolo26s_v2.pt"),
            ("Пантограф — точка контакта V2", "pantograph_contact_roi_yolo26n_v2.pt"),
            ("Пантограф — искрение V2", "pantograph_arc_detector_yolo26n_v2.pt"),
        )
        for title, filename in pantograph_presets:
            model_path = ROOT / "models" / filename
            if model_path.exists():
                self.model_preset.addItem(title, str(model_path))
        rail_presets = (
            ("Рельсы — дефекты пути V1", "rail_defects_yolo26s_v1.pt"),
            ("Крепления — диагностика RFDD V1", "yolo26s_rfdd_fasteners_best.pt"),
            ("Путь — сегментация компонентов V1", "yolo26s_straight_track_seg_best.pt"),
        )
        for title, filename in rail_presets:
            model_path = ROOT / "models" / filename
            if model_path.exists():
                self.model_preset.addItem(title, str(model_path))
        self.model_preset.addItem("Пользовательская", "")
        from trained_models import available_models
        for spec in available_models(ROOT):
            index=self.model_preset.count()-1
            self.model_preset.insertItem(index,spec['title'],spec['path'])
            self.model_preset.setItemData(index,spec['purpose'],Qt.ToolTipRole)
        default_name = Path(defaults["model"]).name.casefold()
        preset_row = next(
            (
                row
                for row in range(self.model_preset.count())
                if Path(str(self.model_preset.itemData(row))).name.casefold() == default_name
            ),
            self.model_preset.count() - 1,
        )
        self.model_preset.setCurrentIndex(preset_row)
        self.model_preset.currentIndexChanged.connect(self._model_preset_changed)

        self.device = QComboBox()
        self.device.addItem("Auto", "Auto")
        self.device.addItem("CPU", "CPU")
        self.device.addItem("GPU (NVIDIA CUDA)", "GPU")
        device_row = self.device.findData(defaults["device"])
        self.device.setCurrentIndex(max(0, device_row))
        self.device.currentIndexChanged.connect(self._inference_settings_changed)

        self.inference_fps = QSpinBox()
        self.inference_fps.setRange(1, 30)
        self.inference_fps.setValue(defaults["inference_fps"])

        self.image_size = QComboBox()
        self.image_size.addItems(["320", "416", "512", "640", "960", "1024", "1280", "1536"])
        self.image_size.setCurrentText(str(defaults["image_size"]))

        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.05, 0.95)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(defaults["confidence"])

        self.toggle = QPushButton("Запустить")
        self.toggle.clicked.connect(self.toggle_camera)
        self.status = QLabel("Готово")
        self.status.setWordWrap(True)
        self.status.setFixedHeight(22)
        self.status.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.health_label=QLabel('Участок не оценён; зона пути не задана.'); self.health_label.setWordWrap(True)
        self.health_label.setFixedHeight(40)
        self.health_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        source=QHBoxLayout(); self.source=QComboBox(); self.source.addItems(['USB-камера','Видеофайл','RTSP / поток']); source.addWidget(self.source)
        self.source_path=QLineEdit(); self.source_path.setPlaceholderText('Путь к видео или адрес потока'); source.addWidget(self.source_path)
        choose_video=QPushButton('Видео…'); choose_video.clicked.connect(self._browse_video); source.addWidget(choose_video)
        observation=QPushButton('Контроль и зона…'); observation.clicked.connect(self._observation_dialog); source.addWidget(observation)

        settings = QGridLayout()
        settings.addWidget(QLabel("Камера"), 0, 0)
        settings.addWidget(self.camera_selector, 0, 1, 1, 3)
        settings.addWidget(QLabel("Анализ FPS"), 0, 4)
        settings.addWidget(self.inference_fps, 0, 5)
        settings.addWidget(QLabel("Устройство"), 1, 0)
        settings.addWidget(self.device, 1, 1)
        settings.addWidget(QLabel("Размер"), 1, 2)
        settings.addWidget(self.image_size, 1, 3)
        settings.addWidget(QLabel("Confidence"), 1, 4)
        settings.addWidget(self.confidence, 1, 5)
        settings.addWidget(QLabel("Задача"), 2, 0)
        settings.addWidget(self.model_preset, 2, 1, 1, 2)
        settings.addWidget(self.model_path, 2, 3, 1, 2)
        settings.addWidget(browse, 2, 5)

        footer = QHBoxLayout()
        footer.addWidget(self.toggle)
        footer.addWidget(self.status, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.video, 1)
        layout.addLayout(source)
        layout.addLayout(settings)
        layout.addLayout(footer)
        layout.addWidget(self.health_label)
        self.model_path.textChanged.connect(self._inference_settings_changed)
        self.image_size.currentIndexChanged.connect(self._inference_settings_changed)
        self.source_path.textChanged.connect(self._source_settings_changed)
        for widget in [self.camera_selector,self.source]: widget.currentIndexChanged.connect(self._source_settings_changed)
        self.confidence.valueChanged.connect(self._inference_settings_changed)
        self.inference_fps.valueChanged.connect(self._inference_settings_changed)

    def _browse_video(self):
        path,_=QFileDialog.getOpenFileName(self,'Записанное видео',str(ROOT),'Видео (*.mp4 *.avi *.mkv *.mov);;Все файлы (*)')
        if path: self.source.setCurrentIndex(1); self.source_path.setText(path)

    def _observation_dialog(self):
        dialog=QDialog(self); dialog.setWindowTitle('Контроль наблюдения и зона пути'); dialog.resize(760,650); layout=QVBoxLayout(dialog)
        note=QLabel('Зона задаётся точками [x,y] от 0 до 1 в изображении этой камеры. Пустая зона означает «не задана». Скорость и длина читаемого участка — ваши исходные данные для расчёта, не измеренные системой. min_blur=0 отключает порог смаза: его нужно подобрать на своём контрольном наборе. Роли классов явно определяют, какие находки создают события.'); note.setWordWrap(True); layout.addWidget(note)
        if self.detections:
            classes=QLabel('Имена классов активной модели: '+', '.join(map(str,self.detections.names.values()))); classes.setWordWrap(True); layout.addWidget(classes)
        edit=QPlainTextEdit(json.dumps(self.policy,ensure_ascii=False,indent=2)); layout.addWidget(edit)
        load=QPushButton('Загрузить сохранённый профиль'); layout.addWidget(load)
        def load_profile():
            path,_=QFileDialog.getOpenFileName(dialog,'Профиль контроля',str(ROOT.parents[1]/'11_Финальный_модуль'),'JSON (*.json)')
            if path:
                try: edit.setPlainText(json.dumps(json.loads(Path(path).read_text(encoding='utf-8')),ensure_ascii=False,indent=2))
                except Exception as exc: QMessageBox.warning(dialog,'Профиль',str(exc))
        load.clicked.connect(load_profile)
        button=QPushButton('Применить и сохранить профиль'); layout.addWidget(button)
        def apply():
            try:
                from final_core import validate_policy,save_json
                self.policy=validate_policy(json.loads(edit.toPlainText())); self._inference_settings_changed()
                if self.event_store: save_json(self.event_store.root/f'channel_{self.channel}_policy.json',self.policy)
                dialog.accept()
            except Exception as exc: QMessageBox.warning(dialog,'Параметры',str(exc))
        button.clicked.connect(apply); dialog.exec()

    def _invalidate(self):
        self.generation+=1; self.scheduler.invalidate(self.channel,self.generation); self.detections=None; self.evidence_frame=None
        self.last_result_sequence=-1; self.last_submitted_sequence=-1; self.analysis_times.clear(); self.frames_ring.clear(); self.last_ring_sequence=-1; self.associator.reset(self.channel)
        self.capture_health=None
        if hasattr(self,'route_state'): self.route_state.reset(); self.path_geometry=None; self.path_raw=None
        if getattr(self,'path_pipeline',None): self.path_pipeline.reset(self.channel)

    def analysis_rate(self,stamp):
        recent=[t for t in self.analysis_times if 0<=stamp-t<=3.]
        return (len(recent)-1)/(recent[-1]-recent[0]) if len(recent)>1 and recent[-1]>recent[0] else 0.

    def _source_settings_changed(self,*_):
        if self.running: self.stop(); self.status.setText('Источник изменён. Нажмите «Запустить».')
        else: self._invalidate()
        self.policy.update(zone=[],min_blur=0.,speed_kmh=0.,visible_length_m=0.,geometry_calibration_file='')
        self.health_label.setText('Источник изменён: заново задайте зону и параметры контроля для этого ракурса.')

    def set_camera_devices(
        self,
        devices: list[CameraDevice],
        preferred_index: int | None = None,
    ) -> None:
        if preferred_index is None:
            preferred_index = self.selected_camera_index()
        self.camera_selector.clear()
        if not devices:
            self.camera_selector.addItem("Камеры не найдены", None)
            self.camera_selector.setEnabled(False)
            return
        self.camera_selector.setEnabled(True)
        selected_row = 0
        for row, device in enumerate(devices):
            kind = " · виртуальная" if device.is_virtual else ""
            self.camera_selector.addItem(
                f"{device.name}{kind}  [ID {device.index}]", device.index
            )
            if device.index == preferred_index:
                selected_row = row
        self.camera_selector.setCurrentIndex(selected_row)

    def selected_camera_index(self) -> int | None:
        value = self.camera_selector.currentData()
        return int(value) if value is not None else None

    def _browse_model(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Выбор модели",
            str(ROOT),
            "YOLO models (*.pt *.onnx *.engine);;Все файлы (*.*)",
        )
        if filename:
            self.model_preset.setCurrentIndex(self.model_preset.count() - 1)
            self.model_path.setText(filename)

    def _model_preset_changed(self, index: int) -> None:
        model_name = str(self.model_preset.itemData(index) or "")
        if model_name:
            self.model_path.setText(model_name)
            recommended_sizes = {
                "yolo26s-seg.pt": "640",
                "pantograph_locator_yolo26s_v2.pt": "1024",
                "pantograph_contact_roi_yolo26n_v2.pt": "1280",
                "pantograph_arc_detector_yolo26n_v2.pt": "960",
                "rail_defects_yolo26s_v1.pt": "1024",
                "yolo26s_rfdd_fasteners_best.pt": "1280",
                "yolo26s_straight_track_seg_best.pt": "1024",
            }
            recommended_size = recommended_sizes.get(Path(model_name).name.casefold())
            from trained_models import available_models
            spec=next((s for s in available_models(ROOT) if Path(model_name).name==s['filename']),None)
            if spec: recommended_size=str(spec['imgsz'])
            if recommended_size:
                self.image_size.setCurrentText(recommended_size)
        self.detections = None
        if self.running:
            self.status.setText("Загрузка выбранной модели…")

    def toggle_camera(self) -> None:
        if self.running:
            self.stop()
            return
        self.detections = None
        self._invalidate()
        self.last_submitted_sequence = -1
        self.capture_restart_count = 0
        self.capture_width = self.defaults["capture_width"]
        self.capture_height = self.defaults["capture_height"]
        self.capture_fps = self.defaults["capture_fps"]
        camera_index = self.selected_camera_index()
        self.capture.stop()
        if self.source.currentIndex()==1:
            from final_capture import FileCapture
            self.capture=FileCapture(loop=True); camera_index=self.source_path.text(); self.source_kind='file'
        else:
            self.capture=CameraCapture(); self.source_kind='stream' if self.source.currentIndex()==2 else 'camera'
            if self.source_kind=='stream': camera_index=self.source_path.text().strip()
        if camera_index is None:
            self.status.setText("Подключённые камеры не найдены")
            return
        if camera_index=='': self.status.setText('Выберите источник видео.'); return
        if self.source_kind=='stream':
            from urllib.parse import urlsplit
            try:
                parsed=urlsplit(str(camera_index))
                if parsed.scheme not in ('rtsp','rtsps','http','https') or not parsed.hostname: raise ValueError('Нужен адрес rtsp/rtsps/http/https с именем сервера.')
                _port=parsed.port
            except ValueError as exc: self.status.setText(str(exc)); return
        try: self.capture.start(camera_index,self.capture_width,self.capture_height,self.capture_fps)
        except Exception as exc: self.status.setText(str(exc)); return
        if self.source_kind=='file':
            path=Path(camera_index).resolve(); stat=path.stat(); self.source_ref={'kind':'file','path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':None,'hash_note':'Live playback does not hash the whole video; CPU rehearsal records its SHA256.'}
        elif self.source_kind=='stream':
            from urllib.parse import urlsplit
            parsed=urlsplit(str(camera_index)); self.source_ref={'kind':'stream','scheme':parsed.scheme,'host':parsed.hostname,'path':parsed.path,'credentials_and_query_recorded':False}
        else: self.source_ref={'kind':'camera','index':camera_index,'name':self.camera_selector.currentText()}
        self.running = True
        self.toggle.setText("Остановить")
        self.status.setText("Подключение…")

    def stop(self) -> None:
        self.running = False
        self._invalidate()
        self.capture.stop()
        self.toggle.setText("Запустить")
        self.video.setText("Камера остановлена")
        self.video.setPixmap(QPixmap())
        self.status.setText("Остановлена")
        self.health_label.setText('Участок не оценён: источник остановлен.')

    @staticmethod
    def _draw_detections(frame: np.ndarray, packet: DetectionPacket | None) -> np.ndarray:
        if packet is None:
            return frame
        output = frame.copy()
        height, width = output.shape[:2]

        if packet.mask_overlay is not None:
            overlay = packet.mask_overlay
            if overlay.shape[:2] != (height, width):
                overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_NEAREST)
            active = np.any(overlay != 0, axis=2)
            if np.any(active):
                blended = (
                    output[active].astype(np.float32) * 0.55
                    + overlay[active].astype(np.float32) * 0.45
                )
                output[active] = blended.astype(np.uint8)

        skeleton = (
            (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
            (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 1), (0, 2), (1, 3), (2, 4),
        )
        if packet.keypoints is not None:
            for person in packet.keypoints:
                def visible(point_index: int) -> bool:
                    return (
                        point_index < len(person)
                        and len(person[point_index]) >= 2
                        and (len(person[point_index]) < 3 or person[point_index][2] >= 0.25)
                    )

                for first, second in skeleton:
                    if visible(first) and visible(second):
                        p1 = tuple(map(int, person[first][:2]))
                        p2 = tuple(map(int, person[second][:2]))
                        cv2.line(output, p1, p2, (40, 220, 255), 2, cv2.LINE_AA)
                for point_index in range(len(person)):
                    if visible(point_index):
                        point = tuple(map(int, person[point_index][:2]))
                        cv2.circle(output, point, 4, (30, 30, 255), -1, cv2.LINE_AA)

        for row in packet.boxes:
            x1, y1, x2, y2, confidence, class_id = row[:6]
            left = max(0, min(width - 1, int(x1)))
            top = max(0, min(height - 1, int(y1)))
            right = max(0, min(width - 1, int(x2)))
            bottom = max(0, min(height - 1, int(y2)))
            color = (255, 90, 20)
            label = f"{packet.names.get(int(class_id), int(class_id))} {confidence:.2f}"
            cv2.rectangle(output, (left, top), (right, bottom), color, 2)
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            label_top = max(0, top - text_height - 8)
            cv2.rectangle(
                output,
                (left, label_top),
                (min(width - 1, left + text_width + 6), top),
                color,
                -1,
            )
            cv2.putText(
                output,
                label,
                (left + 3, max(text_height, top - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return output

    def options(self) -> InferenceOptions:
        from path_geometry import source_key
        return InferenceOptions(
            model=self.model_path.text().strip() or "yolo26n.pt",
            device=str(self.device.currentData()),
            image_size=int(self.image_size.currentText()),
            confidence=self.confidence.value(),
            path_enabled=self.policy['path_enabled'],geometry_model=self.policy['geometry_model'],geometry_confidence=self.policy['geometry_confidence'],
            geometry_image_size=self.policy['geometry_image_size'],geometry_source_id=source_key(self.source_ref),
        )

    def refresh(self) -> None:
        if not self.running:
            return
        if self.capture.error and self.capture.error != self.last_error:
            self.last_error = self.capture.error
            self.status.setText(self.capture.error)

        sequence, frame = self.capture.latest()
        now = time.monotonic()
        watchdog_limit = 18.0 if sequence <= 0 else 4.0
        if self.capture.frame_age() > watchdog_limit and now - self.last_capture_restart > 12.0 and self.source_kind!='file':
            camera_index = self.source_path.text().strip() if self.source_kind=='stream' else self.selected_camera_index()
            if camera_index is not None:
                self.capture_restart_count += 1
                self.last_capture_restart = now
                if self.capture_restart_count == 1:
                    self.capture_fps = min(self.capture_fps, 15)
                elif self.capture_restart_count >= 2:
                    self.capture_width, self.capture_height = 320, 240
                    self.capture_fps = min(self.capture_fps, 15)
                self.status.setText(
                    "Нет новых кадров — перезапуск процесса "
                    f"{self.capture_width}×{self.capture_height} @{self.capture_fps} FPS…"
                )
                self._invalidate()
                self.capture.start(
                    camera_index,
                    self.capture_width,
                    self.capture_height,
                    self.capture_fps,
                )
                return
        if frame is None:
            self.detections=None; self.video.setPixmap(QPixmap()); self.video.setText('Нет актуального кадра')
            self.health_label.setText('Участок не оценён: нет согласованного кадра.')
            return
        if sequence!=self.last_ring_sequence:
            self.frames_ring.append((self.capture.captured_at,frame)); self.last_ring_sequence=sequence
            from final_core import frame_health
            self.capture_health=frame_health(frame,self.capture.captured_at,now,self.analysis_rate(now),self.policy)
        # Draw boxes only on the exact evidence frame. Never transfer them to a newer camera image.
        packet=self.detections
        fresh=packet is not None and not self.capture.error and not getattr(self.capture,'finished',False) and (self.capture_health is None or self.capture_health['assessed']) and (now-packet.captured_at)*1000<=self.policy['max_age_ms']
        display=self._draw_detections(packet.source_frame,packet) if fresh else frame
        if self.policy['zone']:
            display=display.copy(); h,w=display.shape[:2]; points=(np.asarray(self.policy['zone'])*[w-1,h-1]).astype(np.int32)
            cv2.polylines(display,[points],True,(0,220,255),2)
        geometry_fresh=fresh and (now-packet.captured_at)*1000<=self.policy['geometry_max_age_ms'] if packet is not None else False
        if geometry_fresh and self.path_geometry is not None and self.path_geometry.get('corridor_mask') is not None and self.path_geometry['sequence']==packet.sequence:
            display=display.copy(); mask=self.path_geometry['corridor_mask']; display[mask]=(display[mask]*.72+np.array([45,210,85])*.28).astype(np.uint8)
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(display,contours,-1,(50,220,110),2)
        self.video.setPixmap(frame_to_pixmap(display, self.video.size()))
        if not fresh: self.health_label.setText('Участок не оценён: видео завершено.' if getattr(self.capture,'finished',False) else 'Участок не оценён: нет свежего результата. Старые рамки скрыты.')
        else: self.health_label.setText(self._health_summary+f" · текущий возраст {(now-packet.captured_at)*1000:.0f} мс")
        if self.policy['path_enabled'] and not geometry_fresh:
            self.route_state.reset(required=bool(self.route_state.selection) or self.route_state.require_confirmation)
            if self.path_geometry is not None:
                self.path_geometry={**self.path_geometry,'status':'unknown','corridor_mask':None,'candidate_id':None,'quality_score':None,'manual_confirmation':None,'metric_status':'geometry_unknown','reasons':['нет свежих масок для текущего наблюдения']}
            self.health_label.setText('Геометрия неизвестна: нет свежих масок для текущего наблюдения. Метры не выдаются.')
        if self.capture_health and not self.capture_health['assessed']: self.health_label.setText('Участок не оценён: текущий кадр источника — '+', '.join(self.capture_health['reasons']))

        interval = 1.0 / self.inference_fps.value()
        has_pending = self.scheduler.has_pending(self.channel)
        if sequence != self.last_submitted_sequence and (has_pending or now - self.last_submit_time >= interval):
            self.scheduler.submit(self.channel, sequence, frame, self.options(),self.generation,self.capture.captured_at)
            self.last_submitted_sequence = sequence
            if not has_pending:
                self.last_submit_time = now

    def set_result(self, packet: DetectionPacket, elapsed_ms: float, device: str) -> None:
        now=time.monotonic()
        if not self.running or packet.generation!=self.generation or packet.options!=self.options() or packet.sequence<=self.last_result_sequence or packet.source_frame is None or (now-packet.captured_at)*1000>self.policy['max_age_ms']:
            self.rejected_results+=1
            if self.event_store: self.event_store.telemetry({'kind':'rejected_result','channel':self.channel,'generation':packet.generation,'sequence':packet.sequence,'end_to_end_ms':max(0.,(now-packet.captured_at)*1000),'reason':'stale, out-of-order or generation/settings mismatch'})
            return
        from final_core import frame_health
        if self._active_model_sha and self._active_model_sha!=packet.model_sha256:
            self.associator.reset(self.channel)
            if self.path_pipeline is not None: self.path_pipeline.reset(self.channel)
        self._active_model_sha=packet.model_sha256
        self.last_result_sequence=packet.sequence; self.analysis_times.append(packet.captured_at)
        rate=self.analysis_rate(packet.captured_at)
        self.health=frame_health(packet.source_frame,packet.captured_at,now,rate,self.policy)
        if self.capture.error: self.health['assessed']=False; self.health['reasons'].append('источник недоступен')
        zones=self.policy['zone']; quality='Кадр пригоден по заданным проверкам' if self.health['assessed'] else 'Участок не оценён: '+', '.join(self.health['reasons'])
        self._health_summary=quality+f" · анализ {rate:.1f}/с · резкость {self.health['blur_variance']:.0f} · тёмных {self.health['dark_fraction']:.0%} / ярких {self.health['bright_fraction']:.0%} · "+('зона задана' if zones else 'зона не задана')+(' · покрытие не калибровано' if self.health['coverage'] is None else ' · расчёт покрытия: '+self.health['coverage_status'])
        self.health_label.setText(self._health_summary+f" · возраст {self.health['age_ms']:.0f} мс")
        self.path_raw=packet.geometry; calibration=None
        if self.policy['path_enabled']:
            from path_geometry import unknown_geometry,source_key,frame_digest,GroundCalibration
            raw=packet.geometry or unknown_geometry(packet.source_frame.shape,packet.sequence,packet.generation,packet.captured_at,source_key(self.source_ref),'нет сегментации для этого кадра')
            if raw.get('frame_sha256') is not None and raw['frame_sha256']!=frame_digest(packet.source_frame): raw=unknown_geometry(packet.source_frame.shape,packet.sequence,packet.generation,packet.captured_at,source_key(self.source_ref),'маски относятся к другому изображению')
            self.path_raw=raw
            self.path_geometry=self.route_state.evaluate(raw,now,packet.sequence,packet.generation,source_key(self.source_ref),self.policy['geometry_max_age_ms'],self.policy['geometry_switch'],self.health['assessed'])
            if self.policy['geometry_calibration_file']:
                try:
                    calibration=GroundCalibration(json.loads(Path(self.policy['geometry_calibration_file']).read_text(encoding='utf-8')))
                    self.path_geometry['metric_status']='calibration_check_failed' if not calibration.valid else 'calibration_source_or_resolution_mismatch' if calibration.data['source_id']!=source_key(self.source_ref) or calibration.data.get('shape')!=list(packet.source_frame.shape[:2]) else 'calibration_available_ground_region_only'
                except Exception as exc: self.path_geometry['metric_status']='calibration_invalid'; self.path_geometry['reasons'].append(str(exc))
            self._health_summary+=' · геометрия: '+self.path_geometry['status']+' · метры: '+self.path_geometry['metric_status']
            self.health_label.setText(self._health_summary)
        else: self.path_geometry=None
        if self.policy['path_enabled'] or self.policy['tracker']!='iou':
            from path_tracking import PathEventPipeline
            if self.path_pipeline is None or self.path_pipeline.backend.mode!=self.policy['tracker'] or self.path_pipeline.backend.gap_s!=self.policy['group_gap_s'] or self.path_pipeline.backend.min_iou!=self.policy['track_iou']: self.path_pipeline=PathEventPipeline(self.policy['tracker'],self.policy['group_gap_s'],self.policy['track_iou'])
            updates=self.path_pipeline.update(self.channel,self.generation,packet.sequence,packet.captured_at,packet.boxes,packet.names,packet.source_frame.shape,self.policy,self.health,packet.masks,self.path_geometry,calibration)
        else: updates=self.associator.update(self.channel,self.generation,packet.sequence,packet.captured_at,packet.boxes,packet.names,packet.source_frame.shape,self.policy,self.health,packet.masks)
        if self.event_store:
            # File source advances while inference runs: derive media time from the
            # analysed packet sequence, never from the latest capture frame.
            source_time_s=(packet.sequence-1)/self.capture.actual_fps if self.source_kind=='file' else None
            self.event_store.telemetry({'kind':'accepted_result','channel':self.channel,'sequence':packet.sequence,'generation':self.generation,
                'captured_at':packet.captured_at,'end_to_end_ms':(now-packet.captured_at)*1000,'inference_ms':elapsed_ms,'model_load_ms':packet.model_load_ms,
                'health':self.health,'source_kind':self.source_kind,'source_time_s':source_time_s})
            if self.path_geometry is not None:
                from path_geometry import serializable_geometry
                self.event_store.telemetry({'kind':'path_context','channel':self.channel,'sequence':packet.sequence,'generation':packet.generation,
                    'captured_at':packet.captured_at,'source_time_s':source_time_s,'source_ref':self.source_ref,
                    'frame_sha256':frame_digest(packet.source_frame),'shape':list(packet.source_frame.shape[:2]),'boxes':packet.boxes.tolist(),'names':packet.names,
                    'object_model_sha256':packet.model_sha256,'image_assessed':self.health['assessed'],
                    'geometry':serializable_geometry(self.path_geometry),'model_sha256':packet.geometry.get('model_sha256') if packet.geometry else None,
                    'segmentation_ms':packet.geometry.get('segmentation_ms') if packet.geometry else None,'objects':self.path_pipeline.context})
            for event in updates:
                first=event['hits']==1
                evidence=[(t,f) for t,f in self.frames_ring if t<=packet.captured_at][-7:] if first else []
                if first:
                    evidence=[(t,f) for t,f in evidence if t!=packet.captured_at]; evidence.append((packet.captured_at,packet.source_frame))
                self.event_store.submit(event,evidence,{'model':packet.options.model,'model_sha256':packet.model_sha256,
                    'settings':vars(packet.options),'generation':packet.generation,'sequence':packet.sequence,'captured_at_monotonic':packet.captured_at,
                    'end_to_end_ms':(now-packet.captured_at)*1000,'inference_ms':elapsed_ms,'actual_device':device,'source_kind':self.source_kind,
                    'source_time_s':source_time_s,'source_ref':self.source_ref,'zone':self.policy['zone'],'tracker':self.policy['tracker'],
                    'path_geometry':event.get('geometry'),'context_state':event.get('context_state')})
        self.detections = packet
        self.status.setText(
            f"{device} · inference {elapsed_ms:.0f} ms · источник {self.capture.actual_fps:.1f} FPS · кадр {packet.sequence}"
        )

    def set_failure(self, message) -> None:
        if isinstance(message,dict):
            if not self.running or message['generation']!=self.generation: self.rejected_results+=1; return
            message=message['message']
        # Never leave the last annotated frame on screen when inference has stopped.
        self.detections = None
        short = message.replace("\n", " ")[:180]
        self.status.setText(f"Ошибка модели: {short}")
        self.health_label.setText('Участок не оценён: ошибка модели.')

    def _inference_settings_changed(self, _index: int = -1) -> None:
        # Show the live camera while a model/device is being reinitialized.
        self._invalidate()
        if self.running:
            self.status.setText("Переключение устройства…")


class MainWindow(QMainWindow):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.setWindowTitle("Panoptr — наблюдение, события и подготовка финала")
        self.resize(config["window"]["width"], config["window"]["height"])
        self.scheduler = InferenceScheduler()
        self.camera_devices = discover_camera_devices()
        physical_devices = [device for device in self.camera_devices if not device.is_virtual]
        panel_configs = []
        for index, camera in enumerate(config["cameras"][:2]):
            camera_defaults = dict(camera)
            if index < len(physical_devices):
                camera_defaults["camera_index"] = physical_devices[index].index
            elif physical_devices:
                camera_defaults["camera_index"] = physical_devices[0].index
            panel_configs.append(camera_defaults)
        self.panels = [
            CameraPanel(index, camera, self.scheduler, self.camera_devices)
            for index, camera in enumerate(panel_configs)
        ]
        from final_store import EventStore
        from final_core import save_json
        from datetime import datetime
        session=Path(config.get('event_session') or ROOT.parents[1]/'data/sessions'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.event_store=EventStore(session); save_json(session/'config.json',config)
        for panel in self.panels: panel.event_store=self.event_store; save_json(session/f'channel_{panel.channel}_policy.json',panel.policy)
        self.final_dashboard=None; self._telemetry_at=0.
        self.scheduler.signals.completed.connect(self._completed)
        self.scheduler.signals.failed.connect(self._failed)

        cameras = QHBoxLayout()
        for panel in self.panels:
            cameras.addWidget(panel, 1)

        self.resources = QLabel()
        self.resources.setAlignment(Qt.AlignRight)
        start_all = QPushButton("Запустить обе")
        start_all.clicked.connect(self._start_all)
        stop_all = QPushButton("Остановить обе")
        stop_all.clicked.connect(self._stop_all)
        refresh_cameras = QPushButton("Обновить камеры")
        refresh_cameras.clicked.connect(self._refresh_cameras)
        open_simulator = QPushButton("Симулятор ВСМ")
        open_simulator.clicked.connect(self._open_vsm_simulator)
        open_lab = QPushButton("Лаборатория данных")
        open_lab.clicked.connect(self._open_dataset_lab)
        open_final=QPushButton('События и финал'); open_final.clicked.connect(self._open_final_dashboard)
        open_route_memory=QPushButton('Карта рейсов'); open_route_memory.clicked.connect(self._open_route_memory)
        open_finale=QPushButton('Полигон / финал'); open_finale.clicked.connect(self._open_finale)
        self.split_windows_btn = QPushButton('Камеры в 2 окнах')
        self.split_windows_btn.clicked.connect(self._toggle_split_windows)
        self._second_camera_window = None
        self._cameras_layout = cameras
        toolbar = QHBoxLayout()
        toolbar.addWidget(start_all)
        toolbar.addWidget(stop_all)
        toolbar.addWidget(refresh_cameras)
        toolbar.addWidget(self.split_windows_btn)
        toolbar.addWidget(open_simulator)
        toolbar.addWidget(open_lab)
        toolbar.addWidget(open_final)
        toolbar.addWidget(open_route_memory)
        toolbar.addWidget(open_finale)
        toolbar.addStretch(1)
        toolbar.addWidget(self.resources)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(toolbar)
        layout.addLayout(cameras, 1)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        psutil.cpu_percent(None)

    def _toggle_split_windows(self) -> None:
        if len(self.panels) < 2:
            return
        panel2 = self.panels[1]
        if self._second_camera_window is None:
            win = QMainWindow(None)
            win.setWindowTitle("Panoptr — Камера 2 (отдельное окно)")
            win.resize(max(640, self.width() // 2), max(560, self.height() - 60))
            self._cameras_layout.removeWidget(panel2)
            win.setCentralWidget(panel2)
            panel2.show()
            def _on_close(ev):
                win.takeCentralWidget()
                self._cameras_layout.addWidget(panel2, 1)
                panel2.show()
                self._second_camera_window = None
                self.split_windows_btn.setText("Камеры в 2 окнах")
                ev.accept()
            win.closeEvent = _on_close
            self._second_camera_window = win
            self.split_windows_btn.setText("Вернуть Камеру 2 в общее окно")
            win.show()
            win.raise_()
        else:
            self._second_camera_window.close()

    def _open_finale(self) -> None:
        from finale_ui import FinaleDialog
        if not hasattr(self,'finale_dialog'):self.finale_dialog=FinaleDialog(self)
        self.finale_dialog.timer.start(300)
        self.finale_dialog.showNormal()
        self.finale_dialog.raise_()
        self.finale_dialog.activateWindow()

    def _start_all(self) -> None:
        selected = [panel.selected_camera_index() for panel in self.panels if panel.source.currentIndex()==0]
        if None in selected:
            QMessageBox.warning(self, "Камеры", "Подключённые камеры не найдены.")
            return
        for panel in self.panels:
            if not panel.running:
                panel.toggle_camera()

    def _refresh_cameras(self) -> None:
        if any(panel.running for panel in self.panels):
            QMessageBox.information(
                self,
                "Обновление камер",
                "Сначала остановите обе камеры.",
            )
            return
        devices = discover_camera_devices()
        self.camera_devices = devices
        physical_devices = [device for device in devices if not device.is_virtual]
        for index, panel in enumerate(self.panels):
            preferred = (
                physical_devices[index].index
                if index < len(physical_devices)
                else (physical_devices[0].index if physical_devices else panel.selected_camera_index())
            )
            panel.set_camera_devices(devices, preferred)
        if not devices:
            QMessageBox.warning(
                self,
                "Камеры",
                "Windows не обнаружила доступных видеоустройств.",
            )

    def _stop_all(self) -> None:
        for panel in self.panels:
            if panel.running:
                panel.stop()

    def _open_vsm_simulator(self) -> None:
        simulator = ROOT / "vsm_simulator" / "main.py"
        if not simulator.exists():
            QMessageBox.critical(self, "Симулятор ВСМ", f"Модуль не найден: {simulator}")
            return
        started, _process_id = QProcess.startDetached(sys.executable, [str(simulator)], str(simulator.parent))
        if not started:
            QMessageBox.critical(self, "Симулятор ВСМ", "Не удалось запустить отдельный процесс.")

    def _open_dataset_lab(self) -> None:
        module = ROOT / "dataset_lab" / "main.py"
        started, _process_id = QProcess.startDetached(sys.executable, [str(module)], str(module.parent))
        if not started:
            QMessageBox.critical(self, "Лаборатория данных", "Не удалось запустить отдельный процесс.")

    def _open_final_dashboard(self):
        if self.final_dashboard is None:
            from final_ui import FinalDashboard
            self.final_dashboard=FinalDashboard(self.panels,self.event_store,self)
        self.final_dashboard.show(); self.final_dashboard.raise_(); self.final_dashboard.refresh_events()

    def _open_route_memory(self):
        from route_memory_ui import RouteMemoryWindow
        if not getattr(self,'route_memory_window',None):
            self.route_memory_window=RouteMemoryWindow(session=self.event_store.root,parent=self)
            self.route_memory_window.setWindowFlag(Qt.Window,True)
        self.route_memory_window.show(); self.route_memory_window.raise_()
        self.route_memory_window.timer.start(3000)

    def _completed(self, channel: int, frame: object, elapsed_ms: float, device: str) -> None:
        self.panels[channel].set_result(frame, elapsed_ms, device)

    def _failed(self, channel: int, message: str) -> None:
        self.panels[channel].set_failure(message)

    def _tick(self) -> None:
        for panel in self.panels:
            panel.refresh()
        memory = psutil.virtual_memory()
        self.resources.setText(
            f"CPU {psutil.cpu_percent(None):.0f}% · RAM {memory.percent:.0f}%"
        )
        now=time.monotonic()
        if now-self._telemetry_at>=1.:
            self._telemetry_at=now
            for panel in self.panels:
                self.event_store.telemetry({'kind':'channel_status','channel':panel.channel,'monotonic':now,'running':panel.running,
                    'capture_age_ms':max(0.,panel.capture.frame_age()*1000) if panel.running else None,'source_kind':panel.source_kind,
                    'rejected_results':panel.rejected_results,'last_accepted_sequence':panel.last_result_sequence,
                    'assessed':bool(panel.running and not panel.capture.error and panel.detections and panel.health and panel.health['assessed'] and (panel.capture_health is None or panel.capture_health['assessed']) and not getattr(panel.capture,'finished',False) and (now-panel.detections.captured_at)*1000<=panel.policy['max_age_ms'])})
        if self.event_store.error: self.resources.setText('Запись недоступна: '+self.event_store.error)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.final_dashboard and self.final_dashboard.process.state()!=QProcess.NotRunning:
            QMessageBox.information(self,'Инструмент выполняется','Сначала завершите или остановите инструмент подготовки финала.'); event.ignore(); return
        if self.final_dashboard and hasattr(self.final_dashboard,'event_benchmark_dialog') and self.final_dashboard.event_benchmark_dialog.process.state()!=QProcess.NotRunning:
            QMessageBox.information(self,'Оценка событий выполняется','Сначала завершите или остановите инструмент оценки событий.'); event.ignore(); return
        if hasattr(self,'finale_dialog'):
            if self.finale_dialog.series:self.finale_dialog.series.interrupt('Закрыто основное окно; опыт прерван, длительность не засчитана.')
            self.finale_dialog.timer.stop();self.finale_dialog.close()
        self.timer.stop()
        for panel in self.panels:
            panel.stop()
        self.scheduler.close()
        if self.scheduler._thread.is_alive():
            QMessageBox.information(self,'Инференс завершается','Дождитесь завершения текущего вызова модели и закройте окно снова.'); event.ignore(); return
        for panel in self.panels:
            if hasattr(panel.capture,'wait_closed'): panel.capture.wait_closed()
        if getattr(self, '_second_camera_window', None) is not None:
            self._second_camera_window.close()
        if getattr(self,'route_memory_window',None): self.route_memory_window.close()
        if getattr(self,'final_dashboard',None): self.final_dashboard.close()
        self.event_store.close()
        event.accept()


def main() -> int:
    try:
        config = load_config()
    except Exception as exc:
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "Ошибка конфигурации", str(exc))
        return 1
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
