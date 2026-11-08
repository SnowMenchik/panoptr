"""Quick hardware diagnostic for connected USB cameras across DirectShow and Media Foundation."""
import sys, time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
APP = ROOT / "02_Реализация/panoptr"
sys.path.insert(0, str(APP))

print("=== 1. Список устройств DirectShow (pygrabber) ===")
try:
    from pygrabber.dshow_graph import FilterGraph
    for idx, name in enumerate(FilterGraph().get_input_devices()):
        print(f"  [ID {idx}] {name}")
except Exception as e:
    print("  Ошибка pygrabber:", e)

print("\n=== 2. Список устройств QtMultimedia (QMediaDevices) ===")
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtMultimedia import QMediaDevices
    app = QApplication.instance() or QApplication([])
    for idx, dev in enumerate(QMediaDevices.videoInputs()):
        print(f"  [Qt {idx}] {dev.description()}")
except Exception as e:
    print("  Ошибка QtMultimedia:", e)

print("\n=== 3. Проверка открытия индексов 0..3 в OpenCV (CAP_DSHOW и CAP_MSMF) ===")
for backend_name, backend in [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", getattr(cv2, "CAP_MSMF", cv2.CAP_ANY))]:
    for idx in range(4):
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(idx, backend)
        opened = cap.isOpened()
        ok, frame = (False, None)
        if opened:
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    break
                time.sleep(0.04)
        shape = frame.shape if (ok and frame is not None) else None
        cap.release()
        print(f"  {backend_name:10s} index={idx}: isOpened={str(opened):5s} read_ok={str(bool(ok)):5s} shape={shape} ({time.perf_counter()-t0:.2f}s)")
