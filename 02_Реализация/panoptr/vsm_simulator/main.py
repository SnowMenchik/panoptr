from __future__ import annotations

# Reuse the local laboratory runtime when opened by the offline application.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset_lab.bootstrap import setup
setup()

import json
import multiprocessing as mp
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from effects import Scenario
from metrics import METRIC_LABELS
from runner import run_simulation


ROOT = Path(__file__).resolve().parent
APP_ROOT = ROOT.parent


class SimulationThread(QThread):
    progress_changed = Signal(int, str)
    preview_ready = Signal(object)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, arguments: dict) -> None:
        super().__init__()
        self.arguments = arguments
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            report = run_simulation(
                **self.arguments,
                stop_requested=self._stop.is_set,
                progress=self.progress_changed.emit,
                preview=self.preview_ready.emit,
            )
            self.completed.emit(report)
        except Exception as exc:
            self.failed.emit(str(exc))


def pixmap(frame: np.ndarray, target: QLabel) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image).scaled(target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Симулятор ВСМ — видео, отказы и метрики")
        self.resize(1500, 900)
        self.worker: SimulationThread | None = None

        self.video = QLineEdit()
        self.model = QLineEdit(self._default_model())
        self.ground_truth = QLineEdit()
        self.output = QLineEdit(str(ROOT / "results"))
        files = QGroupBox("Входные данные")
        files_grid = QGridLayout(files)
        self._file_row(files_grid, 0, "Видео", self.video, self._browse_video)
        self._file_row(files_grid, 1, "Модель", self.model, self._browse_model)
        self._file_row(files_grid, 2, "Эталон JSON (необязательно)", self.ground_truth, self._browse_truth)
        self._file_row(files_grid, 3, "Папка отчёта", self.output, self._browse_output)

        self.speed = QComboBox()
        self.speed.addItems(["300", "350", "400"])
        self.reference_speed = QSpinBox(); self.reference_speed.setRange(50, 500); self.reference_speed.setValue(300)
        self.motion_blur = QSpinBox(); self.motion_blur.setRange(0, 40); self.motion_blur.setValue(0)
        self.vibration = QSpinBox(); self.vibration.setRange(0, 20)
        self.brightness = QDoubleSpinBox(); self.brightness.setRange(0.3, 2.0); self.brightness.setSingleStep(0.1); self.brightness.setValue(1.0)
        self.weather = QComboBox()
        self.weather.addItem("Нет", "none"); self.weather.addItem("Дождь", "rain"); self.weather.addItem("Снег", "snow"); self.weather.addItem("Туман", "fog")
        self.weather_strength = QSpinBox(); self.weather_strength.setRange(0, 100); self.weather_strength.setSuffix(" %")
        environment = QGroupBox("Имитация движения и среды")
        environment_form = QFormLayout(environment)
        environment_form.addRow("Скорость ВСМ, км/ч", self.speed)
        environment_form.addRow("Скорость исходного видео, км/ч", self.reference_speed)
        environment_form.addRow("Размытие движения", self.motion_blur)
        environment_form.addRow("Вибрация, пикс.", self.vibration)
        environment_form.addRow("Яркость", self.brightness)
        environment_form.addRow("Погода", self.weather)
        environment_form.addRow("Интенсивность погоды", self.weather_strength)

        self.frame_drop = QSpinBox(); self.frame_drop.setRange(0, 70); self.frame_drop.setSuffix(" %")
        self.camera_start = QDoubleSpinBox(); self.camera_start.setRange(0, 86400); self.camera_start.setSuffix(" с")
        self.camera_duration = QDoubleSpinBox(); self.camera_duration.setRange(0, 3600); self.camera_duration.setSuffix(" с")
        self.gnss_start = QDoubleSpinBox(); self.gnss_start.setRange(0, 86400); self.gnss_start.setSuffix(" с")
        self.gnss_duration = QDoubleSpinBox(); self.gnss_duration.setRange(0, 3600); self.gnss_duration.setSuffix(" с")
        self.cpu_load = QSpinBox(); self.cpu_load.setRange(0, 75); self.cpu_load.setSuffix(" %")
        self.gpu_load = QSpinBox(); self.gpu_load.setRange(0, 50); self.gpu_load.setSuffix(" %")
        failures = QGroupBox("Отказы и перегрузка")
        failures_form = QFormLayout(failures)
        failures_form.addRow("Случайный пропуск кадров", self.frame_drop)
        failures_form.addRow("Отключение камеры с", self.camera_start)
        failures_form.addRow("Длительность камеры", self.camera_duration)
        failures_form.addRow("Потеря GNSS с", self.gnss_start)
        failures_form.addRow("Длительность GNSS", self.gnss_duration)
        failures_form.addRow("Дополнительная CPU-нагрузка", self.cpu_load)
        failures_form.addRow("Дополнительная GPU-нагрузка", self.gpu_load)

        self.device = QComboBox(); self.device.addItems(["Auto", "GPU", "CPU"]); self.device.setCurrentText('CPU')
        self.image_size = QComboBox(); self.image_size.addItems(["512", "640", "1024", "1280"]); self.image_size.setCurrentText("640")
        self.confidence = QDoubleSpinBox(); self.confidence.setRange(0.01, 0.99); self.confidence.setSingleStep(0.05); self.confidence.setValue(0.25)
        inference = QGroupBox("Inference")
        inference_form = QFormLayout(inference)
        inference_form.addRow("Устройство", self.device)
        inference_form.addRow("Размер", self.image_size)
        inference_form.addRow("Confidence", self.confidence)

        settings = QHBoxLayout()
        settings.addWidget(environment, 1)
        settings.addWidget(failures, 1)
        settings.addWidget(inference, 1)

        self.preview = QLabel("Предпросмотр появится после запуска")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet("background:#111;color:#aaa")
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Показатель", "Значение"])
        self.table.horizontalHeader().setStretchLastSection(True)
        results = QHBoxLayout()
        results.addWidget(self.preview, 2)
        results.addWidget(self.table, 1)

        self.start_button = QPushButton("Запустить прогон")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("Остановить")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        self.progress = QProgressBar()
        self.status = QLabel("Готов")
        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.progress, 1)
        controls.addWidget(self.status)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(files)
        layout.addLayout(settings)
        layout.addLayout(controls)
        layout.addLayout(results, 1)
        self.setCentralWidget(central)

    @staticmethod
    def _default_model() -> str:
        preferred = APP_ROOT / "models" / "rail_defects_yolo26s_v1.pt"
        return str(preferred if preferred.exists() else APP_ROOT / "yolo26n.pt")

    @staticmethod
    def _file_row(layout: QGridLayout, row: int, title: str, field: QLineEdit, callback) -> None:
        button = QPushButton("Выбрать…")
        button.clicked.connect(callback)
        layout.addWidget(QLabel(title), row, 0)
        layout.addWidget(field, row, 1)
        layout.addWidget(button, row, 2)

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Видео", "", "Видео (*.mp4 *.avi *.mkv *.mov);;Все файлы (*)")
        if path: self.video.setText(path)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Модель", str(APP_ROOT / "models"), "YOLO (*.pt *.onnx *.engine);;Все файлы (*)")
        if path: self.model.setText(path)

    def _browse_truth(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Эталон", "", "JSON (*.json)")
        if path: self.ground_truth.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка отчёта", self.output.text())
        if path: self.output.setText(path)

    def _scenario(self) -> Scenario:
        return Scenario(
            speed_kmh=int(self.speed.currentText()), reference_speed_kmh=self.reference_speed.value(),
            motion_blur=self.motion_blur.value(), vibration_px=self.vibration.value(), brightness=self.brightness.value(),
            weather=str(self.weather.currentData()), weather_strength=self.weather_strength.value(),
            frame_drop_percent=self.frame_drop.value(), camera_off_start_sec=self.camera_start.value(),
            camera_off_duration_sec=self.camera_duration.value(), gnss_loss_start_sec=self.gnss_start.value(),
            gnss_loss_duration_sec=self.gnss_duration.value(), cpu_load_percent=self.cpu_load.value(),
            gpu_load_percent=self.gpu_load.value(),
        )

    def _start(self) -> None:
        if not Path(self.video.text()).is_file():
            QMessageBox.warning(self, "Видео", "Выберите существующий видеофайл.")
            return
        if not Path(self.model.text()).is_file():
            QMessageBox.warning(self, "Модель", "Выберите существующую модель.")
            return
        if self.ground_truth.text() and not Path(self.ground_truth.text()).is_file():
            QMessageBox.warning(self, "Эталон", "Указанный JSON не найден.")
            return
        arguments = dict(
            video_path=self.video.text(), model_path=self.model.text(), ground_truth_path=self.ground_truth.text(),
            output_folder=self.output.text(), scenario=self._scenario(), device_request=self.device.currentText(),
            image_size=int(self.image_size.currentText()), confidence=self.confidence.value(),
        )
        self.worker = SimulationThread(arguments)
        self.worker.progress_changed.connect(self._progress)
        self.worker.preview_ready.connect(self._preview)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.status.setText("Выполняется…")
        self.worker.start()

    def _stop(self) -> None:
        if self.worker:
            self.worker.stop()
            self.status.setText("Остановка после текущего inference…")

    def _progress(self, value: int, text: str) -> None:
        self.progress.setValue(value); self.status.setText(text)

    def _preview(self, frame: object) -> None:
        self.preview.setPixmap(pixmap(frame, self.preview))

    def _completed(self, report: object) -> None:
        metrics = report["metrics"]
        self.table.setRowCount(len(metrics))
        for row, (key, value) in enumerate(metrics.items()):
            self.table.setItem(row, 0, QTableWidgetItem(METRIC_LABELS.get(key, key)))
            self.table.setItem(row, 1, QTableWidgetItem("N/A" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)))
        self.progress.setValue(100); self.status.setText("Завершено")
        self.start_button.setEnabled(True); self.stop_button.setEnabled(False)
        QMessageBox.information(self, "Готово", f"Отчёт сохранён:\n{report['report_json']}\n{report['report_csv']}")

    def _failed(self, message: str) -> None:
        self.status.setText("Остановлено с ошибкой")
        self.start_button.setEnabled(True); self.stop_button.setEnabled(False)
        QMessageBox.critical(self, "Симулятор ВСМ", message)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            if not self.worker.wait(10_000):
                QMessageBox.information(
                    self,
                    "Симулятор ВСМ",
                    "Завершается текущий inference. Закройте окно ещё раз через несколько секунд.",
                )
                event.ignore()
                return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
