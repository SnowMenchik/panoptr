"""Bounded asynchronous evidence storage; disk failures never block acquisition.

Storage roots are routed through :mod:`storage_failover`.  A single root keeps
the original behaviour; several roots form a failover pool with synchronous
mirroring of every successful write and automatic routing to healthy nodes.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import csv
import json
import queue
import tempfile
import threading
import time
import zipfile

import cv2
from PIL import Image

from storage_failover import NodeState, StorageCluster, StorageUnavailableError


def _json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _save_local_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_text(value), encoding="utf-8")
    temporary.replace(path)


class EventStore:
    def __init__(self, root, max_queue=24, nodes=None, settings=None, monitor=True):
        self.cluster = StorageCluster(root=root, nodes=nodes, settings=settings,
                                      create=True, monitor=monitor)
        self.root = self.cluster.primary_root
        self.queue = queue.Queue(maxsize=max_queue)
        self.error = ""
        self.warning = ""
        self.dropped = 0
        self.decisions = {}
        self._lock = threading.Lock()
        self._closed = False
        try:
            self.write_json("session.json", {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "model_outputs_are_not_ground_truth": True,
                "format": "panoptr_events_v1",
                "storage_nodes": self.cluster.node_statuses(),
            })
        except StorageUnavailableError as exc:
            self.error = "Хранилище недоступно: " + str(exc)
        self.thread = threading.Thread(target=self._worker, name="evidence-writer", daemon=True)
        self.thread.start()

    # --------------------------------------------------------------- public API
    @property
    def active_root(self) -> Path:
        return self.cluster.preferred_root()

    def write_json(self, relative, value) -> int:
        return self.cluster.write_text(relative, _json_text(value))

    def read_json(self, relative, missing_ok=False):
        text = self.cluster.read_text(relative, missing_ok=missing_ok)
        if text is None:
            return None
        return json.loads(text)

    def node_statuses(self):
        return self.cluster.node_statuses()

    def storage_status(self):
        return self.cluster.status()

    def storage_history(self, limit=100):
        return self.cluster.history(limit)

    def disable_node(self, node_id, reason=""):
        return self.cluster.disable_node(node_id, reason)

    def enable_node(self, node_id):
        return self.cluster.enable_node(node_id)

    def sync_node(self, node_id):
        return self.cluster.sync_node(node_id)

    def evidence_path(self, identifier):
        try:
            path = self.cluster.find_path(f"events/{identifier}/evidence.png")
        except (OSError, ValueError, StorageUnavailableError):
            path = None
        return path or self.root / "events" / identifier / "evidence.png"

    def submit(self, event, frames, metadata):
        if self._closed:
            return False
        identifier = event.get("event_id", "")
        if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("Неверный ID события.")
        if not self.cluster.ready():
            self.error = "Все узлы хранения недоступны; событие не принято."
            return False
        item = (copy.deepcopy(event),
                [(float(t), image.copy()) for t, image in frames],
                copy.deepcopy(metadata))
        try:
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            self.error = "Очередь записи заполнена: доказательство не сохранено."
            return False

    def _worker(self):
        while True:
            try:
                item = self.queue.get(timeout=.2)
            except queue.Empty:
                if self._closed:
                    return
                continue
            try:
                if item is None:
                    return
                if isinstance(item[0], str) and item[0] == "telemetry":
                    self.cluster.append_text("telemetry.jsonl", _json_text(item[1]) + "\n")
                    continue
                event, frames, metadata = item
                self._save_event(event, frames, metadata)
            except StorageUnavailableError as exc:
                self.error = "Хранилище недоступно: " + str(exc)
            except Exception as exc:
                self.error = "Ошибка записи: " + str(exc)
            finally:
                self.queue.task_done()

    def _save_event(self, event, frames, metadata):
        identifier = event["event_id"]
        relative_dir = f"events/{identifier}"
        existing_evidence = self.cluster.exists(f"{relative_dir}/evidence.png")
        with tempfile.TemporaryDirectory(prefix="panoptr_evidence_") as temporary:
            staging = Path(temporary)
            evidence_written = False
            if frames and not existing_evidence:
                evidence = staging / "evidence.tmp.png"
                Image.fromarray(cv2.cvtColor(frames[-1][1], cv2.COLOR_BGR2RGB)).save(evidence)
                evidence.replace(staging / "evidence.png")
                evidence_written = True
                clip_frames = [(t, f) for t, f in frames if f.shape == frames[-1][1].shape]
                if len(clip_frames) >= 2:
                    duration = clip_frames[-1][0] - clip_frames[0][0]
                    fps = (len(clip_frames) - 1) / duration if duration > 0 else 5.
                    height, width = clip_frames[-1][1].shape[:2]
                    writer = cv2.VideoWriter(str(staging / "context.avi"),
                                             cv2.VideoWriter_fourcc(*"MJPG"),
                                             max(1., min(60., fps)), (width, height))
                    if not writer.isOpened():
                        raise IOError("AVI codec unavailable")
                    try:
                        for _, frame in clip_frames:
                            writer.write(frame)
                    finally:
                        writer.release()
                    _save_local_json(staging / "clip_frames.json", {
                        "timestamps_monotonic": [t for t, _ in clip_frames],
                        "constant_fps_preview": True,
                        "exact_timing": "see timestamp map; AVI spacing is approximate",
                        "fps": fps,
                    })
            elif existing_evidence:
                # Repair an incomplete replica while the original evidence is kept.
                for name in ("evidence.png", "context.avi", "clip_frames.json"):
                    data = self.cluster.read_bytes(f"{relative_dir}/{name}", missing_ok=True)
                    if data is not None:
                        (staging / name).write_bytes(data)

            old = {}
            try:
                old_text = self.cluster.read_text(f"{relative_dir}/event.json", missing_ok=True)
                if old_text:
                    old = json.loads(old_text)
            except (OSError, ValueError):
                old = {}
            event = copy.deepcopy(event)
            event["provenance"] = metadata
            event["evidence_observation"] = old.get("evidence_observation", {
                "box": event["box"],
                "sequence": event["sequence"],
                "quality": event["quality"],
                "provenance": metadata,
            })
            event["evidence_ready_at_monotonic"] = old.get("evidence_ready_at_monotonic", time.monotonic())
            with self._lock:
                if identifier in self.decisions:
                    event.update(self.decisions[identifier])
            event["evidence_saved"] = bool(existing_evidence or evidence_written)
            _save_local_json(staging / "event.json", event)
            self.cluster.commit_tree(relative_dir, staging)

        try:
            self.cluster.append_text("observations.jsonl", _json_text(event) + "\n")
        except StorageUnavailableError as exc:
            self.warning = "Событие сохранено, но журнал наблюдений не записан: " + str(exc)

    def events(self):
        try:
            paths = self.cluster.list_files("events/*/event.json")
        except StorageUnavailableError as exc:
            self.error = "Хранилище недоступно: " + str(exc)
            return []
        rows = []
        for relative in paths:
            try:
                text = self.cluster.read_text(relative)
                if text is None:
                    continue
                row = json.loads(text)
                review_relative = relative.rsplit("/", 1)[0] + "/technical_review.json"
                review_text = self.cluster.read_text(review_relative, missing_ok=True)
                if review_text:
                    try:
                        row["technical_review"] = json.loads(review_text)
                    except (OSError, ValueError):
                        row["technical_review_error"] = "Не удалось прочитать отдельную запись обследования."
                rows.append(row)
            except StorageUnavailableError as exc:
                self.error = "Хранилище недоступно: " + str(exc)
            except (OSError, ValueError):
                pass
        return sorted(rows, key=lambda row: row["first_seen"])

    def save_review(self, identifier, values):
        from rail_reference import validate_review
        if not isinstance(identifier, str) or len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("Неверный event_id.")
        relative = f"events/{identifier}/event.json"
        with self._lock:
            text = self.cluster.read_text(relative)
            row = json.loads(text)
            review = validate_review(row["class_name"], values)
            review["recorded_utc"] = datetime.now(timezone.utc).isoformat()
            self.write_json(f"events/{identifier}/technical_review.json", review)
        return review

    def telemetry(self, row):
        if self._closed:
            return False
        try:
            self.queue.put_nowait(("telemetry", copy.deepcopy(row)))
            return True
        except queue.Full:
            self.dropped += 1
            self.error = "Очередь записи заполнена; часть телеметрии не сохранена."
            return False

    def decide(self, identifier, status, note=""):
        if status not in ("accepted", "rejected", "closed"):
            raise ValueError("Неизвестное решение оператора.")
        if not identifier or len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("Неверный event_id.")
        relative = f"events/{identifier}/event.json"
        with self._lock:
            text = self.cluster.read_text(relative)
            row = json.loads(text)
            decision = {
                "operator_status": status,
                "operator_note": note,
                "decision_utc": datetime.now(timezone.utc).isoformat(),
            }
            self.decisions[identifier] = decision
            row.update(decision)
            self.write_json(relative, row)
            try:
                self.cluster.append_text("decisions.jsonl",
                                         _json_text({"event_id": identifier, **decision}) + "\n")
            except StorageUnavailableError as exc:
                self.warning = "Решение сохранено в событии, но журнал решений не записан: " + str(exc)

    def flush(self):
        self.queue.join()

    def export(self, output):
        self.flush()
        output = Path(output).resolve()
        if (output.exists() or Path(str(output) + ".zip").exists()
                or any(output.is_relative_to(root) for root in self.cluster.roots)):
            raise ValueError("Для отчёта нужна новая папка вне журнала.")
        output.mkdir(parents=True)
        rows = self.events()
        _save_local_json(output / "events.json", rows)
        fields = ["event_id", "channel", "class_name", "role", "observation_state",
                  "operator_status", "hits", "peak_confidence", "first_seen", "last_seen",
                  "evidence_saved"]
        with (output / "events.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        telemetry = []
        telemetry_text = self.cluster.read_text("telemetry.jsonl", missing_ok=True)
        if telemetry_text:
            for line in telemetry_text.splitlines():
                try:
                    telemetry.append(json.loads(line))
                except ValueError:
                    pass
        import numpy as np
        latencies = [row["end_to_end_ms"] for row in telemetry if row.get("kind") == "accepted_result"]
        evidence_latencies = [
            max(0., (row["evidence_ready_at_monotonic"]
                     - row["evidence_observation"]["provenance"]["captured_at_monotonic"]) * 1000)
            for row in rows
            if row.get("evidence_saved")
            and "captured_at_monotonic" in row.get("evidence_observation", {}).get("provenance", {})
        ]
        _save_local_json(output / "telemetry.json", telemetry)
        _save_local_json(output / "summary.json", {
            "events": len(rows),
            "operator_accepted": sum(row["operator_status"] == "accepted" for row in rows),
            "operator_rejected": sum(row["operator_status"] == "rejected" for row in rows),
            "storage_error": self.error,
            "storage_warning": self.warning,
            "storage_nodes": self.cluster.node_statuses(),
            "dropped_evidence_jobs": self.dropped,
            "accepted_result_p95_end_to_end_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "delivered_frame_to_evidence_ready_p95_ms": float(np.percentile(evidence_latencies, 95)) if evidence_latencies else None,
            "latency_definitions": {
                "end_to_end_ms": "After VideoCapture.read delivery in host process to result callback; before evidence I/O. Not physical camera exposure-to-card latency.",
                "evidence_ready": "Host delivered frame to image/clip preparation before publishing event metadata. GUI display and operator delay are not included.",
            },
            "rejected_stale_or_mismatched_results": sum(row.get("kind") == "rejected_result" for row in telemetry),
            "recall": None,
            "false_positives_per_minute": None,
            "reason": "Independent ground truth and observation duration are required for quality metrics.",
        })
        with zipfile.ZipFile(str(output) + ".zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in output.iterdir():
                archive.write(path, path.name)
            for relative in self.cluster.list_files("events/**/*"):
                if ".tmp" in relative:
                    continue
                data = self.cluster.read_bytes(relative)
                if data is not None:
                    archive.writestr("evidence/" + relative.split("events/", 1)[-1], data)
        return output

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                self.error = "Запись ещё не завершилась; журнал сохранён частично."
            try:
                self.write_json("status.json", {
                    "closed": not self.thread.is_alive(),
                    "storage_error": self.error,
                    "storage_warning": self.warning,
                    "dropped_evidence_jobs": self.dropped,
                    "storage_nodes": self.cluster.node_statuses(),
                })
            except (OSError, StorageUnavailableError):
                pass
            self.cluster.close()


class ArchiveStore(EventStore):
    def __init__(self, root, nodes=None, settings=None):
        self.cluster = StorageCluster(root=root, nodes=nodes, settings=settings,
                                      create=False, monitor=False)
        session = self.cluster.find_path("session.json")
        if session is None:
            raise ValueError("Выберите папку с session.json.")
        self.root = session.parent
        self.queue = queue.Queue()
        self._lock = threading.Lock()
        self._closed = True
        self.error = ""
        self.warning = ""
        self.dropped = 0
        self.decisions = {}
        status = self.read_json("status.json", missing_ok=True) or {}
        self.error = status.get("storage_error", "") or (
            "Журнал ещё открыт или завершение сессии не зафиксировано."
            if not status.get("closed") else "")
        self.warning = status.get("storage_warning", "")
        self.dropped = status.get("dropped_evidence_jobs", 0)

    def close(self):
        self.cluster.close()
