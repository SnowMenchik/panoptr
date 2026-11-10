"""Failover across independent storage nodes.

A storage node is an independent filesystem root: an internal disk, a mounted
network share or another local directory.  The router keeps the availability
state of every node, probes failed nodes with bounded timeouts and routes each
operation to a node that is currently usable.  Writes are mirrored to all
currently available nodes before an operation is reported as saved.

This is not a consensus/replication engine and it is not a backup.  If only one
node is reachable, a write is durable only on that node.  When a node returns,
its stale copy is synchronised from a healthy node before it is used again;
conflicting copies are resolved in favour of the newest file, and JSONL logs are
merged line-by-line so lines that exist only on one side are not dropped.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

LOG = logging.getLogger("panoptr.storage")
if not LOG.handlers:
    LOG.addHandler(logging.NullHandler())


class NodeState(str, Enum):
    """The four states required by the storage failover contract."""

    ACTIVE = "ACTIVE"
    UNAVAILABLE = "UNAVAILABLE"
    DISABLED = "DISABLED"
    RECOVERING = "RECOVERING"


class StorageUnavailableError(RuntimeError):
    """Raised when no storage node can serve an operation."""

    def __init__(self, message: str, states: Mapping[str, str] | None = None,
                 errors: Mapping[str, str] | None = None):
        super().__init__(message)
        self.states = dict(states or {})
        self.errors = dict(errors or {})


@dataclass(frozen=True)
class FailoverSettings:
    """Configurable failover behaviour.

    ``connect_timeout_s`` bounds a health probe, ``operation_timeout_s`` bounds
    a single node operation, ``max_attempts`` bounds failover attempts per
    operation, and ``recovery_interval_s`` is the minimum delay between probes
    of an unavailable node.  ``failure_threshold`` consecutive failures move a
    node to UNAVAILABLE.  ``replicate_writes`` mirrors each write to every
    available node; ``sync_on_recovery`` repairs a returning node first.
    """

    connect_timeout_s: float = 2.0
    operation_timeout_s: float = 10.0
    max_attempts: int = 3
    retry_backoff_s: float = 0.05
    failure_threshold: int = 1
    recovery_interval_s: float = 5.0
    monitor_interval_s: float = 2.0
    replicate_writes: bool = True
    sync_on_recovery: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "FailoverSettings" | None) -> "FailoverSettings":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("Параметры отказоустойчивости должны быть объектом.")
        known = {field.name for field in cls.__dataclass_fields__.values()}
        data = {key: val for key, val in value.items() if key in known}
        result = cls(**data)
        return result.validate()

    def validate(self) -> "FailoverSettings":
        numeric = ("connect_timeout_s", "operation_timeout_s", "retry_backoff_s",
                   "recovery_interval_s", "monitor_interval_s")
        for key in numeric:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Неверный параметр отказоустойчивости: {key}.")
        if self.connect_timeout_s <= 0 or self.operation_timeout_s <= 0:
            raise ValueError("Таймауты должны быть положительными.")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("max_attempts должен быть целым числом не меньше 1.")
        if isinstance(self.failure_threshold, bool) or not isinstance(self.failure_threshold, int) or self.failure_threshold < 1:
            raise ValueError("failure_threshold должен быть целым числом не меньше 1.")
        for key in ("replicate_writes", "sync_on_recovery"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"Неверный параметр отказоустойчивости: {key}.")
        return self


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_with_timeout(operation: Callable[[], Any], timeout: float, description: str) -> Any:
    """Run an operation in a daemon thread and enforce a bounded wait.

    Python cannot kill a blocked thread.  If a filesystem call ignores the
    timeout, the node is marked unavailable and the operation is retried on
    another node; the abandoned call can only affect the failed node, which is
    synchronised before it is used again.
    """
    if not timeout or timeout <= 0:
        return operation()
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = operation()
        except BaseException as exc:  # re-raised in the caller thread
            result["error"] = exc

    thread = threading.Thread(target=target, name="storage-op", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"Таймаут операции хранилища ({description}, {timeout:g} с).")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _safe_relative(value: str | os.PathLike[str]) -> str:
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("Недопустимый путь внутри хранилища.")
    return path.as_posix()


def _atomic_write_bytes(root: Path, relative: str, data: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass


def _atomic_write_text(root: Path, relative: str, text: str) -> None:
    _atomic_write_bytes(root, relative, text.encode("utf-8"))


def _merge_jsonl(source: Path, target: Path) -> str | None:
    source_lines = source.read_text(encoding="utf-8").splitlines()
    target_lines = target.read_text(encoding="utf-8").splitlines()
    seen = set(source_lines)
    extra = [line for line in target_lines if line not in seen]
    if not extra:
        return None
    return "\n".join(source_lines + extra) + "\n"


class StorageNode:
    """Mutable runtime state of one independent storage root."""

    def __init__(self, node_id: str, root: str | os.PathLike[str], priority: float = 100.0):
        if not str(node_id).strip():
            raise ValueError("У узла хранения должен быть идентификатор.")
        self.id = str(node_id).strip()
        self.root = Path(root).expanduser().resolve()
        self.priority = float(priority)
        self.state = NodeState.ACTIVE
        self.manual_disabled = False
        self.consecutive_failures = 0
        self.last_error = ""
        self.last_success_at = 0.0
        self.last_failure_at = 0.0
        self.next_probe_at = 0.0
        self.writes_ok = 0
        self.writes_failed = 0
        self.reads_ok = 0
        self.reads_failed = 0
        self.io_lock = threading.RLock()
        self.probe_lock = threading.Lock()

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "root": str(self.root),
            "priority": self.priority,
            "state": self.state.value,
            "manual_disabled": self.manual_disabled,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
            "next_probe_at": self.next_probe_at,
            "writes_ok": self.writes_ok,
            "writes_failed": self.writes_failed,
            "reads_ok": self.reads_ok,
            "reads_failed": self.reads_failed,
        }


def _normalise_nodes(root: str | os.PathLike[str] | None,
                     nodes: Sequence[Any] | None) -> list[dict[str, Any]]:
    if nodes is None:
        if root is None:
            raise ValueError("Не задан корень хранилища.")
        nodes = [root]
    if isinstance(nodes, (str, os.PathLike)):
        nodes = [nodes]
    result: list[dict[str, Any]] = []
    for index, spec in enumerate(nodes):
        if isinstance(spec, StorageNode):
            result.append({"id": spec.id, "root": spec.root, "priority": spec.priority})
        elif isinstance(spec, (str, os.PathLike)):
            result.append({"id": f"node-{index + 1}", "root": spec, "priority": float(index)})
        elif isinstance(spec, Mapping):
            path = spec.get("root", spec.get("path"))
            if path is None:
                raise ValueError("У узла хранения не задан root/path.")
            result.append({
                "id": str(spec.get("id") or spec.get("name") or f"node-{index + 1}"),
                "root": path,
                "priority": spec.get("priority", index),
            })
        else:
            raise ValueError("Неверное описание узла хранения.")
    if not result:
        raise ValueError("Не задан ни один узел хранения.")
    identifiers = [item["id"] for item in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Идентификаторы узлов хранения должны быть уникальными.")
    return result


class StorageCluster:
    """Stateful router for a set of independent storage roots."""

    def __init__(self, root: str | os.PathLike[str] | None = None,
                 nodes: Sequence[Any] | None = None,
                 settings: Mapping[str, Any] | FailoverSettings | None = None,
                 create: bool = True,
                 monitor: bool = True):
        self.settings = FailoverSettings.from_mapping(settings)
        specs = _normalise_nodes(root, nodes)
        self._nodes = [StorageNode(spec["id"], spec["root"], spec["priority"]) for spec in specs]
        self._lock = threading.RLock()
        self._history: deque[dict[str, Any]] = deque(maxlen=1000)
        self._closed = False
        self._create = bool(create)
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        if self._create:
            for node in self._nodes:
                self._probe_node(node, initial=True)
        if monitor and self._create and self.settings.monitor_interval_s > 0:
            self.start_monitor()

    # ------------------------------------------------------------------ public
    @property
    def primary_root(self) -> Path:
        return self._nodes[0].root

    @property
    def roots(self) -> list[Path]:
        return [node.root for node in self._nodes]

    def node_statuses(self) -> list[dict[str, Any]]:
        with self._lock:
            return [node.snapshot() for node in sorted(self._nodes, key=lambda item: (item.priority, item.id))]

    def status(self) -> dict[str, Any]:
        return {
            "nodes": self.node_statuses(),
            "settings": self.settings.__dict__,
            "active": self.has_active(),
            "history": self.history(20),
        }

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._history)
        return rows[-limit:]

    def has_active(self) -> bool:
        with self._lock:
            return any(node.state == NodeState.ACTIVE and not node.manual_disabled for node in self._nodes)

    def ready(self) -> bool:
        """Cheap availability check used before accepting background work.

        A node may still be marked ACTIVE after its mount disappeared without an
        operation noticing yet.  If none of the active roots is a directory, the
        nodes are probed immediately so callers do not accept work that cannot be
        persisted.
        """
        if not self.has_active():
            self.probe_due(force=True)
            return self.has_active()
        with self._lock:
            active = self._active_locked()

        def root_available(node: StorageNode) -> bool:
            try:
                return bool(_run_with_timeout(lambda n=node: n.root.is_dir(),
                                              self.settings.connect_timeout_s,
                                              f"check {node.id}"))
            except Exception as exc:
                self._failure(node, exc, f"check {node.id}")
                return False

        if any(root_available(node) for node in active):
            return True
        self.probe_due(force=True)
        return self.has_active()

    def preferred_root(self) -> Path:
        with self._lock:
            active = self._active_locked()
        return active[0].root if active else self.primary_root

    def disable_node(self, node_id: str, reason: str = "") -> dict[str, Any]:
        with self._lock:
            node = self._node_locked(node_id)
            node.manual_disabled = True
            node.state = NodeState.DISABLED
            node.last_error = reason.strip() or "Узел отключён вручную."
            node.next_probe_at = 0.0
            self._record_locked("node_disabled", node, reason=node.last_error)
        return node.snapshot()

    def enable_node(self, node_id: str) -> dict[str, Any]:
        with self._lock:
            node = self._node_locked(node_id)
            node.manual_disabled = False
            node.state = NodeState.RECOVERING
            node.next_probe_at = 0.0
            self._record_locked("node_enable_requested", node)
        # A node returns to the pool only after a successful probe (and sync).
        self._probe_node(node, force=True)
        return node.snapshot()

    def sync_node(self, node_id: str) -> dict[str, Any]:
        with self._lock:
            node = self._node_locked(node_id)
            if node.manual_disabled:
                raise ValueError("Сначала включите узел: отключённый узел не синхронизируется автоматически.")
            source = self._best_source_locked(node)
        if source is not None:
            _run_with_timeout(lambda s=source, n=node: self._sync_trees(s.root, n.root),
                              self.settings.operation_timeout_s, f"sync {node.id}")
        return node.snapshot()

    def probe_due(self, force: bool = False) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            due = [node for node in self._nodes
                   if not node.manual_disabled
                   and (force or (node.state in (NodeState.UNAVAILABLE, NodeState.RECOVERING)
                                  and node.next_probe_at <= now))]
        for node in due:
            self._probe_node(node, force=force)
        return self.node_statuses()

    def start_monitor(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(target=self._monitor, name="storage-monitor", daemon=True)
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        self._monitor_stop.set()
        thread = self._monitor_thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(0.1, self.settings.connect_timeout_s + 0.2))
        self._monitor_thread = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop_monitor()

    # ------------------------------------------------------------------- read
    def read_bytes(self, relative: str, missing_ok: bool = False) -> bytes | None:
        relative = _safe_relative(relative)
        candidates = self._candidates()
        if not candidates:
            self.probe_due()
            candidates = self._candidates()
        if not candidates:
            raise StorageUnavailableError("Все узлы хранения недоступны.",
                                          self._state_map(), self._errors_map())
        found: list[tuple[int, StorageNode, bytes]] = []
        errors: dict[str, str] = {}
        for node in candidates:
            try:
                path = node.root / relative
                data = _run_with_timeout(lambda p=path: p.read_bytes(),
                                         self.settings.operation_timeout_s, f"read {relative}")
                stat = path.stat()
                found.append((stat.st_mtime_ns, node, data))
                with self._lock:
                    node.reads_ok += 1
            except FileNotFoundError:
                continue
            except Exception as exc:
                errors[node.id] = str(exc)
                self._failure(node, exc, f"read {relative}")
        if found:
            found.sort(key=lambda row: (row[0], -row[1].priority))
            newest = found[-1]
            if len({row[2] for row in found}) > 1:
                self._record("read_conflict", newest[1], path=relative,
                             nodes=[row[1].id for row in found])
            self._success(newest[1], f"read {relative}")
            return newest[2]
        if errors:
            raise StorageUnavailableError("Не удалось прочитать файл на всех проверенных узлах.",
                                          self._state_map(), errors)
        if missing_ok:
            return None
        raise FileNotFoundError(relative)

    def read_text(self, relative: str, missing_ok: bool = False) -> str | None:
        data = self.read_bytes(relative, missing_ok=missing_ok)
        return None if data is None else data.decode("utf-8")

    def exists(self, relative: str) -> bool:
        return self.find_path(relative, strict=False) is not None

    def find_path(self, relative: str, strict: bool = True) -> Path | None:
        relative = _safe_relative(relative)
        candidates = self._candidates()
        if not candidates:
            self.probe_due()
            candidates = self._candidates()
        if not candidates:
            raise StorageUnavailableError("Все узлы хранения недоступны.",
                                          self._state_map(), self._errors_map())
        found: list[tuple[int, StorageNode, Path]] = []
        errors: dict[str, str] = {}
        for node in candidates:
            path = node.root / relative
            try:
                stat = _run_with_timeout(lambda p=path: p.stat(),
                                         self.settings.operation_timeout_s, f"stat {relative}")
                if path.is_file():
                    found.append((stat.st_mtime_ns, node, path))
            except FileNotFoundError:
                continue
            except Exception as exc:
                errors[node.id] = str(exc)
                self._failure(node, exc, f"stat {relative}")
        if found:
            found.sort(key=lambda row: (row[0], -row[1].priority))
            return found[-1][2]
        if errors and strict:
            raise StorageUnavailableError("Не удалось проверить файл на всех узлах хранения.",
                                          self._state_map(), errors)
        return None

    def list_files(self, pattern: str) -> list[str]:
        pattern = _safe_relative(pattern)
        candidates = self._candidates()
        if not candidates:
            self.probe_due()
            candidates = self._candidates()
        if not candidates:
            raise StorageUnavailableError("Все узлы хранения недоступны.",
                                          self._state_map(), self._errors_map())
        result: set[str] = set()
        errors: dict[str, str] = {}
        for node in candidates:
            try:
                rows = _run_with_timeout(lambda n=node: list(n.root.glob(pattern)),
                                         self.settings.operation_timeout_s, f"list {pattern}")
                for path in rows:
                    if path.is_file():
                        result.add(path.relative_to(node.root).as_posix())
            except Exception as exc:
                errors[node.id] = str(exc)
                self._failure(node, exc, f"list {pattern}")
        if not result and errors:
            raise StorageUnavailableError("Не удалось получить список файлов хранилища.",
                                          self._state_map(), errors)
        return sorted(result)

    # --------------------------------------------------------------- generic op
    def execute(self, operation: Callable[[StorageNode], Any], description: str = "") -> Any:
        """Run ``operation(node)`` on a healthy node with bounded failover.

        At most ``max_attempts`` nodes are tried.  Every failed attempt marks the
        node unavailable, so a repeated call cannot cycle through the same node
        forever.
        """
        label = description or getattr(operation, "__name__", "operation")
        candidates = self._candidates()
        if not candidates:
            self.probe_due()
            candidates = self._candidates()
        if not candidates:
            raise StorageUnavailableError("Все узлы хранения недоступны.",
                                          self._state_map(), self._errors_map())
        errors: dict[str, str] = {}
        attempts = min(len(candidates), self.settings.max_attempts)
        for node in candidates[:attempts]:
            try:
                with node.io_lock:
                    value = _run_with_timeout(lambda n=node: operation(n),
                                              self.settings.operation_timeout_s, label)
                self._success(node, label)
                return value
            except Exception as exc:
                errors[node.id] = str(exc)
                self._failure(node, exc, label)
        raise StorageUnavailableError(
            f"Операция «{label}» не выполнена за {attempts} попыт(ки).",
            self._state_map(), errors)

    # ------------------------------------------------------------------ write
    def write_bytes(self, relative: str, data: bytes) -> int:
        relative = _safe_relative(relative)
        candidates = self._candidates()
        if not candidates:
            self.probe_due()
            candidates = self._candidates()
        if not candidates:
            raise StorageUnavailableError("Все узлы хранения недоступны; запись не сохранена.",
                                          self._state_map(), self._errors_map())
        targets = candidates if self.settings.replicate_writes else candidates[:1]
        saved = 0
        errors: dict[str, str] = {}
        for node in targets:
            try:
                with node.io_lock:
                    _run_with_timeout(lambda n=node, r=relative, d=data: _atomic_write_bytes(n.root, r, d),
                                      self.settings.operation_timeout_s, f"write {relative}")
                saved += 1
                with self._lock:
                    node.writes_ok += 1
                self._success(node, f"write {relative}")
            except Exception as exc:
                errors[node.id] = str(exc)
                with self._lock:
                    node.writes_failed += 1
                self._failure(node, exc, f"write {relative}")
        if saved == 0:
            raise StorageUnavailableError("Запись не сохранена ни на одном узле хранения.",
                                          self._state_map(), errors)
        if saved < len(targets):
            self._record("degraded_write", None, path=relative, saved=saved, total=len(targets))
        return saved

    def write_text(self, relative: str, text: str) -> int:
        return self.write_bytes(relative, text.encode("utf-8"))

    def append_text(self, relative: str, text: str) -> int:
        relative = _safe_relative(relative)
        candidates = self._candidates()
        if not candidates:
            self.probe_due()
            candidates = self._candidates()
        if not candidates:
            raise StorageUnavailableError("Все узлы хранения недоступны; запись не сохранена.",
                                          self._state_map(), self._errors_map())
        targets = candidates if self.settings.replicate_writes else candidates[:1]
        saved = 0
        errors: dict[str, str] = {}
        for node in targets:
            try:
                with node.io_lock:
                    _run_with_timeout(lambda n=node, r=relative, t=text: self._append_node(n, r, t),
                                      self.settings.operation_timeout_s, f"append {relative}")
                saved += 1
                with self._lock:
                    node.writes_ok += 1
                self._success(node, f"append {relative}")
            except Exception as exc:
                errors[node.id] = str(exc)
                with self._lock:
                    node.writes_failed += 1
                self._failure(node, exc, f"append {relative}")
        if saved == 0:
            raise StorageUnavailableError("Запись не сохранена ни на одном узле хранения.",
                                          self._state_map(), errors)
        if saved < len(targets):
            self._record("degraded_append", None, path=relative, saved=saved, total=len(targets))
        return saved

    def commit_tree(self, relative_dir: str, source_dir: str | os.PathLike[str]) -> int:
        """Atomically merge a prepared directory into every available node."""
        relative_dir = _safe_relative(relative_dir)
        source = Path(source_dir)
        files = [path for path in source.rglob("*") if path.is_file()]
        if not files:
            raise ValueError("Нет файлов для сохранения.")
        candidates = self._candidates()
        if not candidates:
            self.probe_due()
            candidates = self._candidates()
        if not candidates:
            raise StorageUnavailableError("Все узлы хранения недоступны; данные не сохранены.",
                                          self._state_map(), self._errors_map())
        targets = candidates if self.settings.replicate_writes else candidates[:1]
        saved = 0
        errors: dict[str, str] = {}
        for node in targets:
            try:
                with node.io_lock:
                    _run_with_timeout(lambda n=node, rd=relative_dir, s=source, fs=files: self._commit_tree_node(n, rd, s, fs),
                                      self.settings.operation_timeout_s, f"commit {relative_dir}")
                saved += 1
                with self._lock:
                    node.writes_ok += 1
                self._success(node, f"commit {relative_dir}")
            except Exception as exc:
                errors[node.id] = str(exc)
                with self._lock:
                    node.writes_failed += 1
                self._failure(node, exc, f"commit {relative_dir}")
        if saved == 0:
            raise StorageUnavailableError("Данные не сохранены ни на одном узле хранения.",
                                          self._state_map(), errors)
        if saved < len(targets):
            self._record("degraded_commit", None, path=relative_dir, saved=saved, total=len(targets))
        return saved

    # --------------------------------------------------------------- internals
    def _node_locked(self, node_id: str) -> StorageNode:
        for node in self._nodes:
            if node.id == node_id:
                return node
        raise ValueError(f"Нет узла хранения: {node_id}")

    def _active_locked(self) -> list[StorageNode]:
        active = [node for node in self._nodes
                  if node.state == NodeState.ACTIVE and not node.manual_disabled]
        return sorted(active, key=lambda item: (item.priority, item.id))

    def _candidates(self) -> list[StorageNode]:
        with self._lock:
            for node in self._nodes:
                if node.manual_disabled and node.state != NodeState.DISABLED:
                    node.state = NodeState.DISABLED
            return self._active_locked()

    def _state_map(self) -> dict[str, str]:
        with self._lock:
            return {node.id: node.state.value for node in self._nodes}

    def _errors_map(self) -> dict[str, str]:
        with self._lock:
            return {node.id: node.last_error for node in self._nodes if node.last_error}

    def _best_source_locked(self, target: StorageNode) -> StorageNode | None:
        for node in self._active_locked():
            if node.id != target.id:
                return node
        return None

    def _probe_node(self, node: StorageNode, initial: bool = False, force: bool = False) -> bool:
        if not node.probe_lock.acquire(blocking=False):
            return node.state == NodeState.ACTIVE
        try:
            with self._lock:
                if node.manual_disabled:
                    node.state = NodeState.DISABLED
                    return False
                previous = node.state
                if previous == NodeState.ACTIVE and not force and not initial:
                    return True
                node.state = NodeState.RECOVERING
                source = self._best_source_locked(node) if self.settings.sync_on_recovery else None
                self._record_locked("node_probe", node, previous=previous.value)
            try:
                _run_with_timeout(lambda n=node: self._probe_io(n),
                                  self.settings.connect_timeout_s, f"probe {node.id}")
                if source is not None and previous != NodeState.ACTIVE:
                    _run_with_timeout(lambda s=source, n=node: self._sync_trees(s.root, n.root),
                                      self.settings.operation_timeout_s, f"sync {node.id}")
                with self._lock:
                    if node.manual_disabled:
                        node.state = NodeState.DISABLED
                        return False
                    self._set_active_locked(node, recovered=previous != NodeState.ACTIVE)
                return True
            except Exception as exc:
                with self._lock:
                    self._set_failure_locked(node, exc)
                return False
        finally:
            node.probe_lock.release()

    def _probe_io(self, node: StorageNode) -> None:
        node.root.mkdir(parents=True, exist_ok=True)
        if not node.root.is_dir():
            raise OSError(f"Узел хранения не является папкой: {node.root}")
        marker = json.dumps({"id": node.id, "checked_utc": _utcnow()},
                            ensure_ascii=False).encode("utf-8")
        _atomic_write_bytes(node.root, ".panoptr_node.json", marker)
        if (node.root / ".panoptr_node.json").read_bytes() != marker:
            raise OSError("Узел хранения не прошёл проверку чтения/записи.")

    def _set_active_locked(self, node: StorageNode, recovered: bool = False) -> None:
        previous = node.state
        node.state = NodeState.ACTIVE
        node.consecutive_failures = 0
        node.last_error = ""
        node.last_success_at = time.monotonic()
        node.next_probe_at = 0.0
        if recovered or previous != NodeState.ACTIVE:
            self._record_locked("node_active", node, previous=previous.value)

    def _set_failure_locked(self, node: StorageNode, exc: BaseException) -> None:
        previous = node.state
        node.consecutive_failures += 1
        node.last_error = f"{type(exc).__name__}: {exc}"
        node.last_failure_at = time.monotonic()
        if node.manual_disabled:
            node.state = NodeState.DISABLED
        elif node.consecutive_failures >= self.settings.failure_threshold or previous == NodeState.RECOVERING:
            node.state = NodeState.UNAVAILABLE
            node.next_probe_at = time.monotonic() + self.settings.recovery_interval_s
        if previous != node.state:
            self._record_locked("node_unavailable", node, previous=previous.value)
            LOG.warning("storage node %s unavailable: %s", node.id, node.last_error)
        else:
            LOG.debug("storage node %s failure: %s", node.id, node.last_error)

    def _success(self, node: StorageNode, description: str) -> None:
        with self._lock:
            if node.state != NodeState.ACTIVE and not node.manual_disabled:
                self._set_active_locked(node, recovered=True)
            else:
                node.consecutive_failures = 0
                node.last_success_at = time.monotonic()

    def _failure(self, node: StorageNode, exc: BaseException, description: str) -> None:
        with self._lock:
            self._set_failure_locked(node, exc)

    def _record(self, kind: str, node: StorageNode | None, **extra: Any) -> None:
        with self._lock:
            self._record_locked(kind, node, **extra)

    def _record_locked(self, kind: str, node: StorageNode | None, **extra: Any) -> None:
        row = {"utc": _utcnow(), "kind": kind, "node": node.id if node else None,
               "state": node.state.value if node else None, **extra}
        self._history.append(row)
        if kind in ("node_unavailable", "node_active", "node_disabled", "degraded_write",
                    "degraded_append", "degraded_commit", "read_conflict"):
            LOG.info("storage %s %s", kind, row)

    def _append_node(self, node: StorageNode, relative: str, text: str) -> None:
        path = node.root / relative
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        _atomic_write_text(node.root, relative, existing + text)

    def _commit_tree_node(self, node: StorageNode, relative_dir: str,
                          source: Path, files: Iterable[Path]) -> None:
        target = node.root / relative_dir
        staging_root = node.root / ".staging"
        staging = staging_root / uuid.uuid4().hex
        try:
            staging.mkdir(parents=True, exist_ok=True)
            for path in files:
                destination = staging / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            target.mkdir(parents=True, exist_ok=True)
            for path in staging.rglob("*"):
                if not path.is_file():
                    continue
                destination = target / path.relative_to(staging)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            try:
                staging_root.rmdir()
            except OSError:
                pass

    def _sync_trees(self, source_root: Path, target_root: Path) -> None:
        """Merge a healthy node into a returning node without deleting data."""
        target_root.mkdir(parents=True, exist_ok=True)
        for path in source_root.rglob("*"):
            if path.is_dir():
                continue
            relative_path = path.relative_to(source_root)
            if (".staging" in relative_path.parts or path.name.startswith(".panoptr_")
                    or ".tmp" in path.name):
                continue
            relative = relative_path.as_posix()
            destination = target_root / relative_path
            if destination.exists():
                if path.suffix.lower() == ".jsonl":
                    merged = _merge_jsonl(path, destination)
                    if merged is not None:
                        _atomic_write_text(target_root, relative, merged)
                    continue
                try:
                    if destination.stat().st_mtime_ns >= path.stat().st_mtime_ns:
                        continue
                except OSError:
                    pass
                if path.suffix.lower() == ".json":
                    try:
                        if path.read_bytes() != destination.read_bytes():
                            LOG.warning("storage sync conflict for %s; keeping newest copy", relative)
                    except OSError:
                        pass
            _atomic_write_bytes(target_root, relative, path.read_bytes())

    def _monitor(self) -> None:
        interval = max(0.1, self.settings.monitor_interval_s)
        while not self._monitor_stop.wait(interval):
            try:
                self.probe_due()
            except Exception:
                LOG.exception("storage monitor failed")
        self._monitor_stop.clear()
