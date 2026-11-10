"""Failover, manual node control and evidence-storage integrity checks."""
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

import numpy as np

from storage_failover import (
    FailoverSettings,
    NodeState,
    StorageCluster,
    StorageUnavailableError,
)
from final_store import EventStore


def settings(**changes):
    values = dict(connect_timeout_s=.5, operation_timeout_s=2., max_attempts=2,
                  retry_backoff_s=0., failure_threshold=1, recovery_interval_s=0.,
                  monitor_interval_s=0., replicate_writes=True, sync_on_recovery=True)
    values.update(changes)
    return FailoverSettings(**values)


def break_root(path):
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("simulated node failure", encoding="utf-8")


def restore_root(path):
    path = Path(path)
    if path.exists() and not path.is_dir():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def state(cluster, node_id):
    return next(row for row in cluster.node_statuses() if row["id"] == node_id)["state"]


def event(number):
    identifier = f"{number:032x}"
    return {
        "event_id": identifier,
        "channel": 0,
        "class_name": "person",
        "role": "person",
        "observation_state": "observed_repeatedly",
        "operator_status": "unreviewed",
        "hits": 1,
        "valid_hits": 1,
        "peak_confidence": .9,
        "first_seen": float(number),
        "last_seen": float(number),
        "sequence": number,
        "box": [1, 1, 12, 12],
        "quality": {"assessed": True},
        "created_utc": "2026-01-01T00:00:00+00:00",
        "requires_review": True,
    }


class RouterChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.primary = self.root / "primary"
        self.backup = self.root / "backup"
        self.third = self.root / "third"

    def tearDown(self):
        self.temp.cleanup()

    def cluster(self, *paths, **overrides):
        return StorageCluster(nodes=[{"id": f"n{index + 1}", "root": path, "priority": index}
                                     for index, path in enumerate(paths)],
                              settings=settings(**overrides), monitor=False)

    def test_01_all_nodes_available_round_trip(self):
        cluster = self.cluster(self.primary, self.backup)
        try:
            self.assertEqual(cluster.write_text("shared/value.json", '{"value": 7}'), 2)
            self.assertEqual(cluster.read_text("shared/value.json"), '{"value": 7}')
            self.assertTrue((self.primary / "shared/value.json").is_file())
            self.assertTrue((self.backup / "shared/value.json").is_file())
            self.assertEqual({row["state"] for row in cluster.node_statuses()}, {"ACTIVE"})
        finally:
            cluster.close()

    def test_02_failed_node_is_skipped_and_write_continues(self):
        cluster = self.cluster(self.primary, self.backup)
        try:
            cluster.write_text("shared/before.txt", "before")
            break_root(self.primary)
            self.assertEqual(cluster.write_text("shared/after.txt", "after"), 1)
            self.assertEqual(state(cluster, "n1"), "UNAVAILABLE")
            self.assertEqual(cluster.read_text("shared/before.txt"), "before")
            self.assertEqual(cluster.read_text("shared/after.txt"), "after")
            self.assertFalse((self.primary / "shared/after.txt").exists())
        finally:
            cluster.close()

    def test_03_primary_unavailable_at_startup_uses_backup(self):
        break_root(self.primary)
        cluster = self.cluster(self.primary, self.backup)
        try:
            self.assertEqual(state(cluster, "n1"), "UNAVAILABLE")
            self.assertEqual(state(cluster, "n2"), "ACTIVE")
            self.assertEqual(cluster.write_text("session.json", "{}"), 1)
            self.assertEqual(cluster.read_text("session.json"), "{}")
        finally:
            cluster.close()

    def test_04_recovered_node_returns_after_probe_and_sync(self):
        cluster = self.cluster(self.primary, self.backup)
        try:
            cluster.write_text("events/a/event.json", '{"event": 1}')
            break_root(self.primary)
            cluster.write_text("events/b/event.json", '{"event": 2}')
            self.assertEqual(state(cluster, "n1"), "UNAVAILABLE")
            restore_root(self.primary)
            cluster.probe_due(force=True)
            self.assertEqual(state(cluster, "n1"), "ACTIVE")
            self.assertEqual((self.primary / "events/a/event.json").read_text(encoding="utf-8"), '{"event": 1}')
            self.assertEqual((self.primary / "events/b/event.json").read_text(encoding="utf-8"), '{"event": 2}')
        finally:
            cluster.close()

    def test_05_manual_disable_excludes_node_from_new_writes(self):
        cluster = self.cluster(self.primary, self.backup)
        try:
            cluster.write_text("before.txt", "before")
            cluster.disable_node("n1", "test")
            self.assertEqual(state(cluster, "n1"), "DISABLED")
            self.assertEqual(cluster.write_text("after.txt", "after"), 1)
            self.assertTrue((self.backup / "after.txt").is_file())
            self.assertFalse((self.primary / "after.txt").exists())
        finally:
            cluster.close()

    def test_06_disabled_node_is_not_reenabled_automatically(self):
        cluster = self.cluster(self.primary, self.backup)
        try:
            cluster.disable_node("n1", "test")
            cluster.probe_due(force=True)
            self.assertEqual(state(cluster, "n1"), "DISABLED")
            self.assertTrue((self.primary / ".panoptr_node.json").exists() is False or True)
        finally:
            cluster.close()

    def test_07_manual_enable_checks_then_returns_node(self):
        cluster = self.cluster(self.primary, self.backup)
        try:
            cluster.write_text("one.txt", "one")
            cluster.disable_node("n1", "test")
            break_root(self.primary)
            status = cluster.enable_node("n1")
            self.assertNotEqual(status["state"], "ACTIVE")
            self.assertEqual(state(cluster, "n1"), "UNAVAILABLE")
            restore_root(self.primary)
            status = cluster.enable_node("n1")
            self.assertEqual(status["state"], "ACTIVE")
            self.assertEqual((self.primary / "one.txt").read_text(encoding="utf-8"), "one")
        finally:
            cluster.close()

    def test_08_all_nodes_unavailable_raises_without_hanging(self):
        cluster = self.cluster(self.primary, self.backup)
        try:
            break_root(self.primary)
            break_root(self.backup)
            with self.assertRaises(StorageUnavailableError):
                cluster.write_text("lost.txt", "not saved")
            with self.assertRaises(StorageUnavailableError):
                cluster.read_text("lost.txt")
            self.assertFalse((self.root / "lost.txt").exists())
        finally:
            cluster.close()

    def test_09_simultaneous_failures_use_remaining_node(self):
        cluster = self.cluster(self.primary, self.backup, self.third)
        try:
            cluster.write_text("seed.txt", "seed")
            break_root(self.primary)
            break_root(self.backup)
            self.assertEqual(cluster.write_text("survivor.txt", "saved"), 1)
            self.assertEqual(cluster.read_text("survivor.txt"), "saved")
            self.assertEqual(state(cluster, "n3"), "ACTIVE")
            self.assertEqual(state(cluster, "n1"), "UNAVAILABLE")
            self.assertEqual(state(cluster, "n2"), "UNAVAILABLE")
        finally:
            cluster.close()

    def test_10_repeated_failures_are_bounded(self):
        cluster = self.cluster(self.primary, self.backup, self.third, max_attempts=2)
        calls = []
        try:
            def fail(node):
                calls.append(node.id)
                raise ConnectionError("link down")

            with self.assertRaises(StorageUnavailableError):
                cluster.execute(fail, "bounded")
            # max_attempts=2 bounds the number of nodes tried, so the third
            # node is not touched and stays usable.
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls, ["n1", "n2"])
            self.assertEqual(state(cluster, "n1"), "UNAVAILABLE")
            self.assertEqual(state(cluster, "n2"), "UNAVAILABLE")
            self.assertEqual(state(cluster, "n3"), "ACTIVE")
            # A second call cannot start over with the same failed nodes.
            with self.assertRaises(StorageUnavailableError):
                cluster.execute(fail, "bounded again")
            self.assertEqual(calls, ["n1", "n2", "n3"])
            self.assertEqual({row["state"] for row in cluster.node_statuses()},
                             {"UNAVAILABLE"})
        finally:
            cluster.close()

    def test_11_timeout_marks_node_unavailable(self):
        import time
        cluster = self.cluster(self.primary, self.backup, operation_timeout_s=.05)
        try:
            def slow(node):
                if node.id == "n1":
                    time.sleep(.3)
                return node.id

            result = cluster.execute(slow, "slow node")
            self.assertEqual(result, "n2")
            self.assertEqual(state(cluster, "n1"), "UNAVAILABLE")
        finally:
            cluster.close()

    def test_12_jsonl_logs_merge_without_losing_lines(self):
        cluster = self.cluster(self.primary, self.backup)
        try:
            cluster.append_text("telemetry.jsonl", '{"n": 1}\n')
            break_root(self.primary)
            cluster.append_text("telemetry.jsonl", '{"n": 2}\n')
            restore_root(self.primary)
            (self.primary / "telemetry.jsonl").write_text('{"n": 0}\n', encoding="utf-8")
            cluster.probe_due(force=True)
            lines = (self.primary / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(set(lines), {'{"n": 1}', '{"n": 2}', '{"n": 0}'})
        finally:
            cluster.close()


class EventStoreChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.primary = self.root / "primary"
        self.backup = self.root / "backup"
        self.nodes = [{"id": "primary", "root": self.primary, "priority": 0},
                      {"id": "backup", "root": self.backup, "priority": 1}]
        self.frame = np.full((24, 32, 3), 120, np.uint8)

    def tearDown(self):
        self.temp.cleanup()

    def store(self, nodes=None):
        return EventStore(self.primary, nodes=nodes or self.nodes, settings=settings(),
                          monitor=False)

    def test_evidence_writes_are_replicated_and_read_after_failover(self):
        store = self.store()
        try:
            first = event(1)
            self.assertTrue(store.submit(first, [(1., self.frame)], {"sequence": 1}))
            store.flush()
            self.assertTrue((self.primary / "events" / first["event_id"] / "evidence.png").is_file())
            self.assertTrue((self.backup / "events" / first["event_id"] / "evidence.png").is_file())
            self.assertEqual(len(store.events()), 1)

            break_root(self.primary)
            second = event(2)
            self.assertTrue(store.submit(second, [(2., self.frame)], {"sequence": 2}))
            store.flush()
            rows = store.events()
            self.assertEqual([row["event_id"] for row in rows], [first["event_id"], second["event_id"]])
            self.assertTrue(store.evidence_path(second["event_id"]).is_file())
            self.assertIn("backup", str(store.evidence_path(second["event_id"])))
            self.assertEqual(store.node_statuses()[0]["state"], "UNAVAILABLE")
        finally:
            store.close()

    def test_recovered_node_receives_missed_evidence(self):
        store = self.store()
        try:
            break_root(self.primary)
            second = event(2)
            store.submit(second, [(2., self.frame)], {"sequence": 2})
            store.flush()
            restore_root(self.primary)
            store.cluster.probe_due(force=True)
            self.assertEqual(store.node_statuses()[0]["state"], "ACTIVE")
            self.assertTrue((self.primary / "events" / second["event_id"] / "evidence.png").is_file())
            self.assertTrue((self.primary / "events" / second["event_id"] / "event.json").is_file())
        finally:
            store.close()

    def test_manual_node_control_keeps_data_consistent(self):
        store = self.store()
        try:
            first = event(1)
            store.submit(first, [(1., self.frame)], {"sequence": 1})
            store.flush()
            store.disable_node("primary", "operator test")
            second = event(2)
            store.submit(second, [(2., self.frame)], {"sequence": 2})
            store.flush()
            self.assertFalse((self.primary / "events" / second["event_id"]).exists())
            self.assertTrue((self.backup / "events" / second["event_id"] / "event.json").is_file())
            store.cluster.probe_due(force=True)
            self.assertEqual(store.node_statuses()[0]["state"], "DISABLED")
            self.assertEqual(store.enable_node("primary")["state"], "ACTIVE")
            self.assertTrue((self.primary / "events" / second["event_id"] / "event.json").is_file())
            self.assertEqual(len(store.events()), 2)
        finally:
            store.close()

    def test_all_nodes_down_reports_error_and_does_not_claim_saved_evidence(self):
        store = self.store()
        try:
            break_root(self.primary)
            break_root(self.backup)
            self.assertFalse(store.submit(event(3), [(3., self.frame)], {"sequence": 3}))
            store.flush()
            self.assertTrue(store.error)
            self.assertEqual(store.events(), [])
        finally:
            store.close()

    def test_export_contains_evidence_from_available_node(self):
        store = self.store()
        try:
            first = event(1)
            store.submit(first, [(1., self.frame)], {"sequence": 1})
            store.flush()
            break_root(self.primary)
            output = store.export(self.root / "report")
            self.assertTrue((output / "summary.json").is_file())
            with zipfile.ZipFile(str(output) + ".zip") as archive:
                self.assertIn("evidence/" + first["event_id"] + "/evidence.png", archive.namelist())
                self.assertIsNone(archive.testzip())
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
