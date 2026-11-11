"""Offscreen tests for decision colours, operator notes, exports and trial buttons."""
import json
import os
import sys
import time
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
APP = Path(__file__).resolve().parents[1] / '02_Реализация' / 'panoptr'
sys.path.insert(0, str(APP))

try:
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:
    HAVE_QT = False

from ui_theme import DECISION_STYLES, ROUTE_STYLES, decision_style  # noqa: E402


def event(identifier, channel=0, status='unreviewed', note='', class_name='missing', location='zone_not_set', evidence=True):
    return {'event_id': identifier, 'channel': channel, 'class_name': class_name, 'role': 'defect',
            'observation_state': 'observed_repeatedly', 'operator_status': status, 'operator_note': note,
            'hits': 3, 'peak_confidence': .8, 'first_seen': 1., 'last_seen': 2., 'evidence_saved': evidence,
            'created_utc': '2026-11-06T10:00:00+00:00', 'box': [10, 10, 50, 50], 'sequence': 1,
            'zone': {'location': location, 'overlap': .5, 'basis': 'test'}, 'quality': {'assessed': True, 'reasons': []}}


class FakeStore:
    error = ''

    def __init__(self, root, rows):
        self.root = Path(root); self.rows = rows

    def events(self):
        return [dict(row) for row in self.rows]

    def decide(self, identifier, status, note=''):
        self.rows = [dict(row, operator_status=status, operator_note=note) if row['event_id'] == identifier else row for row in self.rows]


class FakeOptions:
    def __init__(self, model):
        self.model = model


class FakePanel:
    def __init__(self, channel=0, model='models/person.pt'):
        self.channel = channel; self.running = False; self._model = model
        self.detections = None; self.health = {'assessed': True, 'reasons': []}; self.capture_health = {}
        self.policy = {'max_age_ms': 1500.}
        self.capture = None

    def options(self):
        return FakeOptions(str(APP / self._model))


@unittest.skipUnless(HAVE_QT, 'PySide6 is not installed')
class EventDecisionColoursTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def dashboard(self, rows):
        from final_ui import FinalDashboard
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        store = FakeStore(Path(temp.name), rows)
        dialog = FinalDashboard([FakePanel(0), FakePanel(1)], store)
        dialog.notes_path = Path(temp.name) / 'operator_notes.json'
        dialog.notes_history = []
        dialog.show(); self.app.processEvents()
        return dialog

    def test_statuses_are_coloured_and_labelled(self):
        rows = [event('a' * 32, status='accepted', note='рельс повреждён'),
                event('b' * 32, status='rejected', note='тень'),
                event('c' * 32, status='closed', note='кадр нечитаемый'),
                event('d' * 32, status='unreviewed')]
        dialog = self.dashboard(rows)
        dialog.refresh_events(); self.app.processEvents()
        table = dialog.events_table
        self.assertEqual(table.columnCount(), 8)
        headers = [table.horizontalHeaderItem(i).text() for i in range(8)]
        self.assertIn('Решение оператора', headers)
        self.assertIn('Комментарий', headers)
        for index, status in enumerate(['accepted', 'rejected', 'closed', 'unreviewed']):
            expected = DECISION_STYLES[status]
            self.assertEqual(table.item(index, 4).text(), expected['label'])
            self.assertEqual(table.item(index, 0).background().color().name().lower(), expected['bg'].lower())
            self.assertEqual(table.item(index, 0).foreground().color().name().lower(), expected['fg'].lower())
        self.assertEqual(table.item(0, 7).text(), 'рельс повреждён')
        self.assertEqual(table.item(3, 7).text(), '—')
        self.assertIn('ждут решения', dialog.state.text().lower())
        self.assertIn('подтверждено 1', dialog.state.text().lower())
        dialog.close()

    def test_zone_and_snapshot_are_explained(self):
        rows = [event('a' * 32, location='not_applicable_aerial'), event('b' * 32, evidence=False)]
        dialog = self.dashboard(rows); dialog.refresh_events()
        self.assertEqual(dialog.events_table.item(0, 5).text(), 'Воздушное наблюдение')
        self.assertEqual(dialog.events_table.item(1, 6).text(), 'Нет')
        self.assertIn('БПЛА', dialog.zone_hint(rows[0]))
        self.assertIn('не настроена', dialog.zone_hint(rows[1]))
        dialog.close()

    def test_comment_history_recall_and_clearing(self):
        rows = [event('a' * 32), event('b' * 32)]
        dialog = self.dashboard(rows); dialog.refresh_events()
        dialog.events_table.selectRow(0)
        dialog.note.setEditText('осмотрено, дефект подтверждён'); dialog.decide('accepted')
        self.assertEqual(dialog.events_table.item(0, 4).text(), 'Подтверждено')
        self.assertEqual(dialog.events_table.item(0, 7).text(), 'осмотрено, дефект подтверждён')
        self.assertEqual(dialog.note.currentText(), '')
        self.assertIn('осмотрено, дефект подтверждён', dialog.notes_history)
        self.assertTrue(dialog.notes_path.is_file())
        stored = json.loads(dialog.notes_path.read_text(encoding='utf-8'))
        self.assertEqual(stored[0], 'осмотрено, дефект подтверждён')
        dialog.events_table.selectRow(1)
        dialog.note.setCurrentIndex(0)
        self.assertEqual(dialog.note.currentText(), 'осмотрено, дефект подтверждён')
        dialog.close()

    def test_unknown_status_falls_back_to_unreviewed(self):
        self.assertEqual(decision_style('nonsense')['label'], 'Не проверено')
        self.assertIn('accepted', DECISION_STYLES)

    def test_readiness_report_is_explained(self):
        dialog = self.dashboard([event('a' * 32)])
        temp = TemporaryDirectory(); self.addCleanup(temp.cleanup)
        dialog.active_output = Path(temp.name) / 'last_check.json'
        report = {'passed': True, 'python': '3.12.10', 'executable': 'C:/PANOPTR/data/envs/x/python.exe',
                  'modules': {'cv2': 'x', 'torch': 'y'}, 'worker_ok': True, 'cuda': {'available': False},
                  'errors': [], 'weights': [{'path': 'models/rail.pt', 'exists': True, 'sha256': 'ff774' * 8}],
                  'scope': 'камеры не проверяются'}
        dialog.active_output.write_text(json.dumps(report), encoding='utf-8')
        dialog.apply_readiness(0)
        text = dialog.ready_summary.toPlainText()
        for expected in ('ИТОГ: ВСЁ ГОТОВО', 'cv2 — OK', 'Дочерний процесс Python: OK', 'CPU', 'Веса канала 1', 'камеры не проверяются'):
            self.assertIn(expected, text)
        self.assertIn('"passed": true', dialog.ready_raw.toPlainText())
        dialog.close()


@unittest.skipUnless(HAVE_QT, 'PySide6 is not installed')
class TrialButtonsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_dialog(self):
        from finale_ui import FinaleDialog

        class Window:
            def __init__(self):
                self.panels = [FakePanel(0), FakePanel(1)]
                self.event_store = FakeStore

            def _open_final_dashboard(self):
                pass

            def _open_route_memory(self):
                pass

            def _open_dataset_lab(self):
                pass

            def showNormal(self):
                pass

            def raise_(self):
                pass

            def activateWindow(self):
                pass

        temp = TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        window = Window()
        store = type('Store', (), {})()
        store.root = root; store.events = lambda: []
        window.event_store = store
        dialog = FinaleDialog(window)
        dialog.show(); self.app.processEvents()
        return dialog, root

    def test_buttons_follow_trial_state(self):
        from finale import TrialSeries
        dialog, root = self.make_dialog()
        self.assertFalse(dialog.begin_button.isEnabled())
        self.assertFalse(dialog.end_button.isEnabled())
        self.assertFalse(dialog.judge_button.isEnabled())
        series = TrialSeries(root / 'series', {'channel': 0}, 'Тестер', root)
        series.begin('target', 'missing', 'свет дневной')
        dialog.series = series
        dialog.refresh_trials()
        self.assertTrue(dialog.end_button.isEnabled())
        self.assertFalse(dialog.begin_button.isEnabled())
        self.assertFalse(dialog.judge_button.isEnabled())
        self.assertEqual(dialog.trials.item(0, 4).text(), 'идёт')
        time.sleep(0.01)
        series.finish()
        dialog.refresh_trials()
        self.assertFalse(dialog.end_button.isEnabled())
        self.assertTrue(dialog.judge_button.isEnabled())
        self.assertEqual(dialog.trials.item(0, 4).text(), 'завершён, ждёт оценки')
        self.assertEqual(dialog.trials.item(0, 0).background().color().name().lower(), '#eef3fb')
        with self.assertRaisesRegex(ValueError, 'нет ни одной карточки'):
            dialog.judge()
        with self.assertRaisesRegex(ValueError, 'уже завершён'):
            dialog.end_trial()
        dialog.close()

    def test_live_preview_reports_channel_state(self):
        import numpy as np
        dialog, root = self.make_dialog()
        panel = dialog.panel(); panel.running = True
        panel.capture = type('Capture', (), {'latest': lambda self: (1, np.zeros((48, 64, 3), dtype='uint8'))})()
        dialog.refresh_live()
        self.assertIn('запущен', dialog.live_state.text())
        self.assertIn('объектов в последнем кадре', dialog.live_state.text())
        self.assertFalse(dialog.live_preview.pixmap().isNull())
        dialog.close()


@unittest.skipUnless(HAVE_QT, 'PySide6 is not installed')
class RouteColoursAndEmptyStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_empty_map_explains_where_events_are(self):
        from route_memory import RouteMemory
        from route_memory_ui import RouteMemoryWindow
        temp = TemporaryDirectory(); self.addCleanup(temp.cleanup)
        memory = RouteMemory(Path(temp.name) / 'routes.sqlite')
        window = RouteMemoryWindow(memory.path)
        window.show(); self.app.processEvents()
        texts = ' '.join(item.toPlainText() for item in window.scene.items() if hasattr(item, 'toPlainText'))
        self.assertIn('Нет объектов с проверенным местом', texts)
        self.assertIn('объектов с проверенным местом: 0', window.status.text())
        window.close()

    def test_confirmed_rows_are_green(self):
        self.assertEqual(ROUTE_STYLES['Подтверждено']['bg'], '#E4F4EA')
        self.assertEqual(ROUTE_STYLES['Устарело']['fg'], '#41506B')


class ExportContentsTest(unittest.TestCase):
    def test_zip_lists_csv_json_summary_telemetry_and_evidence(self):
        import numpy as np
        from final_store import EventStore
        temp = TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        store = EventStore(root / 'session')
        frames = [(time.monotonic(), np.zeros((32, 48, 3), dtype='uint8'))]
        row = event('f' * 32)
        self.assertTrue(store.submit(row, frames, {'captured_at_monotonic': frames[0][0], 'source': 'test'}))
        store.flush()
        output = store.export(root / 'report')
        archive = Path(str(output) + '.zip')
        self.assertTrue(archive.is_file())
        with zipfile.ZipFile(archive) as handle:
            names = set(handle.namelist())
        for required in ('events.csv', 'events.json', 'summary.json', 'telemetry.json', 'evidence/' + 'f' * 32 + '/evidence.png'):
            self.assertIn(required, names)
        summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['operator_accepted'], 0)
        self.assertIsNone(summary['recall'])
        store.close()


if __name__ == '__main__':
    unittest.main()
