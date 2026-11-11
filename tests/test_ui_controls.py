"""Offscreen smoke test of the themed dropdown and spin buttons."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '02_Реализация' / 'panoptr'))
try:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox
    from ui_theme import apply_competition_theme, _ASSETS
    HAVE_QT = True
except ImportError:
    HAVE_QT = False

@unittest.skipUnless(HAVE_QT, 'PySide6 is not installed')
class ThemedControlsTest(unittest.TestCase):
    def test_controls_render_and_arrows_work(self):
        app = QApplication.instance() or QApplication([])
        window = QWidget(); layout = QVBoxLayout(window)
        combo = QComboBox(); combo.addItems(['Camera', 'Video']); layout.addWidget(combo)
        spin = QSpinBox(); spin.setRange(1, 30); spin.setValue(8); layout.addWidget(spin)
        double = QDoubleSpinBox(); double.setRange(.05, .95); double.setSingleStep(.05); double.setValue(.2); layout.addWidget(double)
        window.resize(400, 150); apply_competition_theme(window); window.show(); app.processEvents()
        self.assertTrue((_ASSETS / 'arrow-up.png').is_file()); self.assertTrue((_ASSETS / 'arrow-down.png').is_file())
        self.assertFalse(window.grab().isNull())
        QTest.mouseClick(spin, Qt.LeftButton, pos=spin.rect().topRight() + QPoint(-10, 5))
        self.assertEqual(spin.value(), 9)
        QTest.mouseClick(double, Qt.LeftButton, pos=double.rect().bottomRight() + QPoint(-10, -5))
        self.assertAlmostEqual(double.value(), .15)
        QTest.mouseClick(combo, Qt.LeftButton, pos=combo.rect().topRight() + QPoint(-10, 10))
        self.assertTrue(combo.view().isVisible())
        combo.hidePopup(); window.close()

if __name__ == '__main__': unittest.main()
