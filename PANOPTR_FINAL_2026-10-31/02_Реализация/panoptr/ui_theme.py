"""Unified competition final design system («Инженеры транспорта» / ОАО «РЖД» / ВСМ)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

BRAND_BLUE = "#305BA7"
BRAND_BLUE_DARK = "#1E3F7A"
BRAND_BLUE_LIGHT = "#EEF3FB"
BRAND_TERRACOTTA = "#C84B31"
BRAND_RZD_RED = "#E21A1A"
SURFACE_BG = "#F4F7FB"
CARD_BG = "#FFFFFF"
TEXT_MAIN = "#0F172A"
TEXT_MUTED = "#475569"

COMPETITION_QSS = """
QMainWindow, QDialog {
    background-color: #F4F7FB;
    color: #0F172A;
}
QWidget {
    color: #0F172A;
    selection-background-color: #305BA7;
    selection-color: #FFFFFF;
}
QGroupBox {
    background-color: #FFFFFF;
    border: 1.5px solid #B8CBE8;
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px 8px 8px 8px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 2px 10px;
    background-color: #305BA7;
    color: #FFFFFF;
    border-radius: 4px;
    font-weight: 700;
}
QTabWidget::pane {
    background-color: #FFFFFF;
    border: 1.5px solid #B8CBE8;
    border-radius: 6px;
    top: -1px;
    padding: 6px;
}
QTabBar::tab {
    background-color: #E2E8F0;
    color: #1E293B;
    border: 1px solid #CBD5E1;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 6px 14px;
    margin-right: 3px;
    font-weight: 600;
}
QTabBar::tab:hover {
    background-color: #DBEAFE;
    color: #1E3A8A;
}
QTabBar::tab:selected {
    background-color: #305BA7;
    color: #FFFFFF;
    border-color: #305BA7;
    border-bottom: 3px solid #C84B31;
    font-weight: 700;
}
QPushButton {
    background-color: #FFFFFF;
    color: #1E3A8A;
    border: 1.5px solid #305BA7;
    border-radius: 6px;
    padding: 5px 12px;
    font-weight: 600;
    min-height: 22px;
}
QPushButton:hover {
    background-color: #EEF3FB;
    border-color: #1E3F7A;
}
QPushButton:pressed {
    background-color: #DBEAFE;
}
QPushButton:disabled {
    background-color: #E2E8F0;
    color: #94A3B8;
    border-color: #CBD5E1;
}
QPushButton[btnRole="primary"] {
    background-color: #305BA7;
    color: #FFFFFF;
    border: 1.5px solid #1E3F7A;
    font-weight: 700;
}
QPushButton[btnRole="primary"]:hover {
    background-color: #244785;
}
QPushButton[btnRole="accent"] {
    background-color: #C84B31;
    color: #FFFFFF;
    border: 1.5px solid #A3361F;
    font-weight: 700;
}
QPushButton[btnRole="accent"]:hover {
    background-color: #B03E26;
}
QPushButton[btnRole="danger"] {
    background-color: #FDF2F0;
    color: #B91C1C;
    border: 1.5px solid #DC2626;
    font-weight: 700;
}
QPushButton[btnRole="danger"]:hover {
    background-color: #FEE2E2;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background-color: #FFFFFF;
    color: #0F172A;
    border: 1px solid #94A3B8;
    border-radius: 5px;
    padding: 3px 6px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
    border: 1.5px solid #305BA7;
    background-color: #FAFCFF;
}
QTableWidget, QTableView {
    background-color: #FFFFFF;
    alternate-background-color: #F8FAFC;
    gridline-color: #CBD5E1;
    border: 1px solid #B8CBE8;
    border-radius: 5px;
    selection-background-color: #DBEAFE;
    selection-color: #0F172A;
}
QHeaderView::section {
    background-color: #305BA7;
    color: #FFFFFF;
    font-weight: 700;
    padding: 5px 6px;
    border: 1px solid #1E3F7A;
}
QScrollBar:vertical {
    background: #F1F5F9;
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #94A3B8;
    min-height: 24px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #305BA7;
}
"""


def _classify_button_role(text: str) -> str:
    low = (text or "").lower()
    if any(k in low for k in ("остановить", "отклонить", "удалить", "сброс")):
        return "danger"
    if any(k in low for k in ("полигон", "симулятор", "экспорт", "финал", "3d-стенд", "репетиц")):
        return "accent"
    if any(k in low for k in ("запустить", "применить", "подтвердить", "создать", "начать опыт", "оценить", "проверить", "сравнить")):
        return "primary"
    return "secondary"


def style_buttons_in_widget(root: QWidget) -> None:
    if not isinstance(root, QWidget):
        return
    for btn in root.findChildren(QPushButton):
        role = _classify_button_role(btn.text())
        if btn.property("btnRole") != role:
            btn.setProperty("btnRole", role)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
    for tbl in root.findChildren(QTableWidget):
        tbl.setAlternatingRowColors(True)


def create_competition_banner(title: str, subtitle: str = "", parent: QWidget | None = None) -> QFrame:
    banner = QFrame(parent)
    banner.setObjectName("competitionBanner")
    banner.setStyleSheet(
        """
        QFrame#competitionBanner {
            background-color: #FFFFFF;
            border: 1.5px solid #C84B31;
            border-left: 6px solid #305BA7;
            border-radius: 8px;
            padding: 4px 10px;
        }
        """
    )
    row = QHBoxLayout(banner)
    row.setContentsMargins(10, 5, 10, 5)
    row.setSpacing(10)

    left = QVBoxLayout()
    left.setSpacing(1)
    lbl_title = QLabel(title)
    lbl_title.setStyleSheet("font-size: 14px; font-weight: 800; color: #305BA7; background: transparent; border: none;")
    left.addWidget(lbl_title)
    if subtitle:
        lbl_sub = QLabel(subtitle)
        lbl_sub.setStyleSheet("font-size: 11px; color: #475569; background: transparent; border: none;")
        lbl_sub.setWordWrap(True)
        left.addWidget(lbl_sub)
    row.addLayout(left, 1)

    pill_track = QLabel("Железнодорожный транспорт • ВСМ")
    pill_track.setAlignment(Qt.AlignCenter)
    pill_track.setStyleSheet(
        """
        background-color: #FFFFFF;
        color: #0F172A;
        border: 1.5px solid #C84B31;
        border-radius: 10px;
        padding: 3px 10px;
        font-size: 10px;
        font-weight: 700;
        """
    )
    pill_it = QLabel("ИНЖЕНЕРЫ ТРАНСПОРТА • ОАО «РЖД»")
    pill_it.setAlignment(Qt.AlignCenter)
    pill_it.setStyleSheet(
        """
        background-color: #305BA7;
        color: #FFFFFF;
        border: 1.5px solid #1E3F7A;
        border-radius: 10px;
        padding: 3px 10px;
        font-size: 10px;
        font-weight: 700;
        """
    )
    row.addWidget(pill_track)
    row.addWidget(pill_it)
    return banner


def apply_competition_theme(widget: QWidget | None = None) -> None:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor("#F4F7FB"))
        pal.setColor(QPalette.WindowText, QColor("#0F172A"))
        pal.setColor(QPalette.Base, QColor("#FFFFFF"))
        pal.setColor(QPalette.AlternateBase, QColor("#F8FAFC"))
        pal.setColor(QPalette.Text, QColor("#0F172A"))
        pal.setColor(QPalette.Button, QColor("#FFFFFF"))
        pal.setColor(QPalette.ButtonText, QColor("#1E3A8A"))
        pal.setColor(QPalette.Highlight, QColor("#305BA7"))
        pal.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
        app.setPalette(pal)
        if "QPushButton[btnRole=" not in (app.styleSheet() or ""):
            app.setStyleSheet(COMPETITION_QSS)
    if isinstance(widget, QWidget):
        style_buttons_in_widget(widget)
