"""Светлая и тёмная темы: один шаблон QSS, две палитры."""

from __future__ import annotations

LIGHT = {
    "bg": "#fafafa", "panel": "#ffffff", "text": "#1f2937", "muted": "#6b7280",
    "border": "#d1d5db", "border_soft": "#e5e7eb", "button": "#f3f4f6",
    "button_hover": "#e5e7eb", "button_pressed": "#d1d5db",
    "accent": "#2563eb", "accent_hover": "#1d4ed8", "accent_pressed": "#1e40af",
    "accent_soft": "#dbeafe", "drop_bg": "#f9fafb", "drop_hover": "#eff6ff",
    "danger": "#dc2626", "ok": "#15803d", "warn": "#b45309",
}

DARK = {
    "bg": "#1e1f22", "panel": "#2b2d31", "text": "#e5e7eb", "muted": "#9ca3af",
    "border": "#3f4248", "border_soft": "#34373c", "button": "#34373c",
    "button_hover": "#3f4248", "button_pressed": "#4b4f56",
    "accent": "#3b82f6", "accent_hover": "#2563eb", "accent_pressed": "#1d4ed8",
    "accent_soft": "#1e3a5f", "drop_bg": "#26282c", "drop_hover": "#1e2a3d",
    "danger": "#f87171", "ok": "#4ade80", "warn": "#fbbf24",
}

THEMES = {"light": LIGHT, "dark": DARK}

_TEMPLATE = """
QMainWindow, QDialog {{ background-color: {bg}; }}
QWidget {{ color: {text}; }}
QGroupBox {{
    font-weight: 600; border: 1px solid {border_soft}; border-radius: 6px;
    margin-top: 10px; padding: 10px 8px 6px 8px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {muted}; }}
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    border: 1px solid {border}; border-radius: 4px; padding: 4px 6px;
    background: {panel}; color: {text}; selection-background-color: {accent};
}}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{ border-color: {accent}; }}
QComboBox QAbstractItemView {{
    background: {panel}; color: {text}; selection-background-color: {accent_soft};
    selection-color: {text}; border: 1px solid {border};
}}
QPushButton {{
    background-color: {button}; border: 1px solid {border}; border-radius: 4px;
    padding: 4px 14px; min-height: 24px;
}}
QPushButton:hover {{ background-color: {button_hover}; }}
QPushButton:pressed {{ background-color: {button_pressed}; }}
QPushButton:disabled {{ color: {muted}; border-color: {border_soft}; background: {bg}; }}
QPushButton#primaryButton {{
    background-color: {accent}; border-color: {accent_hover}; color: white; font-weight: 600;
}}
QPushButton#primaryButton:hover {{ background-color: {accent_hover}; }}
QPushButton#primaryButton:pressed {{ background-color: {accent_pressed}; }}
QPushButton#primaryButton:disabled {{ background-color: {button}; color: {muted}; }}
QTableWidget {{
    background: {panel}; border: 1px solid {border_soft}; border-radius: 4px;
    gridline-color: {border_soft}; selection-background-color: {accent_soft};
    selection-color: {text}; alternate-background-color: {bg};
}}
QHeaderView::section {{
    background-color: {button}; color: {muted}; border: none;
    border-bottom: 1px solid {border_soft}; padding: 4px 6px; font-weight: 600;
}}
QProgressBar {{
    border: 1px solid {border}; border-radius: 4px; text-align: center;
    background: {button}; color: {text}; max-height: 18px;
}}
QProgressBar::chunk {{ background-color: {accent}; border-radius: 3px; }}
QStatusBar {{ background: {button}; border-top: 1px solid {border_soft}; color: {muted}; }}
QToolTip {{ background-color: #111827; color: white; border: none; padding: 4px 8px; }}
QSplitter::handle {{ background: {border_soft}; }}
QMenuBar {{ background: {bg}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {accent_soft}; }}
QMenu {{ background: {panel}; border: 1px solid {border}; }}
DropZone {{
    border: 2px dashed {border}; border-radius: 8px; background: {drop_bg};
}}
DropZone[hover="true"] {{ border-color: {accent}; background: {drop_hover}; }}
QLabel#dropLabel {{ color: {muted}; }}
QLabel#mutedLabel {{ color: {muted}; }}
"""


def palette(theme: str):
    """Палитра Fusion: QSS не перекрашивает чекбоксы, полосы прокрутки и т. п."""
    from PySide6.QtGui import QColor, QPalette

    c = THEMES.get(theme, LIGHT)
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: c["bg"], QPalette.ColorRole.WindowText: c["text"],
        QPalette.ColorRole.Base: c["panel"], QPalette.ColorRole.AlternateBase: c["bg"],
        QPalette.ColorRole.Text: c["text"], QPalette.ColorRole.Button: c["button"],
        QPalette.ColorRole.ButtonText: c["text"], QPalette.ColorRole.Highlight: c["accent"],
        QPalette.ColorRole.HighlightedText: "#ffffff", QPalette.ColorRole.ToolTipBase: "#111827",
        QPalette.ColorRole.ToolTipText: "#ffffff", QPalette.ColorRole.PlaceholderText: c["muted"],
        QPalette.ColorRole.Mid: c["border"], QPalette.ColorRole.Dark: c["border"],
        QPalette.ColorRole.Light: c["panel"], QPalette.ColorRole.Midlight: c["border_soft"],
    }
    for role, value in roles.items():
        pal.setColor(role, QColor(value))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText,
                 QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(c["muted"]))
    return pal


def stylesheet(theme: str) -> str:
    return _TEMPLATE.format(**THEMES.get(theme, LIGHT))


def color(theme: str, name: str) -> str:
    return THEMES.get(theme, LIGHT)[name]
