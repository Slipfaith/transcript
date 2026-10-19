"""Зона для перетаскивания файлов лекций (и папок с ними)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout

from engine import MEDIA_EXTENSIONS

IDLE_TEXT = "Перетащите сюда файлы лекций или папку  ·  или нажмите, чтобы выбрать"
HOVER_TEXT = "Отпустите, чтобы добавить"
BAD_TEXT = "Здесь нет аудио или видео"
MIN_HEIGHT = 44


def media_files_from(paths: list[Path]) -> list[Path]:
    """Файлы с подходящими расширениями; папки обходятся рекурсивно."""
    found: list[Path] = []
    for p in paths:
        if p.is_dir():
            found += sorted(f for f in p.rglob("*")
                            if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS)
        elif p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS:
            found.append(p)
    return found


def paths_from_event(event: QDragEnterEvent | QDropEvent) -> list[Path]:
    mime = event.mimeData()
    if not mime.hasUrls():
        return []
    return [Path(u.toLocalFile()) for u in mime.urls() if u.isLocalFile()]


def accepts(paths: list[Path]) -> bool:
    return any(p.is_dir() or p.suffix.lower() in MEDIA_EXTENSIONS for p in paths)


class DropZone(QFrame):
    """Принимает файлы перетаскиванием; по щелчку просит открыть диалог выбора."""

    files_dropped = Signal(list)   # list[Path]
    browse_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Поддерживаются: " + " ".join(sorted(MEDIA_EXTENSIONS)))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self._label = QLabel(IDLE_TEXT)
        self._label.setObjectName("dropLabel")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

    def _set_hover(self, hover: bool, text: str) -> None:
        self.setProperty("hover", hover)
        self.style().unpolish(self)
        self.style().polish(self)
        self._label.setText(text)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        paths = paths_from_event(event)
        if paths and accepts(paths):
            event.acceptProposedAction()
            self._set_hover(True, HOVER_TEXT)
        else:
            self._set_hover(False, BAD_TEXT if paths else IDLE_TEXT)
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        self._set_hover(False, IDLE_TEXT)

    def dropEvent(self, event: QDropEvent) -> None:
        self._set_hover(False, IDLE_TEXT)
        files = media_files_from(paths_from_event(event))
        if files:
            event.acceptProposedAction()
            self.files_dropped.emit(files)
        else:
            event.ignore()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self.browse_requested.emit()
