"""Главное окно: очередь файлов, параметры, прогресс, журнал."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path

from PySide6.QtCore import QProcess, QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QColor, QDragEnterEvent, QDropEvent, QFontMetrics
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

import engine
from engine import FileResult, Options
from gui import styles
from gui.drop_zone import DropZone, accepts, media_files_from, paths_from_event
from gui.worker import GpuProbe, TranscribeWorker
from outputs import FORMATS, fmt_clock

APP_TITLE = "Транскрибатор лекций"
PROJECT_DIR = Path(__file__).resolve().parent.parent
LANGUAGES = [("ru", "Русский"), ("en", "Английский"), ("auto", "Определить сам"),
             ("uk", "Украинский"), ("de", "Немецкий"), ("fr", "Французский"),
             ("es", "Испанский"), ("it", "Итальянский"), ("zh", "Китайский")]
DEVICES = [("cuda", "GPU (CUDA)"), ("cpu", "CPU (очень медленно)")]
COL_FILE, COL_DURATION, COL_STATUS, COL_SPEED = range(4)
ROLE_PATH = Qt.ItemDataRole.UserRole
ROLE_STATE = Qt.ItemDataRole.UserRole + 1
ROLE_OUTPUTS = Qt.ItemDataRole.UserRole + 2
WINDOW_SIZE = (1100, 620)
MIN_SIZE = (900, 520)
LOG_MAX_LINES = 5000
STOP_WAIT_MS = 60_000
MAX_SPEAKERS_CHOICE = 6
TABLE_MIN_HEIGHT = 80
LOG_MIN_HEIGHT = 60


class State(str, Enum):
    WAITING = "Ожидает"
    RUNNING = "В работе"
    DONE = "Готово"
    SKIPPED = "Уже расшифрован"
    FAILED = "Ошибка"
    STOPPED = "Остановлен"


STATE_COLOR = {State.DONE: "ok", State.SKIPPED: "muted", State.FAILED: "danger",
               State.STOPPED: "warn", State.RUNNING: "accent"}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(*MIN_SIZE)
        self.resize(*WINDOW_SIZE)
        self.setAcceptDrops(True)

        self._settings = QSettings("LectureTranscriber", "LectureTranscriber")
        self._theme = str(self._settings.value("theme", "light"))
        self._worker: TranscribeWorker | None = None
        self._env_proc: QProcess | None = None
        self._file_t0 = 0.0
        self._queue_total = 0
        self._queue_index = 0
        self._last_output_dir: Path | None = None

        self._build_menu()
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)
        self.drop_zone = DropZone()
        root.addWidget(self.drop_zone)
        root.addWidget(self._build_splitter(), stretch=1)
        root.addWidget(self._build_settings())
        root.addLayout(self._build_progress())
        root.addLayout(self._build_buttons())
        self.setCentralWidget(central)

        self._gpu_label = QLabel("Проверяю GPU ...")
        self.statusBar().addPermanentWidget(self._gpu_label)
        self._connect()
        self._load_settings()
        self._apply_theme(self._theme)
        self._update_buttons()
        self._probe_gpu()

    # ---------- построение интерфейса ----------

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("Файл")
        self.act_add = QAction("Добавить файлы…", self, shortcut="Ctrl+O")
        self.act_add_dir = QAction("Добавить папку…", self)
        act_quit = QAction("Выход", self, shortcut="Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addActions([self.act_add, self.act_add_dir])
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        view_menu = self.menuBar().addMenu("Вид")
        self.act_dark = QAction("Тёмная тема", self, checkable=True)
        self.act_dark.setChecked(self._theme == "dark")
        self.act_log = QAction("Журнал", self, checkable=True, shortcut="Ctrl+L")
        view_menu.addActions([self.act_dark, self.act_log])

        tools_menu = self.menuBar().addMenu("Сервис")
        self.act_check = QAction("Проверить окружение (GPU, CUDA)", self)
        self.act_logfile = QAction("Открыть лог скорости", self)
        tools_menu.addActions([self.act_check, self.act_logfile])

    def _build_splitter(self) -> QSplitter:
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Файл", "Длительность", "Статус", "Скорость"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(False)
        self.table.setMinimumHeight(TABLE_MIN_HEIGHT)
        self.table.setToolTip("Двойной щелчок по готовому файлу открывает расшифровку")
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_FILE, QHeaderView.ResizeMode.Stretch)
        for col in (COL_DURATION, COL_STATUS, COL_SPEED):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(LOG_MAX_LINES)
        self.log_view.setPlaceholderText("Здесь появляется распознанный текст и служебные сообщения")

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.table)
        self.splitter.addWidget(self.log_view)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setCollapsible(0, False)
        self.splitter.setSizes([1, 0])  # журнал свёрнут
        return self.splitter

    def _build_settings(self) -> QGroupBox:
        box = QGroupBox("Параметры")
        box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.model_combo = QComboBox()
        self.model_combo.addItems(engine.MODELS)
        self.model_combo.setToolTip(
            "large-v3-turbo — быстро и точно (по умолчанию)\n"
            "large-v3 — чуть точнее, примерно в 3–4 раза медленнее\n"
            "medium / small — быстрее, но хуже на терминах")
        self.lang_combo = QComboBox()
        for code, name in LANGUAGES:
            self.lang_combo.addItem(name, code)
        self.device_combo = QComboBox()
        for code, name in DEVICES:
            self.device_combo.addItem(name, code)
        self.device_combo.setToolTip("CPU только для проверки: часовая лекция будет идти часами")

        self.format_checks = {f: QCheckBox(f".{f}") for f in FORMATS}
        self.format_checks["txt"].setToolTip("Сплошной текст с абзацами")
        self.format_checks["srt"].setToolTip("Субтитры с таймкодами")
        self.format_checks["md"].setToolTip("Таймкоды по сегментам, удобно отдавать Claude")
        formats_row = QHBoxLayout()
        formats_row.setSpacing(8)
        for cb in self.format_checks.values():
            formats_row.addWidget(cb)
        formats_row.addStretch(1)

        grid.addWidget(QLabel("Модель:"), 0, 0)
        grid.addWidget(self.model_combo, 0, 1)
        grid.addWidget(QLabel("Язык:"), 0, 2)
        grid.addWidget(self.lang_combo, 0, 3)
        grid.addWidget(QLabel("Устройство:"), 0, 4)
        grid.addWidget(self.device_combo, 0, 5)
        grid.addWidget(QLabel("Форматы:"), 0, 6)
        grid.addLayout(formats_row, 0, 7)
        self._add_settings_row2(grid)
        for col in (1, 3, 5):
            grid.setColumnStretch(col, 1)
        grid.setColumnStretch(7, 1)
        self.settings_box = box
        return box

    def _add_settings_row2(self, grid: QGridLayout) -> None:
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("рядом с исходным файлом")
        self.out_edit.setToolTip("Пусто — результаты кладутся рядом с файлом лекции")
        self.out_browse = QPushButton("Обзор…")
        out_row = QHBoxLayout()
        out_row.setSpacing(4)
        out_row.addWidget(self.out_edit, stretch=1)
        out_row.addWidget(self.out_browse)

        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("необязательно: термины и фамилии из лекции")
        self.prompt_edit.setToolTip(
            "Например: «Теорема Гамильтона — Кэли, Коши, Лагранж».\n"
            "Помогает правильно писать термины, но немного замедляет работу.")
        self.prompt_edit.setClearButtonEnabled(True)
        self.overwrite_check = QCheckBox("Перезаписывать готовые")
        self.overwrite_check.setToolTip("Иначе уже расшифрованные файлы пропускаются")

        grid.addWidget(QLabel("Папка:"), 1, 0)
        grid.addLayout(out_row, 1, 1, 1, 3)
        grid.addWidget(QLabel("Термины:"), 1, 4)
        grid.addWidget(self.prompt_edit, 1, 5, 1, 3)

        self.speakers_check = QCheckBox("Разметить спикеров")
        self.speakers_check.setToolTip(
            "Подписывает реплики: «Лектор» (кто говорит больше всех), «Слушатель 1, 2…».\n"
            "Считается на CPU параллельно с распознаванием. Файлы пишутся в конце, а не по ходу.\n"
            "Модель голосов (25 МБ) скачивается при первом использовании.")
        self.num_speakers_combo = QComboBox()
        self.num_speakers_combo.addItem("определить сам", None)
        for n in range(2, MAX_SPEAKERS_CHOICE + 1):
            self.num_speakers_combo.addItem(str(n), n)
        self.num_speakers_combo.setToolTip(
            "Если точно знаете, сколько разных голосов в записи, укажите: разметка будет точнее")
        self.speakers_check.toggled.connect(self.num_speakers_combo.setEnabled)
        self.num_speakers_combo.setEnabled(False)
        grid.addWidget(QLabel("Спикеры:"), 2, 0)
        grid.addWidget(self.speakers_check, 2, 1)
        grid.addWidget(QLabel("Голосов:"), 2, 2)
        grid.addWidget(self.num_speakers_combo, 2, 3)
        grid.addWidget(self.overwrite_check, 2, 5, 1, 2)

    def _build_progress(self) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(4)
        self.progress_label = QLabel("Добавьте файлы и нажмите «Начать»")
        self.progress_label.setObjectName("mutedLabel")
        self.progress_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        box.addWidget(self.progress_label)
        box.addWidget(self.progress)
        return box

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_add = QPushButton("Добавить файлы…")
        self.btn_remove = QPushButton("Убрать выбранные")
        self.btn_clear = QPushButton("Очистить")
        self.btn_log = QPushButton("Журнал")
        self.btn_log.setCheckable(True)
        self.btn_open = QPushButton("Открыть папку с результатом")
        self.btn_stop = QPushButton("Остановить")
        self.btn_start = QPushButton("Начать")
        self.btn_start.setObjectName("primaryButton")
        self.btn_start.setMinimumWidth(110)
        for b in (self.btn_add, self.btn_remove, self.btn_clear):
            row.addWidget(b)
        row.addStretch(1)
        for b in (self.btn_log, self.btn_open, self.btn_stop, self.btn_start):
            row.addWidget(b)
        return row

    def _connect(self) -> None:
        self.drop_zone.files_dropped.connect(self.add_files)
        self.drop_zone.browse_requested.connect(self._browse_files)
        self.act_add.triggered.connect(self._browse_files)
        self.act_add_dir.triggered.connect(self._browse_dir)
        self.btn_add.clicked.connect(self._browse_files)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_open.clicked.connect(self._open_output_dir)
        self.btn_log.toggled.connect(self._set_log_visible)
        self.act_log.toggled.connect(self._set_log_visible)
        self.act_dark.toggled.connect(lambda on: self._apply_theme("dark" if on else "light"))
        self.act_check.triggered.connect(self._check_env)
        self.act_logfile.triggered.connect(self._open_speed_log)
        self.out_browse.clicked.connect(self._browse_out_dir)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.itemDoubleClicked.connect(self._open_row_result)

    # ---------- настройки ----------

    def _load_settings(self) -> None:
        s = self._settings
        for combo, key, default in ((self.model_combo, "model", engine.DEFAULT_MODEL),
                                    (self.lang_combo, "lang", "ru"),
                                    (self.device_combo, "device", "cuda")):
            value = str(s.value(key, default))
            combo.blockSignals(True)
            idx = combo.findData(value) if combo.findData(value) >= 0 else combo.findText(value)
            combo.setCurrentIndex(max(idx, 0))
            combo.blockSignals(False)
        saved_formats = s.value("formats", ",".join(FORMATS))
        for f, cb in self.format_checks.items():
            cb.setChecked(f in str(saved_formats).split(","))
        self.out_edit.setText(str(s.value("out_dir", "")))
        self.prompt_edit.setText(str(s.value("prompt", "")))
        self.overwrite_check.setChecked(s.value("overwrite", "false") == "true")
        self.speakers_check.setChecked(s.value("speakers", "false") == "true")
        idx = self.num_speakers_combo.findData(int(s.value("num_speakers", 0) or 0) or None)
        self.num_speakers_combo.setCurrentIndex(max(idx, 0))
        geometry = s.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def _save_settings(self) -> None:
        s = self._settings
        s.setValue("model", self.model_combo.currentText())
        s.setValue("lang", self.lang_combo.currentData())
        s.setValue("device", self.device_combo.currentData())
        s.setValue("formats", ",".join(self._selected_formats()))
        s.setValue("out_dir", self.out_edit.text().strip())
        s.setValue("prompt", self.prompt_edit.text().strip())
        s.setValue("overwrite", "true" if self.overwrite_check.isChecked() else "false")
        s.setValue("speakers", "true" if self.speakers_check.isChecked() else "false")
        s.setValue("num_speakers", self.num_speakers_combo.currentData() or 0)
        s.setValue("theme", self._theme)
        s.setValue("geometry", self.saveGeometry())

    def _selected_formats(self) -> list[str]:
        return [f for f, cb in self.format_checks.items() if cb.isChecked()]

    def _options(self) -> Options:
        out = self.out_edit.text().strip()
        return Options(
            model=self.model_combo.currentText(),
            lang=self.lang_combo.currentData(),
            device=self.device_combo.currentData(),
            out_dir=Path(out) if out else None,
            formats=self._selected_formats(),
            prompt=self.prompt_edit.text().strip() or None,
            diarize=self.speakers_check.isChecked(),
            num_speakers=self.num_speakers_combo.currentData(),
        )

    def _apply_theme(self, theme: str) -> None:
        self._theme = theme
        self.act_dark.blockSignals(True)
        self.act_dark.setChecked(theme == "dark")
        self.act_dark.blockSignals(False)
        app = QApplication.instance()
        app.setPalette(styles.palette(theme))
        app.setStyleSheet(styles.stylesheet(theme))
        for row in range(self.table.rowCount()):
            self._paint_state(row)

    # ---------- очередь файлов ----------

    def add_files(self, paths: list[Path]) -> None:
        """Добавляет файлы в очередь, пропуская уже добавленные."""
        if self._is_running():
            return
        existing = {self._row_path(r) for r in range(self.table.rowCount())}
        added = 0
        for p in paths:
            p = p.resolve()
            if p in existing:
                continue
            existing.add(p)
            self._append_row(p)
            added += 1
        self.statusBar().showMessage(f"Добавлено файлов: {added}", 4000)
        self._update_buttons()

    def _append_row(self, path: Path) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        name = QTableWidgetItem(path.name)
        name.setData(ROLE_PATH, str(path))
        name.setToolTip(str(path))
        self.table.setItem(row, COL_FILE, name)
        for col in (COL_DURATION, COL_SPEED):
            item = QTableWidgetItem("")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, col, item)
        self.table.setItem(row, COL_STATUS, QTableWidgetItem(""))
        self._set_state(row, State.WAITING)

    def _row_path(self, row: int) -> Path:
        return Path(self.table.item(row, COL_FILE).data(ROLE_PATH))

    def _row_state(self, row: int) -> State:
        return State(self.table.item(row, COL_STATUS).data(ROLE_STATE))

    def _set_state(self, row: int, state: State, text: str | None = None) -> None:
        item = self.table.item(row, COL_STATUS)
        item.setData(ROLE_STATE, state.value)
        item.setText(text or state.value)
        self._paint_state(row)

    def _paint_state(self, row: int) -> None:
        item = self.table.item(row, COL_STATUS)
        key = STATE_COLOR.get(self._row_state(row))
        item.setForeground(QColor(styles.color(self._theme, key or "text")))

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self._update_buttons()

    def _clear(self) -> None:
        self.table.setRowCount(0)
        self.progress.setValue(0)
        self.progress_label.setText("Добавьте файлы и нажмите «Начать»")
        self._update_buttons()

    def _browse_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(engine.MEDIA_EXTENSIONS))
        start = str(self._settings.value("last_dir", ""))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите файлы лекций", start, f"Аудио и видео ({exts});;Все файлы (*)")
        if files:
            self._settings.setValue("last_dir", str(Path(files[0]).parent))
            self.add_files([Path(f) for f in files])

    def _browse_dir(self) -> None:
        start = str(self._settings.value("last_dir", ""))
        folder = QFileDialog.getExistingDirectory(self, "Папка с лекциями", start)
        if folder:
            self._settings.setValue("last_dir", folder)
            files = media_files_from([Path(folder)])
            if not files:
                self.statusBar().showMessage("В папке нет аудио или видео", 4000)
            self.add_files(files)

    def _browse_out_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Куда сохранять расшифровки",
                                                  self.out_edit.text())
        if folder:
            self.out_edit.setText(folder)

    # ---------- перетаскивание на всё окно ----------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if not self._is_running() and accepts(paths_from_event(event)):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        files = media_files_from(paths_from_event(event))
        if files:
            event.acceptProposedAction()
            self.add_files(files)

    # ---------- запуск и остановка ----------

    def _is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _start(self) -> None:
        if not self._selected_formats():
            QMessageBox.warning(self, APP_TITLE, "Выберите хотя бы один формат результата.")
            return
        jobs = [(r, self._row_path(r)) for r in range(self.table.rowCount())
                if self._row_state(r) != State.DONE]
        if not jobs:
            QMessageBox.information(self, APP_TITLE, "Все файлы уже расшифрованы.")
            return
        opts = self._options()
        if opts.device == "cpu" and not self._confirm_cpu():
            return
        for r, _ in jobs:
            self._set_state(r, State.WAITING)
        self._save_settings()
        self._queue_total, self._queue_index = len(jobs), 0
        self.progress.setValue(0)
        self._log(f"--- Старт: {len(jobs)} файл(ов), модель {opts.model}, язык {opts.lang}")

        w = TranscribeWorker(jobs, opts, self.overwrite_check.isChecked(), self)
        w.log.connect(self._log)
        w.fatal.connect(self._on_fatal)
        w.file_started.connect(self._on_file_started)
        w.file_status.connect(self._on_file_status)
        w.file_duration.connect(self._on_file_duration)
        w.file_progress.connect(self._on_file_progress)
        w.segment.connect(self._on_segment)
        w.file_done.connect(self._on_file_done)
        w.file_skipped.connect(self._on_file_skipped)
        w.file_failed.connect(self._on_file_failed)
        w.file_cancelled.connect(lambda r: self._set_state(r, State.STOPPED))
        w.finished.connect(self._on_finished)
        self._worker = w
        self.progress_label.setText("Подготовка: проверка GPU и загрузка модели …")
        self.progress.setRange(0, 0)  # «бегущая» полоса, пока грузится модель
        w.start()
        self._update_buttons()

    def _confirm_cpu(self) -> bool:
        answer = QMessageBox.question(
            self, APP_TITLE,
            "Выбран CPU. Расшифровка будет в десятки раз медленнее, чем на GPU: "
            "часовая лекция может идти несколько часов.\n\nПродолжить на CPU?")
        return answer == QMessageBox.StandardButton.Yes

    def _stop(self) -> None:
        if self._is_running():
            self._worker.stop()
            self.btn_stop.setEnabled(False)
            self.progress_label.setText("Останавливаю после текущего фрагмента …")

    # ---------- сигналы от потока ----------

    def _on_fatal(self, msg: str) -> None:
        self._log(f"[ОШИБКА] {msg}")
        QMessageBox.critical(self, APP_TITLE, msg)

    def _on_file_started(self, row: int) -> None:
        self._queue_index += 1
        self._file_t0 = time.monotonic()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self._set_state(row, State.RUNNING)
        self.table.scrollToItem(self.table.item(row, COL_FILE))
        self._log(f"[{self._queue_index}/{self._queue_total}] {self._row_path(row)}")

    def _on_file_status(self, row: int, text: str) -> None:
        self.progress_label.setText(self._label_prefix(row) + text)

    def _on_file_duration(self, row: int, duration: float, lang: str) -> None:
        self.table.item(row, COL_DURATION).setText(fmt_clock(duration))
        if self.lang_combo.currentData() == "auto":
            self._log(f"Язык: {lang}")

    def _on_file_progress(self, row: int, done: float, total: float) -> None:
        frac = done / total if total > 0 else 0.0
        self.progress.setValue(int(frac * 1000))
        self._set_state(row, State.RUNNING, f"В работе {frac:.0%}")
        elapsed = time.monotonic() - self._file_t0
        speed = done / elapsed if elapsed > 0 else 0.0
        eta = elapsed * (1 - frac) / frac if frac > 0.01 else None
        parts = [f"{fmt_clock(done)} / {fmt_clock(total)}", f"x{speed:.1f}"]
        if eta is not None:
            parts.append(f"осталось ~{fmt_clock(eta)}")
        self.progress_label.setText(self._label_prefix(row) + "  ·  ".join(parts))

    def _on_segment(self, row: int, start: float, text: str) -> None:
        self._log(f"[{fmt_clock(start)}] {text}")

    def _on_file_done(self, row: int, r: FileResult) -> None:
        self._set_state(row, State.DONE)
        self.table.item(row, COL_DURATION).setText(fmt_clock(r.duration))
        self.table.item(row, COL_SPEED).setText(f"x{r.speed:.1f}")
        self.table.item(row, COL_FILE).setData(ROLE_OUTPUTS, [str(p) for p in r.outputs])
        self._last_output_dir = r.outputs[0].parent if r.outputs else None
        self._log(f"Готово: аудио {fmt_clock(r.duration)}, обработка {fmt_clock(r.elapsed)}, "
                  f"скорость x{r.speed:.1f}")
        if self.speakers_check.isChecked():
            self._log("Спикеры: " + (", ".join(r.speakers) if r.speakers
                                     else "найден один голос, подписи не нужны"))
        for p in r.outputs:
            self._log(f"  -> {p}")
        self._update_buttons()

    def _on_file_skipped(self, row: int) -> None:
        self._queue_index += 1
        self._set_state(row, State.SKIPPED)
        self._log(f"Пропущен, уже расшифрован: {self._row_path(row)}")

    def _on_file_failed(self, row: int, msg: str) -> None:
        self._set_state(row, State.FAILED)
        self.table.item(row, COL_STATUS).setToolTip(msg)
        self._log(f"[ОШИБКА] {self._row_path(row).name}: {msg}")
        self._set_log_visible(True)

    def _on_finished(self) -> None:
        self.progress.setRange(0, 1000)
        counts = {s: 0 for s in State}
        for r in range(self.table.rowCount()):
            counts[self._row_state(r)] += 1
        summary = (f"Готово: {counts[State.DONE]}, ошибок: {counts[State.FAILED]}, "
                   f"остановлено: {counts[State.STOPPED]}, пропущено: {counts[State.SKIPPED]}")
        self.progress_label.setText(summary)
        self._log(f"--- {summary}")
        self._worker = None
        self._update_buttons()
        QApplication.alert(self)  # мигнуть в панели задач, если окно свёрнуто

    def _label_prefix(self, row: int) -> str:
        name = self._row_path(row).name
        fm = QFontMetrics(self.progress_label.font())
        name = fm.elidedText(name, Qt.TextElideMode.ElideMiddle, max(200, self.width() // 3))
        return f"{self._queue_index}/{self._queue_total}  ·  {name}  ·  "

    # ---------- результат ----------

    def _row_outputs(self, row: int) -> list[Path]:
        data = self.table.item(row, COL_FILE).data(ROLE_OUTPUTS) or []
        return [Path(p) for p in data]

    def _open_output_dir(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        outputs = self._row_outputs(rows[0]) if rows else []
        if outputs and outputs[0].exists():
            subprocess.Popen(["explorer", "/select,", str(outputs[0])])
            return
        folder = self._last_output_dir
        if folder is None and rows:
            folder = self._options().out_dir or self._row_path(rows[0]).parent
        if folder and folder.exists():
            os.startfile(folder)

    def _open_row_result(self, item: QTableWidgetItem) -> None:
        outputs = self._row_outputs(item.row())
        preferred = [p for p in outputs if p.suffix == ".txt"] or outputs
        if preferred and preferred[0].exists():
            os.startfile(preferred[0])

    def _open_speed_log(self) -> None:
        if engine.LOG_PATH.exists():
            os.startfile(engine.LOG_PATH)
        else:
            self.statusBar().showMessage("Лог пока пуст: ещё ничего не расшифровано", 4000)

    # ---------- журнал, GPU, проверка окружения ----------

    def _log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _set_log_visible(self, visible: bool) -> None:
        for w in (self.btn_log, self.act_log):
            w.blockSignals(True)
            w.setChecked(visible)
            w.blockSignals(False)
        total = sum(self.splitter.sizes())
        # таблица не должна сжиматься ниже своего минимума: на маленьком окне журнал уже
        room = max(0, total - self.table.minimumHeight())
        log_h = min(max(total // 3, LOG_MIN_HEIGHT), room) if visible else 0
        self.splitter.setSizes([total - log_h, log_h])
        if visible:
            bar = self.log_view.verticalScrollBar()
            bar.setValue(bar.maximum())

    def _probe_gpu(self) -> None:
        self._probe = GpuProbe(self)
        self._probe.found.connect(self._on_gpu_found)
        self._probe.start()

    def _on_gpu_found(self, text: str, ok: bool) -> None:
        self._gpu_label.setText(text)
        if not ok or "не найдена" in text:
            self._gpu_label.setStyleSheet(f"color: {styles.color(self._theme, 'danger')};")

    def _on_device_changed(self) -> None:
        if self.device_combo.currentData() == "cpu":
            self.statusBar().showMessage("CPU: очень медленно, только для проверки", 6000)

    def _check_env(self) -> None:
        if self._env_proc is not None:
            return
        self._set_log_visible(True)
        self._log("--- Проверка окружения (check_env.py) ...")
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setWorkingDirectory(str(PROJECT_DIR))
        proc.readyReadStandardOutput.connect(
            lambda: self._log(bytes(proc.readAllStandardOutput()).decode("utf-8", "replace")
                              .rstrip()))
        proc.finished.connect(self._on_check_finished)
        device = self.device_combo.currentData()
        proc.start(sys.executable, [str(PROJECT_DIR / "check_env.py"), "--device", device])
        self._env_proc = proc

    def _on_check_finished(self, code: int, _status) -> None:
        self._log(f"--- Проверка завершена, код {code}")
        self._env_proc = None

    # ---------- состояние кнопок и закрытие ----------

    def _update_buttons(self) -> None:
        running = self._is_running()
        has_rows = self.table.rowCount() > 0
        has_sel = bool(self.table.selectedIndexes())
        self.btn_start.setEnabled(not running and has_rows)
        self.btn_stop.setEnabled(running)
        self.btn_add.setEnabled(not running)
        self.act_add.setEnabled(not running)
        self.act_add_dir.setEnabled(not running)
        self.btn_remove.setEnabled(not running and has_sel)
        self.btn_clear.setEnabled(not running and has_rows)
        self.settings_box.setEnabled(not running)
        self.drop_zone.setEnabled(not running)
        self.btn_open.setEnabled(self._last_output_dir is not None or has_sel)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._is_running():
            answer = QMessageBox.question(
                self, APP_TITLE, "Идёт расшифровка. Остановить и выйти?\n"
                                 "Сделанная часть сохранится в *.partial.*")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.stop()
            self.progress_label.setText("Останавливаю …")
            QApplication.processEvents()
            self._worker.wait(STOP_WAIT_MS)
        self._save_settings()
        event.accept()
