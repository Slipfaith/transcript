"""Фоновые потоки: расшифровка очереди файлов и опрос GPU при старте."""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

import engine
import gpu_env
from engine import Cancelled, FatalError, Options


class TranscribeWorker(QThread):
    """Расшифровывает файлы по очереди. Строки таблицы передаются как row-индексы."""

    log = Signal(str)
    file_started = Signal(int)
    file_status = Signal(int, str)
    file_duration = Signal(int, float, str)
    file_progress = Signal(int, float, float)
    segment = Signal(int, float, str)
    file_done = Signal(int, object)        # FileResult
    file_skipped = Signal(int)
    file_failed = Signal(int, str)
    file_cancelled = Signal(int)
    fatal = Signal(str)

    def __init__(self, jobs: list[tuple[int, Path]], opts: Options, overwrite: bool,
                 parent=None) -> None:
        super().__init__(parent)
        self._jobs = jobs
        self._opts = opts
        self._overwrite = overwrite
        self._stop = threading.Event()

    def stop(self) -> None:
        """Просит остановиться; сработает на ближайшем сегменте."""
        self._stop.set()

    def run(self) -> None:
        model = self._prepare()
        if model is None:
            return
        for row, src in self._jobs:
            if self._stop.is_set():
                break
            if not self._overwrite and engine.already_done(src, self._opts):
                self.file_skipped.emit(row)
                continue
            if not self._run_one(model, row, src):
                break

    def _prepare(self):
        """Проверка устройства и загрузка модели. None, если дальше идти нельзя."""
        opts = self._opts
        try:
            for line in engine.setup_device(opts.device):
                self.log.emit(line)
            self.log.emit(f"Загружаю модель {opts.model} "
                          f"({opts.device}, {opts.resolved_compute_type}) ...")
            model = engine.load_model(opts)
        except FatalError as e:
            self.fatal.emit(str(e))
            return None
        except Exception as e:  # неожиданное: показываем как есть
            self.fatal.emit(f"{type(e).__name__}: {e}\n{gpu_env.explain_error(str(e))}")
            return None
        self.log.emit("Модель готова")
        return model

    def _run_one(self, model, row: int, src: Path) -> bool:
        """Один файл. False — остановлено пользователем, очередь прерывается."""
        opts = self._opts
        self.file_started.emit(row)
        try:
            result = engine.transcribe_file(
                model, src, opts,
                on_status=lambda s: self.file_status.emit(row, s),
                on_start=lambda d, lang, p: self.file_duration.emit(row, d, lang),
                on_progress=lambda done, total: self.file_progress.emit(row, done, total),
                on_segment=lambda start, end, text: self.segment.emit(row, start, text),
                cancelled=self._stop.is_set,
            )
        except Cancelled:
            engine.write_log(f"STOPPED\t{src}\t{opts.model}\t{opts.device}")
            self.file_cancelled.emit(row)
            return False
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            engine.write_log(f"FAIL\t{src}\t{opts.model}\t{opts.device}\t{msg}")
            self.file_failed.emit(row, f"{msg}\n{gpu_env.explain_error(msg)}")
            return True
        self.file_done.emit(row, result)
        return True


class GpuProbe(QThread):
    """Спрашивает nvidia-smi при старте, чтобы не тормозить появление окна."""

    found = Signal(str, bool)   # текст, всё ли в порядке

    def run(self) -> None:
        gpus = gpu_env.query_gpus()
        if not gpus:
            self.found.emit("NVIDIA GPU не найдена: расшифровка возможна только на CPU", False)
            return
        g = gpus[0]
        cuda = gpu_env.driver_cuda_version()
        cuda_str = f"{cuda[0]}.{cuda[1]}" if cuda else "?"
        ok = not (g.is_blackwell and (cuda is None or cuda < gpu_env.BLACKWELL_MIN_CUDA))
        text = f"{g.name} · VRAM {g.vram_total_mb / 1024:.0f} ГБ · драйвер {g.driver} · CUDA {cuda_str}"
        if not ok:
            text += " · нужен драйвер с CUDA 12.8+"
        self.found.emit(text, ok)
