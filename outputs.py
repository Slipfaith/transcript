"""Запись расшифровки в .txt / .srt / .md по мере поступления сегментов.

Файлы пишутся потоково во временные `*.part`, так что даже для трёхчасовой
лекции в памяти ничего не копится. При успехе `.part` переименовываются в
итоговые файлы, при сбое — в `*.partial.*`, чтобы сделанное не пропало.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

FORMATS = ("txt", "srt", "md")

SENTENCE_END = (".", "!", "?", "…", "»", '"')


def fmt_srt_time(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_clock(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


@dataclass
class TranscriptMeta:
    source: Path
    duration: float
    model: str
    language: str
    speakers: list[str] | None = None  # имена, если размечены спикеры


class TranscriptWriter:
    def __init__(self, base: Path, formats: list[str], meta: TranscriptMeta,
                 paragraph_gap: float = 2.0, paragraph_max_chars: int = 900):
        """base — путь без расширения: base.txt, base.srt, base.md."""
        self.base = base
        self.formats = formats
        self.paragraph_gap = paragraph_gap
        self.paragraph_max_chars = paragraph_max_chars

        self._files: dict[str, TextIO] = {
            f: open(self._part_path(f), "w", encoding="utf-8", newline="\n") for f in formats
        }
        self._srt_index = 0
        self._prev_end: float | None = None
        self._prev_text: str | None = None
        self._para_chars = 0
        self._speaker: str | None = None

        if "md" in self._files:
            speakers = f"- Спикеры: {', '.join(meta.speakers)}\n" if meta.speakers else ""
            self._files["md"].write(
                f"# {meta.source.stem}\n\n"
                f"- Файл: `{meta.source.name}`\n"
                f"- Длительность: {fmt_clock(meta.duration)}\n"
                f"- Модель: {meta.model}, язык: {meta.language}\n"
                f"{speakers}\n"
            )

    def _part_path(self, fmt: str) -> Path:
        return self.base.with_name(f"{self.base.name}.{fmt}.part")

    def final_path(self, fmt: str) -> Path:
        return self.base.with_name(f"{self.base.name}.{fmt}")

    def _new_paragraph(self, start: float) -> bool:
        if self._prev_end is None:
            return False
        if start - self._prev_end >= self.paragraph_gap:
            return True
        # слишком длинный абзац режем на конце предложения
        return (self._para_chars >= self.paragraph_max_chars
                and self._prev_text is not None and self._prev_text.endswith(SENTENCE_END))

    def add(self, start: float, end: float, text: str, speaker: str | None = None) -> None:
        """speaker — имя спикера; при его смене начинается новый абзац с подписью."""
        text = text.strip()
        if not text:
            return
        new_speaker = speaker is not None and speaker != self._speaker
        new_para = new_speaker or self._new_paragraph(start)
        if new_para:
            self._para_chars = 0
        if speaker is not None:
            self._speaker = speaker
        label = f"{speaker}: " if new_speaker else ""

        if "txt" in self._files:
            f = self._files["txt"]
            if self._prev_end is not None:
                f.write("\n\n" if new_para else " ")
            f.write(label + text)

        if "srt" in self._files:
            self._srt_index += 1
            self._files["srt"].write(
                f"{self._srt_index}\n{fmt_srt_time(start)} --> {fmt_srt_time(end)}\n"
                f"{label}{text}\n\n"
            )

        if "md" in self._files:
            f = self._files["md"]
            if new_para and self._prev_end is not None:
                f.write("\n")
            md_label = f"**{speaker}:** " if new_speaker else ""
            f.write(f"[{fmt_clock(start)}] {md_label}{text}\n")

        self._prev_end = end
        self._prev_text = text
        self._para_chars += len(text) + 1

    def close(self, ok: bool) -> list[Path]:
        """Закрывает файлы. Возвращает итоговые пути (или *.partial.* при сбое)."""
        result = []
        for fmt, f in self._files.items():
            if fmt == "txt" and self._prev_end is not None:
                f.write("\n")
            f.close()
            target = self.final_path(fmt) if ok else self.base.with_name(f"{self.base.name}.partial.{fmt}")
            self._part_path(fmt).replace(target)
            result.append(target)
        return result
