"""Ядро расшифровки: общее для CLI (transcribe.py) и GUI (app.py).

Ничего не печатает само: о ходе работы сообщает через колбэки, чтобы CLI
мог рисовать tqdm, а GUI — свои прогресс-бары.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import gpu_env
from outputs import FORMATS, TranscriptMeta, TranscriptWriter

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODELS = ["large-v3-turbo", "large-v3", "medium", "small", "base", "tiny"]
DEFAULT_MODEL = "large-v3-turbo"
MEDIA_EXTENSIONS = {
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".amr",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".m4v", ".mpg", ".mpeg", ".3gp",
}
LOG_PATH = Path(__file__).resolve().parent / "logs" / "transcribe.log"
_POLL_SECONDS = 0.2


class FatalError(Exception):
    """Ошибка, после которой нет смысла продолжать (нет GPU, не грузится модель)."""


class Cancelled(Exception):
    """Пользователь остановил расшифровку."""


@dataclass
class Options:
    model: str = DEFAULT_MODEL
    lang: str = "ru"                    # "auto" — автоопределение
    device: str = "cuda"                # "cpu" только явно
    compute_type: str | None = None     # None: float16 на GPU, int8 на CPU
    out_dir: Path | None = None         # None: рядом с исходником
    formats: list[str] = field(default_factory=lambda: list(FORMATS))
    beam_size: int = 5
    prompt: str | None = None
    paragraph_gap: float = 2.0
    vad: bool = True
    normalize: bool = True
    condition_prev: bool = False
    diarize: bool = False               # разметка спикеров (лектор / вопросы из зала)
    num_speakers: int | None = None     # None — определить автоматически
    speaker_threshold: float | None = None  # None — diarize.DEFAULT_THRESHOLD

    @property
    def resolved_compute_type(self) -> str:
        return self.compute_type or ("float16" if self.device == "cuda" else "int8")


@dataclass
class FileResult:
    src: Path
    outputs: list[Path]
    duration: float      # длительность аудио, с
    elapsed: float       # всё время на файл, с
    audio_prep: float    # из них на ffmpeg, с
    language: str
    language_prob: float
    speakers: list[str] = field(default_factory=list)

    @property
    def speed(self) -> float:
        return self.duration / self.elapsed if self.elapsed > 0 else 0.0


def output_base(src: Path, opts: Options) -> Path:
    return (opts.out_dir or src.parent) / src.stem


def already_done(src: Path, opts: Options) -> bool:
    base = output_base(src, opts)
    return base.with_name(f"{base.name}.{opts.formats[0]}").exists()


def setup_device(device: str) -> list[str]:
    """Проверяет устройство и возвращает строки с описанием (для лога / статуса).

    Без рабочей CUDA бросает FatalError: молча на CPU не уходим.
    """
    if device == "cpu":
        return ["Режим CPU: часовая лекция может считаться дольше часа. "
                "Для работы используйте NVIDIA GPU."]

    gpus = gpu_env.query_gpus()
    if not gpus:
        raise FatalError("NVIDIA GPU не найдена (nvidia-smi не отвечает). "
                         "Запустите check_env.py, там будет инструкция.")
    cuda = gpu_env.driver_cuda_version()
    cuda_str = f"{cuda[0]}.{cuda[1]}" if cuda else "?"
    lines = []
    for g in gpus:
        lines.append(f"GPU: {g.name}, VRAM {g.vram_total_mb} МБ (занято {g.vram_used_mb}), "
                     f"драйвер {g.driver}, CUDA {cuda_str}")
        if g.is_blackwell and (cuda is None or cuda < gpu_env.BLACKWELL_MIN_CUDA):
            raise FatalError("Для RTX 50xx нужен драйвер с CUDA 12.8+. Обновите драйвер NVIDIA.")

    gpu_env.add_cuda_dll_dirs()
    import ctranslate2
    if ctranslate2.get_cuda_device_count() == 0:
        raise FatalError("CTranslate2 не видит GPU. Запустите check_env.py.")
    return lines


_model_cache: dict[tuple[str, str, str], object] = {}
_model_lock = threading.Lock()


def load_model(opts: Options):
    """Загружает модель; повторный вызов с теми же параметрами отдаёт уже загруженную."""
    from faster_whisper import WhisperModel

    key = (opts.model, opts.device, opts.resolved_compute_type)
    with _model_lock:
        if key in _model_cache:
            return _model_cache[key]
        _model_cache.clear()  # держим в памяти одну модель, VRAM всего 8 ГБ
        try:
            model = WhisperModel(opts.model, device=opts.device,
                                 compute_type=opts.resolved_compute_type)
        except Exception as e:
            raise FatalError(f"Модель не загрузилась: {e}\n{gpu_env.explain_error(str(e))}") from e
        _model_cache[key] = model
        return model


def prepare_audio(src: Path, tmp_dir: Path, normalize: bool,
                  cancelled: Callable[[], bool] | None = None) -> Path:
    """Извлекает первую звуковую дорожку в WAV 16 кГц моно (так Whisper и ждёт).

    Без ffmpeg отдаём файл как есть: faster-whisper декодирует его сам через PyAV.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src
    out = tmp_dir / "audio.wav"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-i", str(src), "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "16000"]
    if normalize:
        cmd += ["-af", "dynaudnorm"]
    cmd += ["-c:a", "pcm_s16le", str(out)]

    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err, creationflags=flags)
        while proc.poll() is None:
            if cancelled and cancelled():
                proc.kill()
                proc.wait()
                raise Cancelled()
            time.sleep(_POLL_SECONDS)
        if proc.returncode != 0:
            err.seek(0)
            msg = err.read().decode("utf-8", "replace").strip()
            raise RuntimeError(f"ffmpeg не смог извлечь звук: {msg or proc.returncode}")
    return out


def write_log(line: str) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{line}\n")


@dataclass
class _Hooks:
    """Колбэки transcribe_file; любой может отсутствовать."""
    status: Callable[[str], None] | None = None
    start: Callable[[float, str, float], None] | None = None
    progress: Callable[[float, float], None] | None = None
    segment: Callable[[float, float, str], None] | None = None
    cancelled: Callable[[], bool] | None = None

    def say(self, text: str) -> None:
        if self.status:
            self.status(text)

    def check_cancel(self) -> None:
        if self.cancelled and self.cancelled():
            raise Cancelled()

    def step(self, seg, duration: float) -> None:
        if self.segment:
            self.segment(seg.start, seg.end, seg.text.strip())
        if self.progress:
            self.progress(min(seg.end, duration), duration)


def transcribe_file(
    model,
    src: Path,
    opts: Options,
    *,
    on_status: Callable[[str], None] | None = None,
    on_start: Callable[[float, str, float], None] | None = None,
    on_progress: Callable[[float, float], None] | None = None,
    on_segment: Callable[[float, float, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> FileResult:
    """Расшифровывает один файл и пишет результаты.

    on_start(duration, language, language_prob) — когда известна длительность;
    on_progress(done_seconds, total_seconds) — по мере готовности сегментов;
    cancelled() — опрашивается между сегментами, True останавливает работу
    (сделанное остаётся в *.partial.*).
    """
    hooks = _Hooks(on_status, on_start, on_progress, on_segment, cancelled)
    base = output_base(src, opts)
    base.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="lecture_") as tmp:
        hooks.say("Извлекаю звук ..." if shutil.which("ffmpeg")
                  else "ffmpeg не найден, декодирую без нормализации громкости")
        audio = prepare_audio(src, Path(tmp), opts.normalize, cancelled)
        t_audio = time.perf_counter()
        embedder = _start_embedder(audio, hooks) if opts.diarize else None

        hooks.say("Распознаю речь ...")
        try:
            segments, info = model.transcribe(
                str(audio),
                language=None if opts.lang == "auto" else opts.lang,
                beam_size=opts.beam_size,
                vad_filter=opts.vad,
                initial_prompt=opts.prompt or None,
                condition_on_previous_text=opts.condition_prev,
                # границы сегментов по словам: иначе Whisper растягивает их на паузы,
                # и .srt отстаёт, а абзацы в .txt не находятся
                word_timestamps=True,
            )
        except BaseException:
            if embedder:
                embedder.abort()
            raise
        duration = info.duration
        if on_start:
            on_start(duration, info.language, info.language_probability)
        meta = TranscriptMeta(source=src, duration=duration, model=opts.model,
                              language=info.language)
        if embedder is None:
            outputs, speakers = _run_streaming(segments, base, opts, meta, hooks), []
        else:
            outputs, speakers = _run_with_speakers(segments, embedder, base, opts, meta, hooks)

    result = FileResult(
        src=src, outputs=outputs, duration=duration,
        elapsed=time.perf_counter() - t_start, audio_prep=t_audio - t_start,
        language=info.language, language_prob=info.language_probability, speakers=speakers,
    )
    spk = f"\tspeakers={len(speakers)}" if opts.diarize else ""
    write_log(f"OK\t{src}\t{opts.model}\t{opts.device}\taudio={duration:.0f}s\t"
              f"time={result.elapsed:.0f}s\tspeed=x{result.speed:.1f}{spk}")
    return result


def _run_streaming(segments, base: Path, opts: Options, meta: TranscriptMeta,
                   hooks: _Hooks) -> list[Path]:
    """Без разметки спикеров: пишем файлы прямо по ходу распознавания."""
    writer = TranscriptWriter(base, opts.formats, meta, paragraph_gap=opts.paragraph_gap)
    ok = False
    try:
        for seg in segments:  # реальная работа модели идёт здесь, по мере чтения
            hooks.check_cancel()
            writer.add(seg.start, seg.end, seg.text)
            hooks.step(seg, meta.duration)
        if hooks.progress:
            hooks.progress(meta.duration, meta.duration)
        ok = True
    finally:
        outputs = writer.close(ok)
    return outputs


def _start_embedder(audio: Path, hooks: _Hooks):
    try:
        import diarize
    except ImportError as e:
        raise RuntimeError(f"Для разметки спикеров нужен пакет sherpa-onnx: "
                           f"pip install -r requirements.txt ({e})") from e
    diarize.download_models(hooks.say)
    return diarize.SpeakerEmbedder(audio)


def _run_with_speakers(segments, embedder, base: Path, opts: Options, meta: TranscriptMeta,
                       hooks: _Hooks) -> tuple[list[Path], list[str]]:
    """С разметкой: отпечатки голоса считаются в фоне, файлы пишутся в конце.

    Текст сегментов копится в памяти (для трёх часов это единицы мегабайт).
    При остановке или сбое сделанное сохраняется без подписей спикеров.
    """
    collected: list[tuple[float, float, str]] = []
    try:
        for seg in segments:
            hooks.check_cancel()
            embedder.add(len(collected), seg.start, seg.end)
            collected.append((seg.start, seg.end, seg.text))
            hooks.step(seg, meta.duration)
        if hooks.progress:
            hooks.progress(meta.duration, meta.duration)
        speakers = _wait_speakers(embedder, collected, opts, hooks)
    except BaseException:
        embedder.abort()
        _write_all(base, opts, meta, collected, None, ok=False)
        raise
    names = list(dict.fromkeys(s for s in speakers if s))
    meta.speakers = names or None
    return _write_all(base, opts, meta, collected, speakers, ok=True), names


def _wait_speakers(embedder, collected: list[tuple[float, float, str]], opts: Options,
                   hooks: _Hooks) -> list[str | None]:
    import diarize

    while (left := embedder.pending()) > 0:
        hooks.check_cancel()
        hooks.say(f"Разметка спикеров: осталось фрагментов {left}")
        time.sleep(_POLL_SECONDS)
    hooks.say("Разметка спикеров ...")
    embeddings = embedder.finish()
    threshold = opts.speaker_threshold or diarize.DEFAULT_THRESHOLD
    durations = [e - s for s, e, _ in collected]
    clusters = diarize.cluster(embeddings, durations, opts.num_speakers, threshold)
    return diarize.assign_speakers(durations, clusters)


def _write_all(base: Path, opts: Options, meta: TranscriptMeta,
               collected: list[tuple[float, float, str]],
               speakers: list[str | None] | None, ok: bool) -> list[Path]:
    writer = TranscriptWriter(base, opts.formats, meta, paragraph_gap=opts.paragraph_gap)
    try:
        for i, (start, end, text) in enumerate(collected):
            writer.add(start, end, text, speakers[i] if speakers else None)
    finally:
        outputs = writer.close(ok)
    return outputs
