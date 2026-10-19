"""Разметка спикеров (кто говорит) офлайн через sherpa-onnx.

Подход: на каждый сегмент Whisper считаем «отпечаток голоса» (эмбеддинг
WeSpeaker ResNet34), потом кластеризуем отпечатки. Каждая секунда звука
обрабатывается один раз, поэтому это в разы быстрее оконного конвейера pyannote,
а эмбеддинги считаются на CPU в отдельном потоке, пока GPU распознаёт речь.
PyTorch и токен Hugging Face не нужны; модель (~25 МБ) качается один раз
в models/diarization.

Самый «говорящий» голос называем «Лектор», остальные — «Слушатель 1, 2, ...».

Ограничение: спикер определяется на весь сегмент Whisper. Если вопрос и ответ
слились в один сегмент без паузы, он целиком уйдёт одному из них.
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models" / "diarization"
EMBEDDING_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                 "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx")
EMBEDDING_MODEL = MODELS_DIR / "wespeaker_en_voxceleb_resnet34_LM.onnx"

SAMPLE_RATE = 16000
# Порог косинусного расстояния для слияния голосов. Меньше — чувствительнее
# (больше спикеров). Подобран на синтетических голосах; на живой лекции стоит проверить.
DEFAULT_THRESHOLD = 0.35
MIN_EMBED_SECONDS = 0.5      # короче — отпечаток бессмысленен, спикер берётся у соседей
CLUSTER_MIN_SECONDS = 1.5    # кластеризуем только надёжные отпечатки, короткие — к ближайшему
NEAREST_MARGIN = 0.05        # насколько ближайший голос должен опережать второй
MAX_EMBED_SECONDS = 20.0     # длиннее не нужно: голос уже понятен
LECTURER = "Лектор"
LISTENER = "Слушатель"


def models_ready() -> bool:
    return EMBEDDING_MODEL.is_file()


def download_models(on_status: Callable[[str], None] | None = None) -> None:
    """Качает модель при первом использовании. Нужен интернет один раз."""
    if models_ready():
        return
    if on_status:
        on_status("Скачиваю модель голосов (25 МБ) ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=MODELS_DIR, suffix=".download")
    os.close(fd)
    try:
        urllib.request.urlretrieve(EMBEDDING_URL, tmp)
        Path(tmp).replace(EMBEDDING_MODEL)
    finally:
        Path(tmp).unlink(missing_ok=True)


class AudioSource:
    """Отдаёт куски звука по времени. WAV читается с диска, без загрузки целиком."""

    def __init__(self, path: Path) -> None:
        self._wav = None
        self._samples = None
        try:
            wav = wave.open(str(path), "rb")
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (SAMPLE_RATE, 1, 2):
                self._wav = wav
            else:
                wav.close()
        except (wave.Error, EOFError):
            pass
        if self._wav is None:  # не наш WAV (нет ffmpeg) — декодируем целиком
            from faster_whisper import decode_audio
            self._samples = decode_audio(str(path), sampling_rate=SAMPLE_RATE)

    def read(self, start: float, end: float) -> np.ndarray:
        a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        if self._samples is not None:
            return self._samples[a:b]
        self._wav.setpos(min(a, self._wav.getnframes()))
        raw = self._wav.readframes(max(0, b - a))
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()


class SpeakerEmbedder:
    """Считает эмбеддинги сегментов в фоновом потоке (CPU), пока идёт распознавание."""

    def __init__(self, audio_path: Path) -> None:
        import sherpa_onnx

        threads = max(1, (os.cpu_count() or 2) // 2)
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(EMBEDDING_MODEL), num_threads=threads))
        self._audio = AudioSource(audio_path)
        self._queue: queue.Queue = queue.Queue()
        self._embeddings: dict[int, np.ndarray] = {}
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add(self, index: int, start: float, end: float) -> None:
        """Ставит сегмент в очередь. Короткие пропускаются: их спикер берётся у соседей."""
        if end - start >= MIN_EMBED_SECONDS:
            mid = (start + end) / 2
            half = min(end - start, MAX_EMBED_SECONDS) / 2
            self._queue.put((index, mid - half, mid + half))

    def pending(self) -> int:
        return self._queue.qsize()

    def finish(self) -> dict[int, np.ndarray]:
        """Ждёт, пока посчитается всё поставленное, и отдаёт эмбеддинги."""
        self._queue.put(None)
        self._thread.join()
        self._audio.close()
        if self._error:
            raise self._error
        return self._embeddings

    def abort(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(None)
        self._thread.join()
        self._audio.close()

    def _run(self) -> None:
        try:
            while (item := self._queue.get()) is not None:
                index, start, end = item
                emb = self._embed(self._audio.read(start, end))
                if emb is not None:
                    self._embeddings[index] = emb
        except BaseException as e:  # отдадим в finish()
            self._error = e

    def _embed(self, samples: np.ndarray) -> np.ndarray | None:
        stream = self._extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            return None
        e = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        norm = np.linalg.norm(e)
        return e / norm if norm > 0 else None


def cluster(embeddings: dict[int, np.ndarray], durations: list[float],
            num_speakers: int | None, threshold: float = DEFAULT_THRESHOLD) -> dict[int, int]:
    """Номер кластера (голоса) для сегментов.

    Кластеризуются длинные сегменты; короткие относятся к ближайшему центру
    голоса, если он явно ближе остальных, иначе остаются без метки.
    """
    import sherpa_onnx

    if not embeddings:
        return {}
    reliable = sorted(k for k in embeddings if durations[k] >= CLUSTER_MIN_SECONDS)
    if len(reliable) < 2:
        reliable = sorted(embeddings)
    if len(reliable) == 1:
        return {reliable[0]: 0}

    x = np.stack([embeddings[k] for k in reliable])
    config = sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers or -1, threshold=threshold)
    result = dict(zip(reliable, sherpa_onnx.FastClustering(config)(x)))

    labels = sorted(set(result.values()))
    centroids = np.stack([x[[result[k] == c for k in reliable]].mean(axis=0) for c in labels])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    for k in embeddings.keys() - result.keys():
        sims = centroids @ embeddings[k]
        order = np.argsort(sims)[::-1]
        if len(order) == 1 or sims[order[0]] - sims[order[1]] >= NEAREST_MARGIN:
            result[k] = labels[order[0]]
    return result


def assign_speakers(durations: list[float], clusters: dict[int, int]) -> list[str | None]:
    """Имя спикера для каждого сегмента.

    Сегменты без эмбеддинга (короткие) получают спикера ближайшего предыдущего,
    а в начале записи — следующего. Если голос один, разметка не нужна: все None.
    """
    talk: dict[int, float] = {}
    first: dict[int, int] = {}
    for i, c in clusters.items():
        talk[c] = talk.get(c, 0.0) + durations[i]
        first[c] = min(first.get(c, i), i)
    if len(talk) < 2:
        return [None] * len(durations)

    lecturer = max(talk, key=talk.get)
    others = sorted((c for c in talk if c != lecturer), key=first.get)
    names = {lecturer: LECTURER}
    names.update({c: f"{LISTENER} {n}" for n, c in enumerate(others, 1)})

    result: list[str | None] = [names[clusters[i]] if i in clusters else None
                                for i in range(len(durations))]
    for i in range(1, len(result)):          # короткие — как предыдущий
        if result[i] is None:
            result[i] = result[i - 1]
    for i in range(len(result) - 2, -1, -1):  # в начале записи — как следующий
        if result[i] is None:
            result[i] = result[i + 1]
    return result
