"""Проверка окружения для транскрибатора.

Запускать первым делом на целевом ноутбуке:
    python check_env.py                 # полная проверка на GPU (модель tiny)
    python check_env.py --model large-v3-turbo
    python check_env.py --device cpu    # только для проверки логики без NVIDIA

Код возврата: 0, если всё в порядке; 1, если GPU-путь не работает.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import gpu_env

# Без режима разработчика Windows не даёт HF делать симлинки; кэш всё равно работает.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"

problems: list[str] = []


def report(status: str, text: str) -> None:
    print(f"{status} {text}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_python() -> None:
    section("Python")
    bits = struct.calcsize("P") * 8
    report(OK, f"Python {platform.python_version()} ({bits}-bit), {sys.executable}")
    if bits != 64:
        report(FAIL, "Нужен 64-битный Python")
        problems.append("Установите 64-битный Python 3.10–3.13 с python.org.")
    if sys.version_info < (3, 9):
        report(FAIL, "Python слишком старый")
        problems.append("Нужен Python 3.10 или новее.")


def check_ffmpeg() -> str | None:
    section("ffmpeg")
    path = shutil.which("ffmpeg")
    if not path:
        report(WARN, "ffmpeg не найден в PATH")
        problems.append("Установите ffmpeg: winget install Gyan.FFmpeg (потом перезапустите консоль).")
        return None
    out = subprocess.run([path, "-hide_banner", "-version"], capture_output=True, text=True)
    report(OK, out.stdout.splitlines()[0] if out.stdout else path)
    return path


def check_gpu() -> bool:
    """True, если видна хотя бы одна NVIDIA GPU."""
    section("NVIDIA GPU (nvidia-smi)")
    gpus = gpu_env.query_gpus()
    if gpus is None:
        report(FAIL, "nvidia-smi не найден или не отвечает")
        problems.append(
            "Не найден драйвер NVIDIA. Поставьте свежий драйвер (GeForce Game Ready / Studio) "
            "с nvidia.com и перезагрузите ноутбук."
        )
        return False
    if not gpus:
        report(FAIL, "nvidia-smi не видит ни одной карты")
        return False

    cuda = gpu_env.driver_cuda_version()
    cuda_str = f"{cuda[0]}.{cuda[1]}" if cuda else "?"
    for i, g in enumerate(gpus):
        cap = f", compute capability {g.compute_cap}" if g.compute_cap else ""
        report(OK, f"GPU {i}: {g.name}, драйвер {g.driver}, CUDA (драйвер) {cuda_str}{cap}")
        report(OK, f"       VRAM: {g.vram_total_mb} МБ всего, {g.vram_used_mb} МБ занято")
        if g.vram_total_mb < 6000:
            report(WARN, "       Меньше 6 ГБ VRAM: large-v3 может не влезть, берите large-v3-turbo/medium")
        if g.is_blackwell or "RTX 50" in g.name:
            if cuda is None or cuda < gpu_env.BLACKWELL_MIN_CUDA:
                report(FAIL, "       Карта Blackwell, а драйвер ниже CUDA 12.8")
                problems.append("Обновите драйвер NVIDIA: для RTX 50xx нужна поддержка CUDA 12.8+.")
            else:
                report(OK, "       Blackwell + драйвер с CUDA 12.8+")
    return True


def check_ctranslate2(device: str) -> bool:
    section("CTranslate2 / faster-whisper")
    added = gpu_env.add_cuda_dll_dirs()
    for d in added:
        report(OK, f"Папка DLL CUDA: {d}")
    if device == "cuda" and not added and sys.platform == "win32":
        report(WARN, "Не найдены pip-пакеты nvidia-cublas-cu12 / nvidia-cudnn-cu12")

    try:
        import ctranslate2
        import faster_whisper
    except ImportError as e:
        report(FAIL, f"Не импортируется: {e}")
        problems.append("Установите зависимости: install.bat (или pip install -r requirements.txt).")
        return False

    report(OK, f"ctranslate2 {ctranslate2.__version__}, faster-whisper {faster_whisper.__version__}")
    if device != "cuda":
        return True

    n = ctranslate2.get_cuda_device_count()
    if n == 0:
        report(FAIL, "CTranslate2 не видит CUDA-устройств")
        problems.append(
            "CTranslate2 не видит GPU. Проверьте драйвер; если nvidia-smi работает, "
            "попробуйте pip install -U ctranslate2 faster-whisper."
        )
        return False
    report(OK, f"CUDA-устройств для CTranslate2: {n}")
    try:
        types = sorted(ctranslate2.get_supported_compute_types("cuda"))
        report(OK, f"Типы вычислений на GPU: {', '.join(types)}")
        if "float16" not in types:
            report(WARN, "float16 не поддерживается, будет медленнее")
    except Exception as e:
        report(WARN, f"Не удалось получить типы вычислений: {e}")
    return True


def make_test_audio(ffmpeg: str | None) -> str | None:
    """3 секунды тона через ffmpeg: заодно проверяем, что ffmpeg пишет файлы."""
    if not ffmpeg:
        return None
    path = Path(tempfile.gettempdir()) / "lecture_transcriber_check.wav"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
           "-ar", "16000", "-ac", "1", str(path)]
    return str(path) if subprocess.run(cmd).returncode == 0 else None


def check_model(model_name: str, device: str, ffmpeg: str | None) -> bool:
    section(f"Пробная расшифровка: {model_name} на {device}")
    if device == "cpu":
        report(WARN, "Режим CPU: только для проверки логики, для лекций будет очень медленно")

    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    audio = make_test_audio(ffmpeg)
    if audio is None:
        import numpy as np
        t = np.arange(16000 * 3) / 16000
        audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        report(WARN, "Тестовый звук сгенерирован без ffmpeg")

    try:
        t0 = time.perf_counter()
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        t1 = time.perf_counter()
        report(OK, f"Модель загружена за {t1 - t0:.1f} с ({compute_type})")

        segments, info = model.transcribe(audio, language="ru", beam_size=1, vad_filter=False)
        list(segments)  # сегменты ленивые: реальный расчёт на GPU идёт здесь
        t2 = time.perf_counter()
        report(OK, f"Расшифровка {info.duration:.1f} с звука заняла {t2 - t1:.1f} с")
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        report(FAIL, msg)
        traceback.print_exc()
        problems.append(gpu_env.explain_error(msg))
        return False

    if device == "cuda":
        gpus = gpu_env.query_gpus() or []
        for g in gpus:
            report(OK, f"VRAM с загруженной моделью: {g.vram_used_mb}/{g.vram_total_mb} МБ ({g.name})")
    return True


def check_diarization() -> None:
    """Разметка спикеров необязательна: проблемы здесь — предупреждения, не ошибки."""
    section("Разметка спикеров (sherpa-onnx)")
    try:
        import sherpa_onnx
    except ImportError:
        report(WARN, "sherpa-onnx не установлен: разметка спикеров недоступна")
        return
    import diarize
    report(OK, f"sherpa-onnx {sherpa_onnx.__version__}")
    if diarize.models_ready():
        report(OK, f"Модель голосов на месте: {diarize.EMBEDDING_MODEL.name}")
    else:
        report(WARN, "Модель голосов ещё не скачана (25 МБ, скачается при первом использовании)")


def check_torch() -> None:
    """Запасной путь: PyTorch cu128. Проверяем, только если он установлен."""
    try:
        import torch
    except ImportError:
        return
    section("PyTorch (запасной путь)")
    report(OK, f"torch {torch.__version__}, CUDA сборки {torch.version.cuda}")
    if not torch.cuda.is_available():
        report(WARN, "torch.cuda.is_available() = False")
        return
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arches = torch.cuda.get_arch_list()
    report(OK, f"{name}, sm_{cap[0]}{cap[1]}")
    if f"sm_{cap[0]}{cap[1]}" not in arches:
        report(FAIL, f"Эта сборка torch не умеет sm_{cap[0]}{cap[1]} (есть: {', '.join(arches)})")
        problems.append("Переустановите torch из индекса cu128: install.bat fallback.")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Проверка окружения транскрибатора")
    ap.add_argument("--model", default="tiny",
                    help="модель для пробного прогона (по умолчанию tiny, быстро качается)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--skip-model", action="store_true", help="не загружать модель")
    args = ap.parse_args()

    check_python()
    ffmpeg = check_ffmpeg()
    gpu_seen = check_gpu() if args.device == "cuda" else True
    ct2_ok = check_ctranslate2(args.device)
    model_ok = True
    if ct2_ok and gpu_seen and not args.skip_model:
        model_ok = check_model(args.model, args.device, ffmpeg)
    check_diarization()
    check_torch()

    section("Итог")
    ok = gpu_seen and ct2_ok and model_ok
    if ok and args.device == "cuda":
        report(OK, "GPU-путь работает, можно расшифровывать.")
    elif ok:
        report(WARN, "Проверка на CPU прошла. На целевом ноутбуке запустите без --device cpu.")
    elif args.device == "cuda":
        report(FAIL, "GPU-путь не работает.")
    else:
        report(FAIL, "Проверка на CPU не прошла.")
    for p in dict.fromkeys(problems):
        print(f"  - {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
