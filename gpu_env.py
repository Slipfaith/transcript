"""Подготовка окружения CUDA на Windows и сведения о GPU.

CTranslate2 (движок faster-whisper) на Windows ищет cuBLAS и cuDNN как обычные
DLL. Их ставим pip-пакетами nvidia-cublas-cu12 / nvidia-cudnn-cu12, а здесь
добавляем их папки bin в путь поиска DLL. Если установлен PyTorch с CUDA,
его папка lib тоже подходит: там лежат те же библиотеки.

Модуль нужно импортировать и вызвать add_cuda_dll_dirs() ДО импорта
ctranslate2 / faster_whisper.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import site
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Blackwell (RTX 50xx) = compute capability 12.x, нужна CUDA 12.8+
BLACKWELL_MIN_CUDA = (12, 8)


def _site_dirs() -> list[Path]:
    dirs = [Path(p) for p in site.getsitepackages()]
    try:
        dirs.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    return [d for d in dirs if d.is_dir()]


def add_cuda_dll_dirs() -> list[str]:
    """Добавляет папки с DLL cuBLAS/cuDNN в поиск. Возвращает добавленные пути."""
    if sys.platform != "win32":
        return []

    candidates: list[Path] = []
    for root in _site_dirs():
        nvidia = root / "nvidia"
        if nvidia.is_dir():
            candidates += [p / "bin" for p in nvidia.iterdir() if (p / "bin").is_dir()]

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec and torch_spec.origin:
        torch_lib = Path(torch_spec.origin).parent / "lib"
        if torch_lib.is_dir():
            candidates.append(torch_lib)

    added: list[str] = []
    for path in candidates:
        s = str(path)
        if s in added:
            continue
        os.add_dll_directory(s)
        added.append(s)

    # Часть DLL грузится через LoadLibrary без флагов и смотрит только в PATH.
    if added:
        os.environ["PATH"] = os.pathsep.join(added + [os.environ.get("PATH", "")])
    return added


@dataclass
class GpuInfo:
    name: str
    driver: str
    vram_total_mb: int
    vram_used_mb: int
    compute_cap: str | None  # например "12.0"; None, если драйвер не отдаёт

    @property
    def is_blackwell(self) -> bool:
        return bool(self.compute_cap and self.compute_cap.split(".")[0] == "12")


def find_nvidia_smi() -> str | None:
    path = shutil.which("nvidia-smi")
    if path:
        return path
    fallback = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
    return str(fallback) if fallback.is_file() else None


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def query_gpus() -> list[GpuInfo] | None:
    """Список NVIDIA GPU по nvidia-smi; None, если nvidia-smi недоступен."""
    smi = find_nvidia_smi()
    if not smi:
        return None

    base = "name,driver_version,memory.total,memory.used"
    out = _run([smi, f"--query-gpu={base},compute_cap", "--format=csv,noheader,nounits"])
    with_cap = out is not None
    if not with_cap:  # старые драйверы не знают compute_cap
        out = _run([smi, f"--query-gpu={base}", "--format=csv,noheader,nounits"])
    if out is None:
        return None

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        gpus.append(GpuInfo(
            name=parts[0],
            driver=parts[1],
            vram_total_mb=int(float(parts[2])),
            vram_used_mb=int(float(parts[3])),
            compute_cap=parts[4] if with_cap and len(parts) > 4 else None,
        ))
    return gpus


def explain_error(msg: str) -> str:
    """Подсказка, что чинить, по тексту ошибки CTranslate2 / CUDA."""
    low = msg.lower()
    if "cublas" in low or "cudnn" in low or ".dll" in low:
        return ("Не найдены библиотеки CUDA (cuBLAS/cuDNN). Выполните "
                "pip install -r requirements.txt в venv, затем повторите проверку.")
    if ("no kernel image" in low or "invalid device function" in low
            or "sm_120" in low or "unsupported gpu" in low):
        return ("Эта сборка CTranslate2 не поддерживает Blackwell (sm_120). "
                "Попробуйте pip install -U ctranslate2; если не поможет, "
                "запасной путь: install.bat fallback (PyTorch cu128).")
    if "out of memory" in low:
        return "Не хватило VRAM: закройте другие программы или возьмите модель поменьше."
    return "Неизвестная ошибка, запустите check_env.py и пришлите полный вывод."


def driver_cuda_version() -> tuple[int, int] | None:
    """Максимальная версия CUDA, которую поддерживает драйвер (из шапки nvidia-smi)."""
    smi = find_nvidia_smi()
    out = _run([smi]) if smi else None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out or "")
    return (int(m.group(1)), int(m.group(2))) if m else None
