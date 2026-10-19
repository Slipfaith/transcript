"""Расшифровка лекций на NVIDIA GPU через faster-whisper (командная строка).

Примеры:
    python transcribe.py lecture.mp4
    python transcribe.py a.m4a b.mkv --model large-v3 --lang ru
    python transcribe.py "D:\\Лекции\\*.mp3" --out-dir transcripts
    python transcribe.py test.wav --model tiny --device cpu   # без NVIDIA, только для проверки

На выходе рядом с исходником (или в --out-dir): <имя>.txt, <имя>.srt, <имя>.md.
Окно с drag-and-drop: app.py.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import engine
import gpu_env
from engine import FatalError, Options
from outputs import FORMATS, fmt_clock


def parse_args(argv: list[str] | None = None) -> tuple[list[str], Options, argparse.Namespace]:
    ap = argparse.ArgumentParser(
        description="Локальная расшифровка лекций (faster-whisper, NVIDIA GPU)")
    ap.add_argument("files", nargs="+", help="файлы лекций; можно маски вида *.mp3")
    ap.add_argument("--model", default=engine.DEFAULT_MODEL, choices=engine.MODELS,
                    help="модель Whisper (по умолчанию large-v3-turbo)")
    ap.add_argument("--lang", default="ru",
                    help="язык речи: ru, en, ... или auto (по умолчанию ru)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu только явно: это очень медленно")
    ap.add_argument("--compute-type", default=None,
                    help="по умолчанию float16 на GPU, int8 на CPU")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="куда класть результаты (по умолчанию рядом с исходником)")
    ap.add_argument("--formats", default=",".join(FORMATS),
                    help="через запятую из: txt,srt,md (по умолчанию все)")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--prompt", default=None,
                    help="подсказка с терминами и именами из лекции, улучшает их написание")
    ap.add_argument("--paragraph-gap", type=float, default=2.0,
                    help="пауза в секундах, после которой в .txt начинается новый абзац")
    ap.add_argument("--no-vad", action="store_true", help="не вырезать тишину")
    ap.add_argument("--no-normalize", action="store_true",
                    help="не выравнивать громкость через ffmpeg")
    ap.add_argument("--condition-prev", action="store_true",
                    help="учитывать предыдущий текст (связнее, но на длинных записях "
                         "иногда зацикливается)")
    ap.add_argument("--speakers", action="store_true",
                    help="разметить спикеров: «Лектор», «Слушатель 1», ...")
    ap.add_argument("--num-speakers", type=int, default=None,
                    help="сколько голосов в записи, если известно (иначе определится само)")
    ap.add_argument("--speaker-threshold", type=float, default=None,
                    help="чувствительность разметки: меньше — больше спикеров (по умолчанию 0.35)")
    ap.add_argument("--show-text", action="store_true", help="печатать текст по ходу")
    ap.add_argument("--overwrite", action="store_true",
                    help="перезаписывать уже готовые расшифровки")
    args = ap.parse_args(argv)

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    bad = [f for f in formats if f not in FORMATS]
    if bad or not formats:
        ap.error(f"неизвестные форматы: {', '.join(bad) or '(пусто)'}")

    opts = Options(
        model=args.model, lang=args.lang, device=args.device, compute_type=args.compute_type,
        out_dir=args.out_dir, formats=formats, beam_size=args.beam_size, prompt=args.prompt,
        paragraph_gap=args.paragraph_gap, vad=not args.no_vad,
        normalize=not args.no_normalize, condition_prev=args.condition_prev,
        diarize=args.speakers or args.num_speakers is not None,
        num_speakers=args.num_speakers, speaker_threshold=args.speaker_threshold,
    )
    return args.files, opts, args


def expand_files(patterns: list[str]) -> list[Path]:
    """cmd.exe не раскрывает маски сам, поэтому раскрываем здесь."""
    files: list[Path] = []
    for p in patterns:
        matches = glob.glob(p) if any(c in p for c in "*?[") else [p]
        if not matches:
            print(f"[WARN] Ничего не найдено по маске: {p}")
        files += [Path(m) for m in matches]
    return list(dict.fromkeys(files))


def run_file(model, src: Path, opts: Options, show_text: bool) -> float:
    """Расшифровывает файл с прогресс-баром tqdm. Возвращает длительность аудио."""
    from tqdm import tqdm

    state: dict = {"bar": None, "done": 0, "status_key": None}

    def on_status(text: str) -> None:
        key = text.split(":")[0]  # «осталось фрагментов N» обновляется часто: печатаем один раз
        if key == state["status_key"]:
            return
        state["status_key"] = key
        tqdm.write(text) if state["bar"] is not None else print(text)

    def on_start(duration: float, lang: str, prob: float) -> None:
        if opts.lang == "auto":
            print(f"Язык: {lang} (уверенность {prob:.0%})")
        print(f"Длительность: {fmt_clock(duration)}")
        state["bar"] = tqdm(total=round(duration), unit="с", dynamic_ncols=True,
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} с [{elapsed}<{remaining}]")

    def on_progress(done: float, total: float) -> None:
        bar = state["bar"]
        pos = min(round(done), bar.total)
        bar.update(pos - state["done"])
        state["done"] = pos

    def on_segment(start: float, end: float, text: str) -> None:
        if show_text:
            state["bar"].write(f"[{fmt_clock(start)}] {text}")

    try:
        r = engine.transcribe_file(model, src, opts, on_status=on_status, on_start=on_start,
                                   on_progress=on_progress, on_segment=on_segment)
    finally:
        if state["bar"] is not None:
            state["bar"].close()

    if opts.diarize:
        print(f"Спикеры: {', '.join(r.speakers)}" if r.speakers
              else "Спикеры: найден один голос, подписи не нужны")
    for p in r.outputs:
        print(f"  -> {p}")
    print(f"Готово: аудио {fmt_clock(r.duration)}, обработка {fmt_clock(r.elapsed)} "
          f"(звук {r.audio_prep:.0f} с), скорость x{r.speed:.1f} realtime")
    return r.duration


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    patterns, opts, args = parse_args(argv)

    files = expand_files(patterns)
    missing = [f for f in files if not f.is_file()]
    for f in missing:
        print(f"[FAIL] Нет такого файла: {f}")

    todo = []
    for f in files:
        if f in missing:
            continue
        if engine.already_done(f, opts) and not args.overwrite:
            print(f"[SKIP] Уже расшифрован: {f} (флаг --overwrite, чтобы переделать)")
        else:
            todo.append(f)
    if not todo:
        print("Нечего расшифровывать.")
        return 1 if missing else 0

    try:
        for line in engine.setup_device(opts.device):
            print(("[WARN] " if opts.device == "cpu" else "") + line)
        print(f"Загружаю модель {opts.model} ({opts.device}, {opts.resolved_compute_type}) ...")
        t0 = time.perf_counter()
        model = engine.load_model(opts)
        print(f"Модель готова за {time.perf_counter() - t0:.1f} с")
    except FatalError as e:
        print(f"[FAIL] {e}")
        return 2

    failed: list[Path] = []
    total_audio = 0.0
    t0 = time.perf_counter()
    for i, src in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {src}")
        try:
            total_audio += run_file(model, src, opts, args.show_text)
        except KeyboardInterrupt:
            print("\nОстановлено. Сделанная часть сохранена в *.partial.*")
            engine.write_log(f"STOPPED\t{src}\t{opts.model}\t{opts.device}")
            return 130
        except Exception as e:
            print(f"[FAIL] {src}: {type(e).__name__}: {e}")
            print(f"       {gpu_env.explain_error(str(e))}")
            engine.write_log(f"FAIL\t{src}\t{opts.model}\t{opts.device}\t{type(e).__name__}: {e}")
            failed.append(src)

    if len(todo) > 1:
        print(f"\nВсего: {len(todo) - len(failed)} из {len(todo)} файлов, "
              f"аудио {fmt_clock(total_audio)} за {fmt_clock(time.perf_counter() - t0)}")
    for f in failed + missing:
        print(f"[FAIL] Не получилось: {f}")
    return 1 if failed or missing else 0


if __name__ == "__main__":
    sys.exit(main())
