"""Окно транскрибатора: drag-and-drop, выбор модели и языка, прогресс.

Запуск: app.bat или .venv\\Scripts\\pythonw.exe app.py [файлы...]
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

import engine
from gui.drop_zone import media_files_from
from gui.main_window import APP_TITLE, MainWindow


def exception_hook(exc_type, exc_value, exc_tb) -> None:
    text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    engine.write_log(f"CRASH\t{text!r}")
    QMessageBox.critical(None, APP_TITLE, f"Непредвиденная ошибка:\n\n{text}")
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def main() -> int:
    sys.excepthook = exception_hook
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName(APP_TITLE)
    window = MainWindow()
    window.show()
    # файлы, перетащенные на app.bat, приходят аргументами
    files = media_files_from([Path(a) for a in sys.argv[1:]])
    if files:
        window.add_files(files)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
