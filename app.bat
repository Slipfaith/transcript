@echo off
chcp 65001 >nul
setlocal

rem Окно транскрибатора. Файлы можно перетащить прямо на этот .bat

set "VPYW=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%VPYW%" (
    echo [FAIL] Нет окружения .venv. Сначала запустите install.bat
    pause
    exit /b 1
)
start "" "%VPYW%" "%~dp0app.py" %*
