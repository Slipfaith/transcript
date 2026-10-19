@echo off
chcp 65001 >nul
setlocal

rem Перетащите файлы лекций на этот файл или запустите:
rem   transcribe.bat lecture.mp4 --model large-v3 --lang ru

set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo [FAIL] Нет окружения .venv. Сначала запустите install.bat
    pause
    exit /b 1
)
if "%~1"=="" (
    echo Перетащите файлы лекций на transcribe.bat или укажите их в командной строке.
    echo.
    "%VPY%" "%~dp0transcribe.py" --help
    pause
    exit /b 1
)

"%VPY%" "%~dp0transcribe.py" %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
