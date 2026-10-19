@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem Использование:
rem   install.bat            основной путь (faster-whisper)
rem   install.bat fallback   плюс PyTorch cu128 как запасной путь

set "PY="
where py >nul 2>nul
if not errorlevel 1 (
    for %%v in (3.12 3.13 3.11 3.10) do (
        if not defined PY py -%%v -c "import sys" >nul 2>nul && set "PY=py -%%v"
    )
)
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [FAIL] Python не найден. Установите Python 3.12 с python.org, отметив "Add python.exe to PATH".
    goto :fail
)
echo Использую Python: %PY%

if not exist ".venv\Scripts\python.exe" (
    echo Создаю виртуальное окружение .venv ...
    %PY% -m venv .venv || goto :fail
)
set "VPY=.venv\Scripts\python.exe"

"%VPY%" -m pip install --upgrade pip || goto :fail
echo Ставлю зависимости, cuDNN качается долго ...
"%VPY%" -m pip install -r requirements.txt || goto :fail

if /i "%~1"=="fallback" (
    echo Ставлю запасной путь: PyTorch cu128 ...
    "%VPY%" -m pip install -r requirements-fallback.txt || goto :fail
)

echo Скачиваю модель голосов для разметки спикеров ...
"%VPY%" -c "import diarize; diarize.download_models(print)" || echo [WARN] Модель голосов не скачалась, она скачается при первом использовании

where ffmpeg >nul 2>nul || echo [WARN] ffmpeg не найден. Установите: winget install Gyan.FFmpeg

echo.
echo Проверка окружения ...
"%VPY%" check_env.py
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
    echo Готово.
) else (
    echo Проверка нашла проблемы, см. вывод выше.
)
pause
exit /b %RC%

:fail
echo.
echo [FAIL] Установка прервана.
pause
exit /b 1
