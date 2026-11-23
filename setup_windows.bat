@echo off
REM setup_windows.bat - one-time setup on Windows 10/11. Double-click it or run it
REM from Command Prompt inside the project folder. Needs internet once.
cd /d "%~dp0"

REM 1. Python 3.12 (or 3.11) through the Windows "py" launcher.
py -3.12 --version >nul 2>&1 && (set PY=py -3.12) || (set PY=py -3.11)
%PY% --version || (echo Install Python 3.12 from python.org and tick "Add to PATH". & pause & exit /b 1)

REM 2. ffmpeg check (recommended: winget install Gyan.FFmpeg)
where ffmpeg >nul 2>&1 || echo NOTE: ffmpeg not found. Recommended: winget install Gyan.FFmpeg

REM 3. Virtual environment and packages
%PY% -m venv .venv || goto :fail
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || goto :fail

REM 4-8. Models, references, calibration, demo clips, tests, experiments
python scripts\prefetch_models.py --all || goto :fail
python scripts\pregenerate_references.py || goto :fail
python scripts\calibrate.py --auto --save
python scripts\make_demo_audio.py || goto :fail
python -m pytest -q
python experiments\run_experiments.py

echo.
echo Setup complete. Start the app with:
echo     .venv\Scripts\activate
echo     python app.py
pause
exit /b 0

:fail
echo Setup stopped because a step failed. Read the message above.
pause
exit /b 1
