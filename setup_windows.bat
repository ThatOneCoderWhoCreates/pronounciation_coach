@echo off
REM setup_windows.bat - one-time setup on Windows 10/11.
REM Double-click it, or run it from Command Prompt or PowerShell in this folder.
REM It needs internet once (Whisper models and gTTS reference voices).
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo [1/8] Looking for Python 3.12 or 3.11...
set "PY="
py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info[:2] in [(3, 11), (3, 12)] else 1)" >nul 2>&1 && set "PY=python"
if not defined PY goto :nopython
%PY% --version

echo [2/8] Checking ffmpeg (optional)...
where ffmpeg >nul 2>&1 || echo NOTE: ffmpeg not found. It is optional. To install it: winget install Gyan.FFmpeg

echo [3/8] Creating the virtual environment and installing packages...
if exist ".venv\Scripts\python.exe" goto :venv_ready
%PY% -m venv .venv || goto :fail
:venv_ready
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip || goto :fail
"%VPY%" -m pip install -r requirements.txt || goto :fail

echo [4/8] Downloading Whisper models (about 700 MB)...
"%VPY%" scripts\prefetch_models.py --all || goto :fail

echo [5/8] Creating reference voices...
"%VPY%" scripts\pregenerate_references.py || echo Some clips failed. Run setup_windows.bat again later.

echo [6/8] Calibrating...
"%VPY%" scripts\calibrate.py --auto --save

echo [7/8] Making demo clips...
"%VPY%" scripts\make_demo_audio.py || goto :fail

echo [8/8] Running tests and experiments...
"%VPY%" -m pytest -q || echo Some tests failed. See the output above.
"%VPY%" experiments\run_experiments.py

echo.
echo Setup complete. Start the app by double-clicking run_app.bat
pause
exit /b 0

:nopython
echo Python 3.12 or 3.11 was not found.
echo Install Python 3.12 from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
pause
exit /b 1

:fail
echo.
echo Setup stopped because a step failed. Read the message above.
pause
exit /b 1
