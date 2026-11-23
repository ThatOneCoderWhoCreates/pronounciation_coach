@echo off
REM run_app.bat - start Pronunciation Coach. Double-click it after setup_windows.bat.
REM Extra options are passed on, for example: run_app.bat --port 7861
cd /d "%~dp0"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
if not exist ".venv\Scripts\python.exe" goto :nosetup
".venv\Scripts\python.exe" app.py %*
pause
exit /b 0

:nosetup
echo The virtual environment is missing. Run setup_windows.bat first.
pause
exit /b 1
