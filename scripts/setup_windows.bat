@echo off
cd /d %~dp0\..
python -m venv .venv
if errorlevel 1 exit /b 1
call .venv\Scripts\activate
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo Setup complete. Start the app with scripts\run_windows.bat.
