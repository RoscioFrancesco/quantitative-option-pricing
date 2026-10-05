@echo off
cd /d %~dp0\..
if not exist .venv\Scripts\streamlit.exe (
  echo Environment not configured. Run scripts\setup_windows.bat first.
  exit /b 1
)
call .venv\Scripts\activate
streamlit run app.py
