@echo off
rem Double-click to start the prompt-injection defence dashboard.
cd /d "%~dp0"
python run_dashboard.py %*
pause
