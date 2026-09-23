@echo off
set "PROJECT_DIR=C:\Users\visha\trading-dashboard"
cd /d "%PROJECT_DIR%"
"%PROJECT_DIR%\.venv\Scripts\python.exe" "%PROJECT_DIR%\run_market_scheduler.py" --time 16:00
