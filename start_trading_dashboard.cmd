@echo off
set "PROJECT_DIR=C:\Users\visha\trading-dashboard"
cd /d "%PROJECT_DIR%"
for /f %%P in ('powershell -NoProfile -Command "if (Get-NetTCPConnection -State Listen -LocalPort 8507 -ErrorAction SilentlyContinue) { Write-Output 1 }"') do (
	start "" "http://localhost:8507"
	exit /b 0
)
"%PROJECT_DIR%\.venv\Scripts\streamlit.exe" run "%PROJECT_DIR%\app.py" --server.port 8507 --server.headless false
