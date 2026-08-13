@echo off
cd /d "%~dp0"

set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
set NO_PROXY=*

:: Kill any previous instance on port 5000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING" 2^>nul') do (
    taskkill /f /pid %%a >nul 2>&1
)

timeout /t 1 /nobreak >nul

:: Launch desktop app with the project virtual environment
.venv\Scripts\python.exe desktop.py
