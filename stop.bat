@echo off
echo Stopping DataBoard...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do (
    taskkill /f /pid %%a 2>nul
)
echo Done.
timeout /t 2 /nobreak >nul
