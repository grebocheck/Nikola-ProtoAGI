@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-nikola-stack.ps1" %*
if errorlevel 1 (
  echo.
  echo Telegram bot failed to start. Check .env and runs\llama-server.stderr.log.
  exit /b 1
)
endlocal
