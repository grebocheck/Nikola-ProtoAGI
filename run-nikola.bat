@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-nikola-stack.ps1" %*
if errorlevel 1 (
  echo.
  echo Nikola stack stopped with an error or interrupt. Check logs if this was unexpected.
  exit /b 1
)
endlocal
