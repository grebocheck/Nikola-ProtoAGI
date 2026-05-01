@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=src
python -m unittest discover -s tests -v
endlocal
