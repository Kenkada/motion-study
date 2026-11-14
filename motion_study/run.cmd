@echo off
rem Motion Study - starts the local server and opens the browser.
cd /d "%~dp0"
if not exist "..\.venv\Scripts\python.exe" (
  echo The Python environment ..\.venv was not found. See README.md, "Install".
  pause
  exit /b 1
)
start "" http://127.0.0.1:8765
"..\.venv\Scripts\python.exe" server.py %*
pause
