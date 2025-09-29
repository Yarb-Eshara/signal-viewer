@echo off
REM Activate the venv located one level up (in signal-viewer\venv)
call "%~dp0..\venv\Scripts\activate.bat"

REM Start the Dash app
start cmd /k python "%~dp0app.py"

REM Wait a bit to let the server start (2 seconds)
timeout /t 2 >nul

REM Open the app in the default browser
start http://127.0.0.1:8050/
