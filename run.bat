@echo off
REM ============================================================
REM  CV Analyzer - start the application
REM
REM  Double-click this file, or run it from a terminal.
REM  Closing the window stops the server.
REM ============================================================

cd /d "%~dp0"

echo.
echo   CV Analyzer
echo   ------------------------------------------------------
echo.

if not exist ".venv\Scripts\python.exe" (
    echo   No virtual environment found in this folder.
    echo.
    echo   Create one first:
    echo       python -m venv .venv
    echo       .venv\Scripts\activate
    echo       pip install -r requirements.txt
    echo       python -m spacy download en_core_web_sm
    echo.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo   app.py was not found in this folder.
    echo   Put this file in the project root, next to app.py.
    echo.
    pause
    exit /b 1
)

REM Stale bytecode can mask replaced source files, so it is cleared each start.
if exist "__pycache__"       rmdir /s /q "__pycache__"
if exist "utils\__pycache__" rmdir /s /q "utils\__pycache__"

echo   Starting on http://127.0.0.1:8001
echo.
echo     Single analysis    http://127.0.0.1:8001
echo     Batch automation   http://127.0.0.1:8001/batch
echo.
echo   Press Ctrl+C to stop.
echo.

".venv\Scripts\python.exe" -m uvicorn app:app --reload --port 8001

echo.
echo   Server stopped.
pause
