@echo off
title OpenNotebook Backend Server
echo ============================================
echo   OpenNotebook - Starting Backend Server
echo ============================================
echo.

cd /d "c:\PROJECTS\Dsignz Media\OpenNotebookLM\backend"

echo [1/2] Starting FastAPI Backend on port 8080...
start "OpenNotebook-FastAPI" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8080"

echo [2/2] Starting Celery Worker on main-queue...
timeout /t 3 /nobreak >nul
start "OpenNotebook-Celery" cmd /k ".venv\Scripts\python.exe -m celery -A app.worker.tasks worker --loglevel=info --pool=solo -Q main-queue"

echo.
echo ============================================
echo   Both servers started successfully!
echo   FastAPI:  http://localhost:8080
echo   Celery:   Listening on main-queue
echo ============================================
echo.
echo You can close this window. The servers
echo are running in separate terminal windows.
echo.
pause
