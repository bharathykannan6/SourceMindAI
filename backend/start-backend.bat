@echo off
title OpenNotebook Backend Server
echo ============================================
echo   OpenNotebook - Starting Backend Server
echo ============================================
echo.

cd /d "c:\PROJECTS\Dsignz Media\OpenNotebookLM\backend"

echo Starting FastAPI Backend on port 8080...
start "OpenNotebook-FastAPI" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8080 --limit-max-requests 10000 --timeout-keep-alive 300 --h11-max-incomplete-event-size 536870912"

echo.
echo ============================================
echo   Server started!
echo   FastAPI:  http://localhost:8080
echo   Docs:     http://localhost:8080/docs
echo ============================================
echo.
pause
