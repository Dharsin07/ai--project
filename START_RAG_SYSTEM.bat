@echo off
echo ========================================
echo Starting RAG System - Easy Launcher
echo ========================================
echo.

REM Check if Python is installed
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

echo Python found. Starting RAG system...
echo.

REM Start the Python launcher
py start_rag_system.py

echo.
echo System stopped. Press any key to exit...
pause >nul
