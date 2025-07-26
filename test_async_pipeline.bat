@echo off
echo.
echo ======================================================
echo 🤖 DAISY Async Pipeline - Dependency Test and Launch
echo ======================================================
echo.

echo 🔍 Testing dependencies...
python test_async_dependencies.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Dependencies check passed!
    echo.
    echo 🚀 Launching DAISY Async Pipeline...
    echo Press Ctrl+C to stop the pipeline
    echo.
    python daisy_async.py
) else (
    echo.
    echo ❌ Dependencies check failed!
    echo Please install missing dependencies and try again.
    echo.
    pause
) 