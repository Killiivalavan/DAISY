@echo off
echo Starting DAISY Voice Assistant with GUI...

REM Activate the virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found
)

REM Launch DAISY GUI
python daisy_gui.py %*

REM Keep the window open if there was an error
if %ERRORLEVEL% neq 0 (
    echo.
    echo An error occurred while running DAISY GUI.
    pause
) 