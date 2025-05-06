@echo off
echo Starting DAISY Voice Assistant...
echo.
echo Make sure Ollama is running with the llama3.2 model.
echo If not, open a new terminal and run: ollama serve
echo.

REM Check if wake word mode parameter is provided
if "%1"=="--no-wake-word" (
    echo Starting DAISY without wake word detection...
    python daisy.py --no-wake-word %2 %3 %4 %5
) else (
    echo Starting DAISY with wake word detection...
    echo Make sure you have set up your Porcupine access key in .env file.
    python daisy.py %1 %2 %3 %4 %5
)

pause
 