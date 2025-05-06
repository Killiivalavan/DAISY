@echo off
echo Testing DAISY Wake Word Detection...
echo.
echo Say "Hey DAISY" to test the wake word detection.
echo Press Ctrl+C to exit.
echo.

python test_wake_word.py %*
pause 