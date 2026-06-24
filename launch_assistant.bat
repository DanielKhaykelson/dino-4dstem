@echo off
REM ===== DINO-4DSTEM Assistant launcher (GUI window, no console) =====
REM Opens the standalone assistant window. Optional: drag a cube onto this
REM .bat to load it on start (%1).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call "%~dp0_activate.bat" || ( echo. & echo Press any key to close. & pause >nul & exit /b 1 )
REM 'start' + pythonw -> the GUI opens in its own window and this console closes.
start "" pythonw "%~dp0src\assistant_gui.py" %1
