@echo off
REM ===== DINO-4DSTEM Assistant launcher (GUI window, no console) =====
REM Opens the standalone assistant window. Optional: drag a cube onto this
REM .bat to load it on start (%1).
cd /d "%~dp0"
if not exist "%~dp0src\assistant_gui.py" (
  echo.
  echo [ERROR] This folder is incomplete -- src\assistant_gui.py is missing.
  echo.
  echo         Most likely you ran this file straight from INSIDE the .zip,
  echo         or copied it out of the DINO-4DSTEM folder. The launcher needs
  echo         the whole folder next to it.
  echo.
  echo         Fix: right-click the .zip, choose "Extract All...", then run
  echo         this file from the extracted folder.
  echo.
  echo         This file is currently in:
  echo           "%~dp0"
  echo.
  echo Press any key to close.
  pause >nul
  exit /b 1
)
set PYTHONIOENCODING=utf-8
call "%~dp0_activate.bat" || ( echo. & echo Press any key to close. & pause >nul & exit /b 1 )
REM 'start' + pythonw -> the GUI opens in its own window and this console closes.
start "" pythonw "%~dp0src\assistant_gui.py" %1
