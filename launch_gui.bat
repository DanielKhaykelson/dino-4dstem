@echo off
REM ===== DINO-4DSTEM GUI launcher (portable) =====
REM Works from wherever this folder lives -- no hardcoded paths.
cd /d "%~dp0"
if not exist "%~dp0src\gui_dino4dstem.py" (
  echo.
  echo [ERROR] This folder is incomplete -- src\gui_dino4dstem.py is missing.
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
python "%~dp0src\gui_dino4dstem.py"
if errorlevel 1 (
  echo.
  echo The GUI exited with an error. Press any key to close.
  pause >nul
)
