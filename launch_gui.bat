@echo off
REM ===== DINO-4DSTEM GUI launcher (portable) =====
REM Works from wherever this folder lives -- no hardcoded paths.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call "%~dp0_activate.bat" || ( echo. & echo Press any key to close. & pause >nul & exit /b 1 )
python "%~dp0src\gui_dino4dstem.py"
if errorlevel 1 (
  echo.
  echo The GUI exited with an error. Press any key to close.
  pause >nul
)
