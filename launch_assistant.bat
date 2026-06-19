@echo off
REM ===== DINO-4DSTEM headless Assistant launcher (portable) =====
REM Works from wherever this folder lives -- no hardcoded paths.
REM Optional: drag a cube file onto this .bat to load it on start (%1).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call "%~dp0_activate.bat" || ( echo. & echo Press any key to close. & pause >nul & exit /b 1 )
python "%~dp0src\assistant_headless.py" %1
echo.
echo Assistant closed. Press any key to exit.
pause >nul
