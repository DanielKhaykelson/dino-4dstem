@echo off
REM ===== DINO-4DSTEM tutorial-notebooks launcher (portable) =====
REM Opens Jupyter Notebook in the "notebooks" folder using the DINO-4DSTEM
REM conda environment. Works from wherever this folder lives.
cd /d "%~dp0"
if not exist "%~dp0notebooks" (
  echo.
  echo [ERROR] This folder is incomplete -- notebooks is missing.
  echo.
  echo         Most likely you ran this file straight from INSIDE the .zip,
  echo         or copied it out of the DINO-4DSTEM folder. The launcher needs
  echo         the whole folder next to it.
  echo.
  echo         Fix: right-click the .zip, choose "Extract All...", then run
  echo         this file from the extracted folder.
  echo.
  echo         This file is currently in:
  echo           %~dp0
  echo.
  echo Press any key to close.
  pause >nul
  exit /b 1
)
set PYTHONIOENCODING=utf-8
call "%~dp0_activate.bat" || ( echo. & echo Press any key to close. & pause >nul & exit /b 1 )

REM First run only: make sure Jupyter + the interactive-plot widgets are present.
python -c "import notebook" 2>nul || (
  echo.
  echo Installing Jupyter into the environment ^(one time, ~1 min^)...
  python -m pip install --quiet notebook ipywidgets ipympl || (
    echo [ERROR] Could not install Jupyter. Check your internet connection.
    echo Press any key to close. & pause >nul & exit /b 1
  )
)

cd /d "%~dp0notebooks"
echo.
echo Launching Jupyter Notebook in: %CD%
echo (A browser tab will open. Close this window to stop Jupyter.)
echo.
jupyter notebook
if errorlevel 1 (
  echo.
  echo Jupyter exited with an error. Press any key to close.
  pause >nul
)
