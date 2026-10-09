@echo off
REM ============================================================
REM  DINO-4DSTEM  --  STEP 2 of 2:  install DINO-4DSTEM itself
REM
REM  Checks the app can run in the environment from step 1, then
REM  creates the Desktop icons.  This step is FAST -- re-run it any
REM  time you update the code (you do NOT need to redo step 1).
REM ============================================================
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
REM Suppress conda's "a newer version exists" nag (see install_1 for why).
set "CONDA_NOTIFY_OUTDATED_CONDA=false"

call "%~dp0_activate.bat" || (
  echo.
  echo [ERROR] The environment isn't ready.
  echo         Run  install_1_environment.bat  first.
  echo.
  pause
  exit /b 1
)

echo Checking DINO-4DSTEM can start in this environment...
python -c "import sys, numpy, torch, customtkinter, py4DSTEM; print('  python     ', sys.version.split()[0]); print('  numpy      ', numpy.__version__); print('  torch      ', torch.__version__); print('  py4DSTEM   ', py4DSTEM.__version__); print('  customtkinter', customtkinter.__version__)"
if errorlevel 1 (
  echo.
  echo [ERROR] The environment is missing packages.
  echo         Re-run  install_1_environment.bat.
  echo.
  pause
  exit /b 1
)

echo.
echo Creating Desktop icons...
powershell -ExecutionPolicy Bypass -File "%~dp0make_desktop_shortcuts.ps1"
if errorlevel 1 (
  echo.
  echo [WARN] Could not create the Desktop icons automatically.
  echo        You can still launch with  launch_gui.bat  in this folder.
)

echo.
echo ============================================================
echo  DINO-4DSTEM is installed.
echo.
echo  Launch it from the Desktop icon "DINO-4DSTEM GUI",
echo  or run  launch_gui.bat  in this folder.
echo.
echo  Tutorial notebooks: the "DINO-4DSTEM Notebooks" Desktop icon
echo  (or run  launch_notebooks.bat).
echo ============================================================
pause
