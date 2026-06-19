@echo off
REM Activates the DINO-4DSTEM conda environment in the current shell.
REM (Helper used by the launchers -- not run directly.)
REM
REM If your environment has a different name, change ENV_NAME below.
set "ENV_NAME=dino4dstem"

call "%~dp0_find_conda.bat" || exit /b 1
call "%CONDA_BAT%" activate %ENV_NAME%
if errorlevel 1 (
  echo [ERROR] Could not activate conda env "%ENV_NAME%".
  echo         Run install.bat first, or edit ENV_NAME in _activate.bat.
  exit /b 1
)
exit /b 0
