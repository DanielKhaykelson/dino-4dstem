@echo off
REM Activates the DINO-4DSTEM conda environment in the current shell.
REM (Helper used by the launchers -- not run directly.)
REM
REM If your environment has a different name, change ENV_NAME below.
set "ENV_NAME=dino4dstem"
REM Optional override: put a different env name in env_name.txt next to this file.
if exist "%~dp0env_name.txt" set /p ENV_NAME=<"%~dp0env_name.txt"
REM Quiet conda's "newer version exists" nag on activate (window-local only).
set "CONDA_NOTIFY_OUTDATED_CONDA=false"

call "%~dp0_find_conda.bat" || exit /b 1
call "%CONDA_BAT%" activate %ENV_NAME%
if errorlevel 1 (
  echo [ERROR] Could not activate conda env "%ENV_NAME%".
  echo         Run install.bat first, or edit ENV_NAME in _activate.bat.
  exit /b 1
)
exit /b 0
