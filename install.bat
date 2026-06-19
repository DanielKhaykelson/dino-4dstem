@echo off
REM ============================================================
REM  DINO-4DSTEM one-time installer
REM  Creates (or updates) the conda environment "dino4dstem".
REM  Prerequisite: Miniconda or Anaconda installed.
REM ============================================================
cd /d "%~dp0"
set "ENV_NAME=dino4dstem"

call "%~dp0_find_conda.bat" || ( echo. & pause & exit /b 1 )
echo Using conda at: %CONDA_BAT%
echo.

REM Determine conda root (CONDA_BAT = <root>\condabin\conda.bat)
for %%I in ("%CONDA_BAT%\..\..") do set "CONDA_ROOT=%%~fI"

if exist "%CONDA_ROOT%\envs\%ENV_NAME%\python.exe" (
  echo Environment "%ENV_NAME%" already exists -- updating it...
  call "%CONDA_BAT%" env update -n %ENV_NAME% -f "%~dp0environment.yml" --prune
) else (
  echo Creating environment "%ENV_NAME%" -- this can take several minutes...
  call "%CONDA_BAT%" env create -f "%~dp0environment.yml"
)

if errorlevel 1 (
  echo.
  echo [ERROR] Environment setup failed. See the messages above.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo  Install complete.
echo.
echo  Next step -- copy/paste this line to create desktop icons:
echo.
echo     powershell -ExecutionPolicy Bypass -File "%~dp0make_desktop_shortcuts.ps1"
echo.
echo  After that, launch DINO-4DSTEM from the desktop icons.
echo ============================================================
pause
