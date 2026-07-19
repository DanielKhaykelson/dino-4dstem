@echo off
REM ============================================================
REM  DINO-4DSTEM  --  STEP 1 of 2:  install the ENVIRONMENT
REM
REM  Creates (or updates) the conda environment "dino4dstem" with
REM  all Python dependencies.  This is the big, slow, one-time step
REM  (downloads a few GB).  You only repeat it if the dependencies
REM  change -- updating the app itself does NOT need this.
REM
REM  Prerequisite: Miniconda or Anaconda installed.
REM ============================================================
cd /d "%~dp0"
set "ENV_NAME=dino4dstem"
if exist "%~dp0env_name.txt" set /p ENV_NAME=<"%~dp0env_name.txt"

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
  echo         Common cause: no internet, or a pinned package is
  echo         unavailable for this Python version.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo  STEP 1 complete -- the environment "%ENV_NAME%" is ready.
echo.
echo  NEXT:  double-click  install_2_dino4dstem.bat
echo         (sets up DINO-4DSTEM itself + desktop icons)
echo ============================================================
pause
