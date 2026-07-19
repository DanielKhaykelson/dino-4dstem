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

REM ------------------------------------------------------------
REM  Does the environment already exist?
REM
REM  We ASK CONDA rather than guessing a folder: envs do not always
REM  live under the conda install root.  When that root isn't
REM  user-writable (e.g. C:\ProgramData\miniconda3), conda puts them
REM  in the per-user dir instead, e.g.
REM      C:\Users\<you>\AppData\Local\.conda\envs\dino4dstem
REM  Guessing the path made this script try to CREATE an env that
REM  already existed -> "CondaValueError: prefix already exists".
REM ------------------------------------------------------------
set "ENVLIST=%TEMP%\dino4dstem_envlist.txt"
call "%CONDA_BAT%" env list > "%ENVLIST%" 2>&1

set "ENV_EXISTS="
findstr /b /c:"%ENV_NAME% " "%ENVLIST%" >nul 2>&1 && set "ENV_EXISTS=1"
if not defined ENV_EXISTS (
  findstr /i /c:"\envs\%ENV_NAME%" "%ENVLIST%" >nul 2>&1 && set "ENV_EXISTS=1"
)
del "%ENVLIST%" >nul 2>&1

if defined ENV_EXISTS goto :update

:create
echo Creating environment "%ENV_NAME%" -- this can take several minutes...
call "%CONDA_BAT%" env create -f "%~dp0environment.yml"
if not errorlevel 1 goto :ok
echo.
echo Create did not succeed -- the environment may already exist somewhere
echo conda didn't list. Trying an update instead...
echo.

:update
echo Environment "%ENV_NAME%" already exists -- updating it...
call "%CONDA_BAT%" env update -n %ENV_NAME% -f "%~dp0environment.yml" --prune
if errorlevel 1 goto :failed

:ok
echo.
echo ============================================================
echo  STEP 1 complete -- the environment "%ENV_NAME%" is ready.
echo.
echo  NEXT:  double-click  install_2_dino4dstem.bat
echo         (sets up DINO-4DSTEM itself + desktop icons)
echo ============================================================
pause
exit /b 0

:failed
echo.
echo [ERROR] Environment setup failed. See the messages above.
echo.
echo  Common causes:
echo    * No internet connection.
echo    * A pinned package is unavailable for this Python version.
echo    * The existing environment is broken/half-installed.
echo.
echo  To start completely clean, run this once, then re-run this script:
echo.
echo      "%CONDA_BAT%" env remove -n %ENV_NAME%
echo.
pause
exit /b 1
