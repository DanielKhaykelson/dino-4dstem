@echo off
REM ===== DINO-4DSTEM uninstaller (Windows) =====
REM Removes what the installer added OUTSIDE this folder:
REM   - the conda environment (default "dino4dstem", or the name in env_name.txt)
REM   - the three Desktop shortcuts
REM It does NOT delete this folder or any of your data / results.
setlocal
cd /d "%~dp0"
set "ENV_NAME=dino4dstem"
if exist "%~dp0env_name.txt" set /p ENV_NAME=<"%~dp0env_name.txt"

echo ============================================================
echo  DINO-4DSTEM uninstall
echo ============================================================
echo.
echo  This will remove:
echo    - the conda environment "%ENV_NAME%"
echo    - the Desktop shortcuts "DINO-4DSTEM GUI / Assistant / Notebooks"
echo.
echo  It will NOT touch this folder, your data, or your results.
echo.
if /i "%ENV_NAME%"=="base" (
  echo [ERROR] Refusing to remove the conda "base" environment.
  echo         Check env_name.txt. Press any key to close.
  pause >nul
  exit /b 1
)
set "ANSWER="
set /p ANSWER=Type YES to uninstall, anything else to cancel: 
if /i not "%ANSWER%"=="YES" (
  echo.
  echo Cancelled -- nothing was changed. Press any key to close.
  pause >nul
  exit /b 0
)

echo.
echo Removing Desktop shortcuts...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop'); foreach($n in 'DINO-4DSTEM GUI','DINO-4DSTEM Assistant','DINO-4DSTEM Notebooks'){ $p=Join-Path $d ($n+'.lnk'); if(Test-Path -LiteralPath $p){ Remove-Item -LiteralPath $p; Write-Host ('  removed: '+$p) } }"

echo.
call "%~dp0_find_conda.bat"
if errorlevel 1 goto :noconda
echo Removing the conda environment "%ENV_NAME%" (can take a minute)...
call "%CONDA_BAT%" env remove -n %ENV_NAME% -y
if errorlevel 1 (
  echo.
  echo [WARN] conda could not remove "%ENV_NAME%" -- it may not exist, or a
  echo        DINO-4DSTEM window is still open. Close all DINO-4DSTEM / Jupyter
  echo        windows and run this again.
  goto :end
)
echo   removed environment "%ENV_NAME%".

echo.
echo ============================================================
echo  Uninstall complete.
echo.
echo  To remove the program files too, close this window and delete
echo  this folder yourself:
echo    "%~dp0"
echo  (Keep anything you want from your data folders first -- results are
echo   saved next to your data, in ^<data^>_DINO4DSTEM folders.)
echo ============================================================
goto :end

:noconda
echo [WARN] conda not found -- the environment was not removed.

:end
echo.
echo Press any key to close.
pause >nul
endlocal
