@echo off
REM ============================================================
REM  DINO-4DSTEM  --  full install (runs both steps)
REM
REM  This is the "just do everything" option.  It runs:
REM     install_1_environment.bat   (conda env + dependencies, slow)
REM     install_2_dino4dstem.bat    (app check + Desktop icons, fast)
REM
REM  You can also run those two separately -- handy when updating the
REM  app later, since only step 2 is needed for a code update.
REM ============================================================
cd /d "%~dp0"

echo ============================================================
echo  DINO-4DSTEM install  --  step 1 of 2: environment
echo ============================================================
echo.
call "%~dp0install_1_environment.bat"
if errorlevel 1 (
  echo.
  echo [ERROR] Step 1 failed -- stopping. Fix the errors above and re-run.
  exit /b 1
)

echo.
echo ============================================================
echo  DINO-4DSTEM install  --  step 2 of 2: the app
echo ============================================================
echo.
call "%~dp0install_2_dino4dstem.bat"
exit /b %errorlevel%
