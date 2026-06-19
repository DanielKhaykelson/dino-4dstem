@echo off
REM ===== DINO-4DSTEM GUI launcher =====
cd /d "D:\DINOSR\Claude\PaperRun_claude\dino_sr_contrastive"
set PYTHONIOENCODING=utf-8
"C:\Users\danielkh\AppData\Local\anaconda3\envs\py4DSTEM_SAM\python.exe" gui_dino4dstem.py
if errorlevel 1 (
  echo.
  echo The GUI exited with an error. Press any key to close.
  pause >nul
)
