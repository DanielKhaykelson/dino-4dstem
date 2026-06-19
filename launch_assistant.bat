@echo off
REM ===== DINO-4DSTEM headless Assistant launcher (no GUI) =====
cd /d "D:\DINOSR\Claude\PaperRun_claude\dino_sr_contrastive"
set PYTHONIOENCODING=utf-8
REM Optional: drag a cube onto this .bat to load it on start (%1).
"C:\Users\danielkh\AppData\Local\anaconda3\envs\py4DSTEM_SAM\python.exe" src\assistant_headless.py %1
echo.
echo Assistant closed. Press any key to exit.
pause >nul
