@echo off
REM Locates condabin\conda.bat and sets CONDA_BAT. Exits 1 if not found.
REM (Helper used by install.bat and _activate.bat -- not run directly.)
set "CONDA_BAT="
for %%P in (
  "%USERPROFILE%\anaconda3" "%USERPROFILE%\Anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\miniforge3"
  "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\Continuum\anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\miniforge3"
  "%PROGRAMDATA%\anaconda3" "%PROGRAMDATA%\Anaconda3" "%PROGRAMDATA%\miniconda3"
  "C:\Anaconda3" "C:\Miniconda3" "C:\ProgramData\miniforge3"
) do (
  if exist "%%~P\condabin\conda.bat" set "CONDA_BAT=%%~P\condabin\conda.bat"
)
if not defined CONDA_BAT (
  REM last resort: conda on PATH
  for /f "delims=" %%i in ('where conda.bat 2^>nul') do set "CONDA_BAT=%%i"
)
if not defined CONDA_BAT (
  echo [ERROR] Could not find conda. Please install Miniconda first:
  echo         https://docs.conda.io/projects/miniconda/en/latest/
  exit /b 1
)
exit /b 0
