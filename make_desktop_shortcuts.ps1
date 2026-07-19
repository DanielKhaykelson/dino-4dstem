# Creates two Desktop shortcuts (GUI + Assistant) pointing at this folder's
# launchers. Portable: uses its own location, no hardcoded paths.
#
# Run from this folder:
#     powershell -ExecutionPolicy Bypass -File make_desktop_shortcuts.ps1

$proj    = $PSScriptRoot
$desktop = [Environment]::GetFolderPath("Desktop")
$ws      = New-Object -ComObject WScript.Shell

# Icon: prefer our own pixel-dino, fall back to the env's python.exe.
$icon    = $null
$dinoIco = Join-Path $proj "assets\dino.ico"
if (Test-Path $dinoIco) { $icon = $dinoIco }
$envName = "dino4dstem"
$roots = @(
  "$env:USERPROFILE\anaconda3", "$env:USERPROFILE\Anaconda3",
  "$env:USERPROFILE\miniconda3", "$env:USERPROFILE\miniforge3",
  "$env:LOCALAPPDATA\anaconda3", "$env:LOCALAPPDATA\miniconda3",
  "$env:LOCALAPPDATA\miniforge3", "$env:PROGRAMDATA\anaconda3",
  "$env:PROGRAMDATA\miniconda3", "C:\Anaconda3", "C:\Miniconda3"
)
if (-not $icon) {
  foreach ($r in $roots) {
    $p = Join-Path $r "envs\$envName\python.exe"
    if (Test-Path $p) { $icon = $p; break }
  }
}

function New-AppShortcut($name, $bat) {
    $lnkPath = Join-Path $desktop ($name + ".lnk")
    $lnk = $ws.CreateShortcut($lnkPath)
    $lnk.TargetPath       = (Join-Path $proj $bat)
    $lnk.WorkingDirectory = $proj
    $lnk.Description       = $name
    if ($icon) { $lnk.IconLocation = "$icon,0" }
    $lnk.Save()
    Write-Host "Created: $lnkPath"
}

New-AppShortcut "DINO-4DSTEM GUI"        "launch_gui.bat"
New-AppShortcut "DINO-4DSTEM Assistant"  "launch_assistant.bat"
Write-Host ""
Write-Host "Done. Two icons are on your Desktop."
