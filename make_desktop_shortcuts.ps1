# Creates two Desktop shortcuts: the GUI and the headless Assistant.
# Run once:  right-click -> "Run with PowerShell"   (or from a PS prompt:
#            powershell -ExecutionPolicy Bypass -File make_desktop_shortcuts.ps1)

$proj    = "D:\DINOSR\Claude\PaperRun_claude\dino_sr_contrastive"
$desktop = [Environment]::GetFolderPath("Desktop")
$py      = "C:\Users\danielkh\AppData\Local\anaconda3\envs\py4DSTEM_SAM\python.exe"
$ws      = New-Object -ComObject WScript.Shell

function New-AppShortcut($name, $bat, $iconIndex) {
    $lnkPath = Join-Path $desktop ($name + ".lnk")
    $lnk = $ws.CreateShortcut($lnkPath)
    $lnk.TargetPath       = (Join-Path $proj $bat)
    $lnk.WorkingDirectory = $proj
    $lnk.Description       = $name
    # Use a python.exe icon so the shortcuts are recognizable.
    if (Test-Path $py) { $lnk.IconLocation = "$py,0" }
    $lnk.Save()
    Write-Host "Created: $lnkPath"
}

New-AppShortcut "DINO-4DSTEM GUI"        "launch_gui.bat"        0
New-AppShortcut "DINO-4DSTEM Assistant"  "launch_assistant.bat"  0
Write-Host ""
Write-Host "Done. Two icons are on your Desktop."
