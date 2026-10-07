<#
.SYNOPSIS
    Set up the built NeuroPest app on this PC: shortcuts, optional autostart, optional local code signing.

.DESCRIPTION
    Works on dist\NeuroPest\NeuroPest.exe (see tools\build_exe.py) or on an extracted release zip.
    Nothing here needs administrator rights; everything is per-user (HKCU, current user's folders).

    -Desktop     Desktop shortcut.
    -StartMenu   Start menu shortcut (makes it searchable and pinnable to the taskbar).
    -Startup     Start NeuroPest at sign-in, straight to the tray (--tray), via HKCU\...\Run.
    -Sign        Sign NeuroPest.exe with a self-signed certificate and trust it for THIS user only.
                 Windows shows one confirmation dialog when the certificate is added to the trusted roots.
    -Uninstall   Remove the shortcuts and the autostart entry (the signing certificate stays; see docs).

    With no switches it creates the Desktop and Start menu shortcuts.

    A self-signed signature only helps on the PC that trusts the certificate. It does not give other people's
    PCs a known publisher and it does not build SmartScreen reputation. See docs\WINDOWS_APP.md.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\install_windows.ps1 -Sign -Desktop -StartMenu -Startup
#>
[CmdletBinding()]
param(
    [string]$ExePath,
    [switch]$Desktop,
    [switch]$StartMenu,
    [switch]$Startup,
    [switch]$Sign,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$AppName = "NeuroPest"
$CertSubject = "CN=NeuroPest Local Code Signing"
$RunKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"

function Get-ShortcutPath([string]$Folder) { Join-Path $Folder "$AppName.lnk" }
$DesktopLnk = Get-ShortcutPath ([Environment]::GetFolderPath("Desktop"))
$StartMenuLnk = Get-ShortcutPath ([Environment]::GetFolderPath("Programs"))

if ($Uninstall) {
    foreach ($p in $DesktopLnk, $StartMenuLnk) {
        if (Test-Path $p) { Remove-Item $p -Force; Write-Host "removed $p" }
    }
    if (Get-ItemProperty $RunKey -Name $AppName -ErrorAction SilentlyContinue) {
        Remove-ItemProperty $RunKey -Name $AppName
        Write-Host "removed autostart entry"
    }
    return
}

if (-not ($Desktop -or $StartMenu -or $Startup -or $Sign)) { $Desktop = $true; $StartMenu = $true }

if (-not $ExePath) {   # $PSScriptRoot is not reliable inside param() defaults on Windows PowerShell 5.1
    $ExePath = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\dist\NeuroPest\NeuroPest.exe"
}
if (-not (Test-Path $ExePath)) {
    throw "NeuroPest.exe not found at '$ExePath'. Build it first (uv run python tools/build_exe.py) or pass -ExePath."
}
$Exe = (Resolve-Path $ExePath).Path
$AppDir = Split-Path $Exe -Parent

# Files that came from a download carry a Mark-of-the-Web stream; that is what triggers the SmartScreen /
# "Open File - Security Warning" prompts. Locally built files do not have it, so this is a no-op for them.
Get-ChildItem $AppDir -Recurse -File | Unblock-File

if ($Sign) {
    $cert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert |
        Where-Object { $_.Subject -eq $CertSubject -and $_.NotAfter -gt (Get-Date).AddDays(30) } |
        Sort-Object NotAfter -Descending | Select-Object -First 1
    if (-not $cert) {
        Write-Host "creating self-signed code-signing certificate ($CertSubject, 3 years)"
        $cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject $CertSubject `
            -KeyAlgorithm RSA -KeyLength 3072 -HashAlgorithm SHA256 -KeyExportPolicy NonExportable `
            -CertStoreLocation Cert:\CurrentUser\My -NotAfter (Get-Date).AddYears(3)
    }
    # Trust it for this user only: as a root (so the chain validates) and as a trusted publisher.
    # Adding to CurrentUser\Root makes Windows show a confirmation dialog; answer Yes.
    $cer = Join-Path $env:TEMP "neuropest-signing.cer"
    Export-Certificate -Cert $cert -FilePath $cer | Out-Null
    foreach ($store in "Root", "TrustedPublisher") {
        if (-not (Get-ChildItem "Cert:\CurrentUser\$store" | Where-Object Thumbprint -eq $cert.Thumbprint)) {
            Import-Certificate -FilePath $cer -CertStoreLocation "Cert:\CurrentUser\$store" | Out-Null
        }
    }
    Remove-Item $cer -Force

    try {
        $sig = Set-AuthenticodeSignature -FilePath $Exe -Certificate $cert -HashAlgorithm SHA256 `
            -TimestampServer "http://timestamp.digicert.com"
    } catch {
        Write-Warning "timestamping failed ($($_.Exception.Message)); signing without a timestamp"
        $sig = Set-AuthenticodeSignature -FilePath $Exe -Certificate $cert -HashAlgorithm SHA256
    }
    $check = Get-AuthenticodeSignature $Exe
    Write-Host ("signature status: {0} (signer: {1})" -f $check.Status, $check.SignerCertificate.Subject)
    if ($check.Status -ne "Valid") { Write-Warning "Signature is not Valid yet; the certificate may not be trusted. See docs\WINDOWS_APP.md." }
}

function New-Shortcut([string]$Path) {
    $sh = New-Object -ComObject WScript.Shell
    $lnk = $sh.CreateShortcut($Path)
    $lnk.TargetPath = $Exe
    $lnk.WorkingDirectory = $AppDir
    $lnk.IconLocation = "$Exe,0"
    $lnk.Description = "NeuroPest - a desktop fly pet driven by the FlyWire connectome"
    $lnk.Save()
    Write-Host "shortcut: $Path"
}

if ($Desktop) { New-Shortcut $DesktopLnk }
if ($StartMenu) { New-Shortcut $StartMenuLnk }
if ($Startup) {
    Set-ItemProperty $RunKey -Name $AppName -Value ('"{0}" --tray' -f $Exe)
    Write-Host "autostart: NeuroPest will start at sign-in (tray only)"
}
