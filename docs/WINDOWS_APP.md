# Running NeuroPest as a normal Windows app

## Build and install on your own PC

```powershell
uv run python tools/build_exe.py                       # -> dist\NeuroPest\NeuroPest.exe
powershell -ExecutionPolicy Bypass -File tools\install_windows.ps1 -Desktop -StartMenu -Startup
```

`install_windows.ps1` is per-user and needs no administrator rights.

| Switch | Effect |
|---|---|
| `-Desktop`, `-StartMenu` | shortcuts with the fly icon (the Start menu entry can be pinned to the taskbar) |
| `-Startup` | starts NeuroPest at sign-in via `HKCU\...\Run`, with `--tray` so only the tray icon appears |
| `-Sign` | signs `NeuroPest.exe` with a self-signed certificate and trusts it for the current user |
| `-Uninstall` | removes the shortcuts and the autostart entry |

Keep the whole `dist\NeuroPest` folder together (the exe needs the files beside it) and move or rename it before
installing, not after: the shortcuts and the autostart entry point at the exe's current path.

## About the "unknown publisher" warning

There are three different things behind that wording. Which one you see decides the fix.

1. **SmartScreen / "Open File - Security Warning" on a downloaded zip.** Windows tags downloaded files with a
   Mark-of-the-Web. A file you built yourself has no tag and never triggers this; a release zip has it.
   `install_windows.ps1` removes the tag (`Unblock-File`) from everything it installs.
2. **"Publisher: Unknown" in Task Manager > Startup apps, or on the file's Digital Signatures tab.** This is
   simply an unsigned exe. Signing it fills the publisher field.
3. **Smart App Control (Windows 11) blocking unsigned or untrusted apps.** Check with
   `Get-ItemProperty HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy VerifiedAndReputablePolicyState`
   (0 = off, 1 = on, 2 = evaluation). Smart App Control does not accept self-signed certificates and cannot be
   turned back on once off.

### What signing can and cannot do

| Option | Cost | Removes the warning on your PC | Removes it for other people |
|---|---|---|---|
| Self-signed cert (`-Sign`) | free | yes, for item 2 (publisher shows as "NeuroPest Local Code Signing") | no |
| SignPath Foundation | free, but only for OSI-approved open-source licences | yes | mostly, after reputation builds |
| Azure Artifact Signing | about 10 USD/month | yes | mostly, after reputation builds |
| Commercial OV certificate | roughly 70-300 USD/year | yes | mostly, after reputation builds |
| Microsoft Store (MSIX) | one-time developer fee | yes | yes, Microsoft signs the package |

There is **no free way to remove the warning for other people's PCs**. Even with a paid certificate, SmartScreen
builds reputation per publisher or file over time, so the first downloads can still show a prompt. Notes specific
to this repository:

- The licence is PolyForm Noncommercial, which is not OSI-approved, so SignPath Foundation's free programme
  would not accept it as is.
- Azure Artifact Signing is limited to the US and Canada for individuals; check the current list for organisations.
- Do not UPX-pack the exe; `neuropest.spec` has UPX off because packed binaries trigger far more antivirus
  and SmartScreen heuristics.

### How `-Sign` works and what it costs you

It creates `CN=NeuroPest Local Code Signing` in your personal certificate store (private key non-exportable),
adds the public certificate to your **Trusted Root** and **Trusted Publishers** stores (Windows asks you to
confirm), then signs the exe. The certificate is trusted for your user account only, and it can only sign code (it
is not a CA), but any program signed with that key will also be trusted on your account, so keep your account
secure. Rebuilding the exe replaces the signature: run `install_windows.ps1 -Sign` again after each build.

To undo it: open `certmgr.msc`, delete `NeuroPest Local Code Signing` from Personal, Trusted Root Certification
Authorities and Trusted Publishers.
