# PowerShell script to install desktop-commander for VS Code Insiders

# Check if VS Code Insiders is installed
try {
    $codeInsidersPath = (Get-Command code-insiders -ErrorAction Stop).Source
    Write-Host "VS Code Insiders found at: $codeInsidersPath"
} catch {
    Write-Host "VS Code Insiders not found in PATH. Please make sure it's installed and accessible."
    exit 1
}

# Create a settings file for the Smithery CLI
$settingsDir = "$env:APPDATA\smithery"
if (-not (Test-Path $settingsDir)) {
    New-Item -ItemType Directory -Path $settingsDir -Force | Out-Null
}

$settingsFile = "$settingsDir\settings.json"
$settingsContent = @{
    userId = "51b49ce5-13f7-4661-8431-a13d87b34b7b"
    analyticsConsent = $true
    askedConsent = $true
    cache = @{
        servers = @{}
    }
} | ConvertTo-Json

Set-Content -Path $settingsFile -Value $settingsContent

# Create a configuration file for the desktop-commander extension
$configContent = @{
    mcpServers = @{
        "desktop-commander" = @{
            command = "cmd"
            args = @(
                "/c",
                "npx",
                "-y",
                "@smithery/cli@latest",
                "run",
                "@wonderwhy-er/desktop-commander",
                "--key",
                "0f552d54-94f7-4f3c-b89c-cb286cd042d0"
            )
        }
    }
} | ConvertTo-Json -Depth 10

$vscodeInsidersDir = "$env:APPDATA\Code - Insiders\User"
if (-not (Test-Path $vscodeInsidersDir)) {
    New-Item -ItemType Directory -Path $vscodeInsidersDir -Force | Out-Null
}

$configFile = "$vscodeInsidersDir\mcp-servers.json"
Set-Content -Path $configFile -Value $configContent

Write-Host "Configuration file created at: $configFile"
Write-Host "Please restart VS Code Insiders to apply the changes."