# PowerShell script to install @jlia0/servers for VS Code Insiders

# Check if VS Code Insiders is installed
try {
    $codeInsidersPath = (Get-Command code-insiders -ErrorAction Stop).Source
    Write-Host "VS Code Insiders found at: $codeInsidersPath"
} catch {
    Write-Host "VS Code Insiders not found in PATH. Please make sure it's installed and accessible."
    exit 1
}

# Read the existing mcp-servers.json file if it exists
$vscodeInsidersDir = "$env:APPDATA\Code - Insiders\User"
$configFile = "$vscodeInsidersDir\mcp-servers.json"
$configContent = $null

if (Test-Path $configFile) {
    $configContent = Get-Content -Path $configFile -Raw | ConvertFrom-Json
    Write-Host "Existing configuration file found. Adding new server."
} else {
    Write-Host "No existing configuration file found. Creating new one."
    if (-not (Test-Path $vscodeInsidersDir)) {
        New-Item -ItemType Directory -Path $vscodeInsidersDir -Force | Out-Null
    }
    $configContent = @{
        mcpServers = @{}
    }
}

# Add the @jlia0/servers configuration
if ($configContent.mcpServers -eq $null) {
    $configContent.mcpServers = @{}
}

$configContent.mcpServers | Add-Member -MemberType NoteProperty -Name "servers" -Value @{
    command = "cmd"
    args = @(
        "/c",
        "npx",
        "-y",
        "@smithery/cli@latest",
        "run",
        "@jlia0/servers",
        "--key",
        "0f552d54-94f7-4f3c-b89c-cb286cd042d0"
    )
} -Force

# Save the updated configuration
$configContent | ConvertTo-Json -Depth 10 | Set-Content -Path $configFile

Write-Host "Configuration file updated at: $configFile"
Write-Host "Please restart VS Code Insiders to apply the changes."