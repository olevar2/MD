# Data Pipeline Service Poetry Setup

## What Has Been Done
1. Converted the existing pyproject.toml to Poetry format
2. Added all dependencies including:
   - Python packages (fastapi, uvicorn, pydantic, pandas, etc.)
   - Local dependencies (core-foundations, common-lib)
   - Development dependencies

## Next Steps
To complete the setup:

1. Ensure Poetry is installed on your system:
   ```
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   ```

2. Add Poetry to your PATH (if not already added):
   ```
   $env:PATH += ";$env:USERPROFILE\.poetry\bin"
   ```

3. Navigate to the data-pipeline-service directory:
   ```
   cd d:\MD\forex_trading_platform\data-pipeline-service
   ```

4. Run Poetry install to generate the lock file:
   ```
   poetry install
   ```

## Troubleshooting
If you encounter issues with Poetry installation:

1. Try using Python module approach:
   ```
   python -m poetry install
   ```

2. Check if Poetry is installed but not in PATH:
   ```
   Get-ChildItem -Path $env:USERPROFILE\.poetry -Recurse -Filter "poetry.exe"
   ```

3. Run the included install_dependencies.ps1 script:
   ```
   .\install_dependencies.ps1
   ```
