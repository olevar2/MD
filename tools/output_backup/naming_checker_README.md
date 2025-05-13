# Naming Convention Tools

This package provides tools for checking and fixing naming conventions in the codebase according to the standards defined in the `docs/standards/naming_conventions.md` file.

## Features

### Naming Checker

- Checks module and package names
- Checks class names (including interfaces, abstract classes, and exceptions)
- Checks function and method names (including private and magic methods)
- Checks variable and constant names
- Checks type variable names

### Naming Fixer

- Automatically fixes naming convention issues
- Renames modules, classes, functions, methods, variables, and constants
- Updates references to renamed identifiers
- Supports dry run mode to preview changes

## Usage

### Naming Checker

```bash
# Check a single file
python naming_checker.py path/to/file.py

# Check a directory
python naming_checker.py path/to/directory

# Exclude directories
python naming_checker.py path/to/directory --exclude venv node_modules

# Verbose output
python naming_checker.py path/to/directory -v
```

### Naming Fixer

```bash
# Fix a single file
python naming_fixer.py path/to/file.py

# Fix a directory
python naming_fixer.py path/to/directory

# Exclude directories
python naming_fixer.py path/to/directory --exclude venv node_modules

# Dry run (don't modify files)
python naming_fixer.py path/to/directory --dry-run

# Verbose output
python naming_fixer.py path/to/directory -v
```

## Examples

### Naming Checker

```bash
# Check the entire codebase
python naming_checker.py ../../

# Check a specific service
python naming_checker.py ../../data-pipeline-service

# Check a specific module
python naming_checker.py ../../common-lib/common_lib/config
```

### Naming Fixer

```bash
# Dry run on the entire codebase
python naming_fixer.py ../../ --dry-run

# Fix a specific service
python naming_fixer.py ../../data-pipeline-service

# Fix a specific module
python naming_fixer.py ../../common-lib/common_lib/config
```

## Output

### Naming Checker

The naming checker outputs a list of issues found in the codebase, including:

- Module and package names that don't follow the naming convention
- Class names that don't follow the naming convention
- Function and method names that don't follow the naming convention
- Variable and constant names that don't follow the naming convention

Example output:

```
Issues found in 3 files:

../../common-lib/common_lib/config/Config.py:
  - Module name 'Config' does not follow naming convention (lowercase with underscores)
  - Class name 'configManager' does not follow naming convention (PascalCase)
  - Method name 'GetConfig' in class 'configManager' does not follow naming convention (lowercase with underscores)
  - Variable 'Config_Path' does not follow naming convention (lowercase with underscores)

../../data-pipeline-service/data_pipeline_service/adapters/marketDataAdapter.py:
  - Module name 'marketDataAdapter' does not follow naming convention (lowercase with underscores)
  - Class name 'marketDataAdapter' does not follow naming convention (PascalCase)
  - Method name 'getData' in class 'marketDataAdapter' does not follow naming convention (lowercase with underscores)

../../feature-store-service/feature_store_service/api/API.py:
  - Module name 'API' does not follow naming convention (lowercase with underscores)
  - Function name 'GetFeatures' does not follow naming convention (lowercase with underscores)
```

### Naming Fixer

The naming fixer outputs a list of changes made to the codebase, including:

- Renamed modules, classes, functions, methods, variables, and constants
- Updated references to renamed identifiers

Example output:

```
Changes in 3 files:

../../common-lib/common_lib/config/Config.py:
  - Renamed module from 'Config' to 'config'
  - Renamed class from 'configManager' to 'ConfigManager'
  - Renamed method from 'GetConfig' in class 'ConfigManager' to 'get_config'
  - Renamed variable from 'Config_Path' to 'config_path'

../../data-pipeline-service/data_pipeline_service/adapters/marketDataAdapter.py:
  - Renamed module from 'marketDataAdapter' to 'market_data_adapter'
  - Renamed class from 'marketDataAdapter' to 'MarketDataAdapter'
  - Renamed method from 'getData' in class 'MarketDataAdapter' to 'get_data'

../../feature-store-service/feature_store_service/api/API.py:
  - Renamed module from 'API' to 'api'
  - Renamed function from 'GetFeatures' to 'get_features'
```

## Integration with CI/CD

You can integrate these tools with your CI/CD pipeline to ensure that all code follows the naming conventions. For example, you can add the following steps to your GitHub Actions workflow:

```yaml
- name: Check naming conventions
  run: |
    python tools/naming_checker/naming_checker.py . --exclude venv node_modules
    if [ $? -ne 0 ]; then
      echo "Naming convention check failed"
      exit 1
    fi

- name: Fix naming conventions
  run: |
    python tools/naming_checker/naming_fixer.py . --exclude venv node_modules
    if [ $? -ne 0 ]; then
      echo "Naming convention fix failed"
      exit 1
    fi
```

## Customization

### Naming Checker

You can customize the naming conventions by modifying the regular expressions in the `naming_checker.py` file:

```python
# Regular expressions for naming conventions
MODULE_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")
PACKAGE_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")
CLASS_REGEX = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
INTERFACE_REGEX = re.compile(r"^I[A-Z][a-zA-Z0-9]*$")
ABSTRACT_CLASS_REGEX = re.compile(r"^Abstract[A-Z][a-zA-Z0-9]*$")
EXCEPTION_REGEX = re.compile(r"^[A-Z][a-zA-Z0-9]*Error$")
FUNCTION_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")
PRIVATE_METHOD_REGEX = re.compile(r"^_[a-z][a-z0-9_]*$")
MAGIC_METHOD_REGEX = re.compile(r"^__[a-z][a-z0-9_]*__$")
VARIABLE_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")
CONSTANT_REGEX = re.compile(r"^[A-Z][A-Z0-9_]*$")
TYPE_VAR_REGEX = re.compile(r"^[A-Z]$")
```

### Naming Fixer

You can customize the naming conversion functions in the `naming_fixer.py` file:

```python
def to_snake_case(name: str) -> str:
    """
    Convert a name to snake_case.

    Args:
        name: Name to convert

    Returns:
        Name in snake_case
    """
    # Replace non-alphanumeric characters with underscores
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)

    # Insert underscores before uppercase letters
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)

    # Convert to lowercase
    name = name.lower()

    # Replace multiple underscores with a single underscore
    name = re.sub(r"_+", "_", name)

    # Remove leading and trailing underscores
    name = name.strip("_")

    return name

def to_pascal_case(name: str) -> str:
    """
    Convert a name to PascalCase.

    Args:
        name: Name to convert

    Returns:
        Name in PascalCase
    """
    # Replace non-alphanumeric characters with underscores
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)

    # Split by underscores
    words = name.split("_")

    # Capitalize each word
    words = [word.capitalize() for word in words if word]

    # Join words
    name = "".join(words)

    return name
```
