import os
import re
import subprocess
import sys

def check_python_syntax(file_path):
    """Check Python file for syntax errors."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', file_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return result.stderr
        return None
    except Exception as e:
        return str(e)

def check_js_syntax(file_path):
    """Check JavaScript file for syntax errors using Node.js."""
    try:
        # Check if Node.js is installed
        subprocess.run(['node', '--version'], capture_output=True, check=True)
        
        # Use Node.js to check syntax
        result = subprocess.run(
            ['node', '--check', file_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return result.stderr
        return None
    except subprocess.CalledProcessError:
        return "Node.js not installed or error running Node.js"
    except Exception as e:
        return str(e)

def check_file_issues(file_path):
    """Check a file for various issues."""
    issues = []
    
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # Check file size
    try:
        size = os.path.getsize(file_path)
        if size > 1024 * 1024:  # Larger than 1MB
            issues.append(f"Large file: {size / (1024 * 1024):.2f} MB")
    except Exception as e:
        issues.append(f"Error checking file size: {e}")
    
    # Check for syntax errors in Python files
    if ext == '.py':
        error = check_python_syntax(file_path)
        if error:
            issues.append(f"Python syntax error: {error}")
    
    # Check for syntax errors in JavaScript files
    elif ext in ['.js', '.jsx', '.ts', '.tsx']:
        error = check_js_syntax(file_path)
        if error:
            issues.append(f"JavaScript/TypeScript syntax error: {error}")
    
    # Check for hardcoded credentials
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Check for potential API keys, tokens, passwords
            patterns = [
                r'api[_-]?key["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                r'password["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                r'secret[_-]?key["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                r'token["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                r'auth[_-]?token["\']?\s*[:=]\s*["\']([^"\']+)["\']',
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Ignore if it's clearly a placeholder or variable
                    value = match.group(1)
                    if (value.startswith('$') or 
                        '{' in value or 
                        value.lower() in ['your_api_key', 'your_password', 'your_token']):
                        continue
                    issues.append(f"Potential hardcoded credential: {match.group(0)}")
    except Exception as e:
        issues.append(f"Error checking file content: {e}")
    
    return issues

def scan_directory(directory):
    """Scan directory for technical issues."""
    print(f"Scanning directory: {directory}")
    
    # Dictionary to store files with issues
    files_with_issues = {}
    
    # Extensions to check
    check_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.yml', '.yaml', '.md', '.html', '.css', '.scss']
    
    # Directories to ignore
    ignore_dirs = {'node_modules', '__pycache__', '.git', '.idea', 'venv', 'env', '.vscode'}
    
    # Count files processed
    file_count = 0
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for filename in files:
            # Only check files with specified extensions
            _, ext = os.path.splitext(filename)
            if ext.lower() not in check_extensions:
                continue
                
            filepath = os.path.join(root, filename)
            try:
                issues = check_file_issues(filepath)
                if issues:
                    files_with_issues[filepath] = issues
                
                file_count += 1
                if file_count % 100 == 0:
                    print(f"Processed {file_count} files...")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
    
    print(f"Total files processed: {file_count}")
    
    return files_with_issues

def check_dependency_issues():
    """Check for dependency issues in package.json and requirements.txt files."""
    dependency_issues = {}
    
    # Check Python requirements
    req_files = []
    for root, _, files in os.walk('.'):
        for file in files:
            if file in ['requirements.txt', 'pyproject.toml', 'setup.py']:
                req_files.append(os.path.join(root, file))
    
    if req_files:
        dependency_issues['python'] = {
            'files': req_files,
            'message': "Multiple Python dependency files found. Consider consolidating requirements."
        }
    
    # Check JavaScript dependencies
    package_jsons = []
    for root, _, files in os.walk('.'):
        if 'package.json' in files:
            package_jsons.append(os.path.join(root, 'package.json'))
    
    if len(package_jsons) > 1:
        dependency_issues['javascript'] = {
            'files': package_jsons,
            'message': "Multiple package.json files found. Consider using a monorepo approach or consolidating dependencies."
        }
    
    return dependency_issues

if __name__ == "__main__":
    directory = "."  # Current directory
    
    # Scan for technical issues
    print("Scanning for technical issues...")
    files_with_issues = scan_directory(directory)
    
    if files_with_issues:
        print(f"\nFound {len(files_with_issues)} files with potential issues:")
        for filepath, issues in files_with_issues.items():
            print(f"\n{filepath}:")
            for issue in issues:
                print(f"  - {issue}")
    else:
        print("No technical issues found in scanned files.")
    
    # Check dependency issues
    print("\nChecking for dependency issues...")
    dependency_issues = check_dependency_issues()
    
    if dependency_issues:
        print("\nPotential dependency management issues:")
        for lang, info in dependency_issues.items():
            print(f"\n{lang.capitalize()} dependencies:")
            print(f"  {info['message']}")
            print("  Files:")
            for file in info['files']:
                print(f"    - {file}")
    else:
        print("No dependency management issues found.")
