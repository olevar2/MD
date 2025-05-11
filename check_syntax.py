import os
import sys

def check_syntax(directory):
    errors = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        compile(f.read(), file_path, 'exec')
                except SyntaxError as e:
                    errors.append((file_path, e))
    return errors

if __name__ == "__main__":
    errors = check_syntax('.')
    if errors:
        print("Syntax errors found:")
        for file_path, error in errors:
            print(f"{file_path}: {error}")
    else:
        print("No syntax errors found.")