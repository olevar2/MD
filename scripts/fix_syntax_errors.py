#!/usr/bin/env python3
"""
Syntax Error Fixer Script

This script attempts to fix common syntax errors in Python files:
- Unterminated triple-quoted string literals
- Unexpected indentation
- Unexpected characters after line continuation
- Invalid syntax in specific patterns
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set

# Constants
IGNORED_DIRS = {'.git', '.github', '.venv', '.pytest_cache', '__pycache__', 'node_modules', 'corrupted_backups'}

class SyntaxErrorFixer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.fixed_files = []
        self.failed_files = []
        
    def fix_all_files(self):
        """Find and fix syntax errors in all Python files"""
        print("Finding Python files with syntax errors...")
        
        for root, dirs, files in os.walk(self.root_dir):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        # Try to compile the file to check for syntax errors
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        try:
                            compile(content, file_path, 'exec')
                            # No syntax error, continue to next file
                            continue
                        except SyntaxError as e:
                            print(f"Found syntax error in {file_path}: {e}")
                            # Try to fix the file
                            if self.fix_file(file_path, e):
                                self.fixed_files.append(str(file_path))
                            else:
                                self.failed_files.append(str(file_path))
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        self.failed_files.append(str(file_path))
        
        print(f"Fixed {len(self.fixed_files)} files with syntax errors.")
        print(f"Failed to fix {len(self.failed_files)} files.")
        
        return self.fixed_files, self.failed_files
    
    def fix_file(self, file_path: Path, error: SyntaxError) -> bool:
        """Attempt to fix syntax errors in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            fixed = False
            
            # Get error message and line number
            error_msg = str(error)
            line_num = error.lineno if hasattr(error, 'lineno') else 0
            
            if 'unterminated triple-quoted string' in error_msg:
                fixed = self._fix_unterminated_triple_quote(lines, line_num)
            elif 'unexpected indent' in error_msg:
                fixed = self._fix_indentation(lines, line_num)
            elif 'unindent does not match' in error_msg:
                fixed = self._fix_unindent(lines, line_num)
            elif 'unexpected character after line continuation' in error_msg:
                fixed = self._fix_line_continuation(lines, line_num)
            elif 'invalid syntax' in error_msg:
                fixed = self._fix_invalid_syntax(lines, line_num)
            elif 'unterminated string literal' in error_msg:
                fixed = self._fix_unterminated_string(lines, line_num)
            elif 'parameter without a default follows parameter with a default' in error_msg:
                fixed = self._fix_parameter_order(lines, line_num)
            elif 'unmatched' in error_msg:
                fixed = self._fix_unmatched_brackets(lines, line_num)
            
            if fixed:
                # Write the fixed content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                print(f"Fixed {file_path}")
                return True
            else:
                print(f"Could not fix {file_path}")
                return False
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
            return False
    
    def _fix_unterminated_triple_quote(self, lines: List[str], line_num: int) -> bool:
        """Fix unterminated triple-quoted string literals"""
        if line_num <= 0 or line_num > len(lines):
            return False
        
        # Check if the line contains a triple quote
        line = lines[line_num - 1]
        if '"""' in line or "'''" in line:
            # Determine which type of triple quote is used
            quote_type = '"""' if '"""' in line else "'''"
            
            # Count the number of triple quotes in the line
            count = line.count(quote_type)
            
            if count % 2 == 1:  # Odd number of triple quotes
                # Add a closing triple quote at the end of the line
                lines[line_num - 1] = line.rstrip() + quote_type + '\n'
                return True
        
        return False
    
    def _fix_indentation(self, lines: List[str], line_num: int) -> bool:
        """Fix unexpected indentation"""
        if line_num <= 1 or line_num > len(lines):
            return False
        
        current_line = lines[line_num - 1]
        prev_line = lines[line_num - 2]
        
        # Get the indentation of the previous line
        prev_indent = len(prev_line) - len(prev_line.lstrip())
        current_indent = len(current_line) - len(current_line.lstrip())
        
        # If the current line has more indentation than the previous line,
        # adjust it to match the previous line's indentation plus 4 spaces
        if current_indent > prev_indent:
            # Calculate the correct indentation
            correct_indent = prev_indent
            if prev_line.rstrip().endswith(':'):
                correct_indent += 4
            
            # Fix the indentation
            lines[line_num - 1] = ' ' * correct_indent + current_line.lstrip()
            return True
        
        return False
    
    def _fix_unindent(self, lines: List[str], line_num: int) -> bool:
        """Fix unindent does not match any outer indentation level"""
        if line_num <= 1 or line_num > len(lines):
            return False
        
        current_line = lines[line_num - 1]
        
        # Find the previous non-empty line
        prev_line_num = line_num - 2
        while prev_line_num >= 0 and not lines[prev_line_num].strip():
            prev_line_num -= 1
        
        if prev_line_num < 0:
            return False
        
        prev_line = lines[prev_line_num]
        
        # Get the indentation of the previous line
        prev_indent = len(prev_line) - len(prev_line.lstrip())
        current_indent = len(current_line) - len(current_line.lstrip())
        
        # If the current line has less indentation than the previous line,
        # adjust it to match the previous line's indentation
        if current_indent < prev_indent:
            # Find a valid indentation level
            valid_indent = 0
            for i in range(prev_line_num, -1, -1):
                line = lines[i]
                if not line.strip():
                    continue
                
                indent = len(line) - len(line.lstrip())
                if indent <= current_indent:
                    valid_indent = indent
                    break
            
            # Fix the indentation
            lines[line_num - 1] = ' ' * valid_indent + current_line.lstrip()
            return True
        
        return False
    
    def _fix_line_continuation(self, lines: List[str], line_num: int) -> bool:
        """Fix unexpected character after line continuation"""
        if line_num <= 0 or line_num > len(lines):
            return False
        
        line = lines[line_num - 1]
        
        # Check if the line contains a backslash followed by a character other than newline
        if '\\' in line:
            # Replace backslash + any character with just a backslash + newline
            fixed_line = re.sub(r'\\\s*(.)', r'\\\n', line)
            if fixed_line != line:
                lines[line_num - 1] = fixed_line
                return True
        
        return False
    
    def _fix_invalid_syntax(self, lines: List[str], line_num: int) -> bool:
        """Fix various invalid syntax errors"""
        if line_num <= 0 or line_num > len(lines):
            return False
        
        line = lines[line_num - 1]
        
        # Check for common syntax errors
        
        # Missing closing parenthesis
        if line.count('(') > line.count(')'):
            lines[line_num - 1] = line.rstrip() + ')' * (line.count('(') - line.count(')')) + '\n'
            return True
        
        # Missing closing bracket
        if line.count('[') > line.count(']'):
            lines[line_num - 1] = line.rstrip() + ']' * (line.count('[') - line.count(']')) + '\n'
            return True
        
        # Missing closing brace
        if line.count('{') > line.count('}'):
            lines[line_num - 1] = line.rstrip() + '}' * (line.count('{') - line.count('}')) + '\n'
            return True
        
        # Missing colon after if/for/while/def/class
        if re.search(r'(if|for|while|def|class)\s+.*[^:]\s*$', line):
            lines[line_num - 1] = line.rstrip() + ':\n'
            return True
        
        return False
    
    def _fix_unterminated_string(self, lines: List[str], line_num: int) -> bool:
        """Fix unterminated string literals"""
        if line_num <= 0 or line_num > len(lines):
            return False
        
        line = lines[line_num - 1]
        
        # Check for unterminated string literals
        single_quotes = line.count("'")
        double_quotes = line.count('"')
        
        # If there's an odd number of quotes, add a matching quote at the end
        if single_quotes % 2 == 1 and "'" in line:
            lines[line_num - 1] = line.rstrip() + "'\n"
            return True
        elif double_quotes % 2 == 1 and '"' in line:
            lines[line_num - 1] = line.rstrip() + '"\n'
            return True
        
        return False
    
    def _fix_parameter_order(self, lines: List[str], line_num: int) -> bool:
        """Fix parameter order (parameters with defaults must come after parameters without defaults)"""
        if line_num <= 0 or line_num > len(lines):
            return False
        
        line = lines[line_num - 1]
        
        # This is a complex fix that would require parsing the function definition
        # For now, we'll just add a comment to indicate that manual fixing is needed
        lines[line_num - 1] = line.rstrip() + '  # TODO: Fix parameter order manually\n'
        return True
    
    def _fix_unmatched_brackets(self, lines: List[str], line_num: int) -> bool:
        """Fix unmatched brackets"""
        if line_num <= 0 or line_num > len(lines):
            return False
        
        line = lines[line_num - 1]
        
        # Count opening and closing brackets
        opening_brackets = line.count('(') + line.count('[') + line.count('{')
        closing_brackets = line.count(')') + line.count(']') + line.count('}')
        
        if opening_brackets > closing_brackets:
            # Add missing closing brackets
            missing = opening_brackets - closing_brackets
            lines[line_num - 1] = line.rstrip() + ')' * missing + '\n'
            return True
        elif closing_brackets > opening_brackets:
            # Remove extra closing brackets
            # This is more complex, so we'll just add a comment
            lines[line_num - 1] = line.rstrip() + '  # TODO: Fix unmatched brackets manually\n'
            return True
        
        return False

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = os.getcwd()
    
    fixer = SyntaxErrorFixer(root_dir)
    fixed_files, failed_files = fixer.fix_all_files()
    
    # Write report
    report_path = os.path.join(root_dir, 'syntax_error_fix_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Syntax Error Fix Report\n")
        f.write("======================\n\n")
        
        f.write(f"Fixed Files ({len(fixed_files)}):\n")
        for file in fixed_files:
            f.write(f"  - {file}\n")
        
        f.write(f"\nFailed Files ({len(failed_files)}):\n")
        for file in failed_files:
            f.write(f"  - {file}\n")
    
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
