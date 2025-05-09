import os
import sys
import subprocess
import datetime

def run_script(script_name):
    """Run a Python script and capture its output."""
    print(f"\n{'=' * 80}")
    print(f"Running {script_name}...")
    print(f"{'=' * 80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Errors:\n{result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

def create_report(results):
    """Create a summary report of all checks."""
    report = f"""
Code Quality Check Report
=========================
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary
-------
"""
    
    all_passed = all(results.values())
    if all_passed:
        report += "✅ All checks passed successfully!\n"
    else:
        report += "❌ Some checks failed. See details below.\n"
    
    report += "\nDetailed Results\n---------------\n"
    
    for script, passed in results.items():
        status = "✅ Passed" if passed else "❌ Failed"
        report += f"{status} - {script}\n"
    
    report += """
Next Steps
----------
1. Review the detailed output of each check
2. Fix any identified issues
3. Consider adding these checks to your CI/CD pipeline
4. Run these checks periodically to maintain code quality

For any duplicate files:
- Determine which one should be kept
- Update references to point to the correct file
- Remove the duplicate

For unnecessary files:
- Verify they are truly unnecessary before removing
- Add patterns to .gitignore to prevent future commits

For technical issues:
- Prioritize fixing security issues (like hardcoded credentials)
- Address syntax errors and large files
- Consider refactoring complex code
"""
    
    return report

if __name__ == "__main__":
    # Create output directory
    output_dir = "code_quality_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # Redirect stdout to file
    original_stdout = sys.stdout
    with open(os.path.join(output_dir, "full_report.txt"), 'w') as f:
        sys.stdout = f
        
        print(f"Code Quality Check Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Run all scripts and collect results
        results = {
            "find_duplicates.py": run_script("find_duplicates.py"),
            "find_unnecessary_files.py": run_script("find_unnecessary_files.py"),
            "check_tech_issues.py": run_script("check_tech_issues.py")
        }
    
    # Reset stdout
    sys.stdout = original_stdout
    
    # Create summary report
    summary_report = create_report(results)
    with open(os.path.join(output_dir, "summary_report.txt"), 'w') as f:
        f.write(summary_report)
    
    print(f"Code quality check completed. Reports saved to {output_dir}/")
    print("\nSummary Report:")
    print(summary_report)
