#!/usr/bin/env python3
"""
Assistant Initialization Script for Forex Trading Platform

This script provides instructions to the AI assistant to read and understand
the platform architecture and fixing log before starting any implementation work.
"""

import os
import sys
import time

# ANSI color codes for better readability
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(text, color):
    """Print text with color"""
    print(f"{color}{text}{Colors.ENDC}")

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print_colored(f"  {text}", Colors.HEADER + Colors.BOLD)
    print("=" * 80)

def print_step(step_num, text):
    """Print a step with number"""
    print_colored(f"\n[Step {step_num}] {text}", Colors.BLUE + Colors.BOLD)

def check_file_exists(file_path):
    """Check if a file exists and return its size"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / 1024  # Size in KB
        print_colored(f"  ✓ Found: {file_path} ({size:.1f} KB)", Colors.GREEN)
        return True
    else:
        print_colored(f"  ✗ Missing: {file_path}", Colors.RED)
        return False

def main():
    """Main function to run the assistant initialization"""
    print_header("Forex Trading Platform - Assistant Initialization")
    
    print_colored("\nThis script prepares the AI assistant to work on the Forex Trading Platform.", Colors.YELLOW)
    print_colored("It ensures the assistant reads and understands the platform architecture", Colors.YELLOW)
    print_colored("and implementation plans before starting any work.", Colors.YELLOW)
    
    # Define critical files
    platform_fixing_log = "D:/MD/forex_trading_platform/platform_fixing_log.md"
    architecture_report = "D:/MD/forex_trading_platform/tools/output/architecture-report.md"
    
    # Check if files exist
    print_step(1, "Checking for critical platform documentation files")
    fixing_log_exists = check_file_exists(platform_fixing_log)
    arch_report_exists = check_file_exists(architecture_report)
    
    if not fixing_log_exists or not arch_report_exists:
        print_colored("\n⚠️ Some critical files are missing. The assistant may not have complete context.", Colors.RED + Colors.BOLD)
        if not fixing_log_exists:
            print_colored("  The platform fixing log is missing. This contains the implementation plans.", Colors.RED)
        if not arch_report_exists:
            print_colored("  The architecture report is missing. This contains the platform structure.", Colors.RED)
        
        proceed = input("\nDo you want to proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            print_colored("Exiting...", Colors.RED)
            sys.exit(1)
    
    # Generate instructions for the assistant
    print_step(2, "Generating instructions for the assistant")
    
    instructions = [
        "Before starting any implementation work, please read and understand these files:",
        f"1. Platform Fixing Log: {platform_fixing_log}",
        f"2. Architecture Report: {architecture_report}",
        "",
        "These files contain:",
        "- The current state of the platform components",
        "- Detailed implementation plans for completing each component",
        "- The platform architecture and service integration map",
        "- Specific missing features and functionality",
        "",
        "When implementing new features or fixing issues:",
        "- Follow the implementation plans in the platform fixing log",
        "- Ensure proper integration with existing components as shown in the architecture report",
        "- Place new files in the correct location according to the platform structure",
        "- Maintain consistent coding patterns with the existing codebase",
        "",
        "Do not start any implementation until you have read and understood these files."
    ]
    
    for line in instructions:
        print_colored(f"  {line}", Colors.GREEN)
        time.sleep(0.1)  # Slight delay for readability
    
    # Final instructions
    print_step(3, "Ready for assistant handoff")
    print_colored("\nInstructions for you:", Colors.YELLOW + Colors.BOLD)
    print_colored("1. Copy the following prompt when starting a new chat with the assistant", Colors.YELLOW)
    print_colored("2. Wait for the assistant to read the files before requesting implementation work", Colors.YELLOW)
    print_colored("3. Reference specific sections of the platform fixing log when requesting work", Colors.YELLOW)
    
    # Generate prompt for the user to copy
    print_header("COPY THIS PROMPT FOR THE ASSISTANT")
    
    prompt = f"""I'd like to continue our work on fixing the forex trading platform. Please start by reading these two critical files:

1. Platform Fixing Log: {platform_fixing_log}
   This contains our detailed analysis and implementation plans for each component.

2. Architecture Report: {architecture_report}
   This contains the platform structure and integration map.

After reading these files, let me know you understand the current state of the platform and the implementation plans. We'll be following a code-first approach with minimal documentation, and we need to run any scripts immediately after writing them.

Once you've confirmed you've read and understood these files, we can proceed with implementation work.
"""
    
    print(prompt)
    print("=" * 80)
    
    print_colored("\nScript execution complete. You can now start a new chat with the assistant.", Colors.GREEN + Colors.BOLD)

if __name__ == "__main__":
    main()