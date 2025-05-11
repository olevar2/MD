#!/usr/bin/env python3
"""
Forex Trading Platform User Experience Metrics Analyzer

This script analyzes the user experience aspects of the forex trading platform:
1. UI responsiveness (client-side performance, animations, transitions)
2. Workflow completion rates (user journey steps, form validations)
3. Error handling from user perspective (error messages, recovery paths)
4. Accessibility compliance (ARIA attributes, semantic HTML, keyboard navigation)

Output is a comprehensive JSON file that maps the user experience metrics of the platform.
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
import concurrent.futures
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_OUTPUT_DIR = r"D:\MD\forex_trading_platform\tools\output"

# Directories to exclude from analysis
EXCLUDE_DIRS = {
    ".git", ".github", ".pytest_cache", "__pycache__", 
    "node_modules", ".venv", "venv", "env", ".vscode"
}

# File extensions to analyze
HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
CSS_EXTENSIONS = {".css", ".scss", ".sass", ".less"}
JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}
TEMPLATE_EXTENSIONS = {".vue", ".svelte", ".ejs", ".hbs", ".pug", ".jade"}
ALL_EXTENSIONS = HTML_EXTENSIONS | CSS_EXTENSIONS | JS_EXTENSIONS | TEMPLATE_EXTENSIONS

# Patterns for detecting UI responsiveness
UI_RESPONSIVENESS_PATTERNS = {
    'animation': [
        r'(animation|transition|transform|keyframes)',
        r'(animate|fade|slide|zoom|move)',
        r'(ease|linear|cubic-bezier)',
        r'(duration|delay|timing-function)'
    ],
    'performance': [
        r'(debounce|throttle|requestAnimationFrame)',
        r'(lazy\s*load|defer|async)',
        r'(virtual\s*scroll|windowing|pagination)',
        r'(memoize|useMemo|useCallback|shouldComponentUpdate)'
    ],
    'loading_states': [
        r'(loading|spinner|skeleton|shimmer|placeholder)',
        r'(isLoading|loading\s*state|pending)',
        r'(progress\s*bar|progress\s*indicator)',
        r'(suspense|fallback)'
    ]
}

# Patterns for detecting workflow completion
WORKFLOW_COMPLETION_PATTERNS = {
    'form_validation': [
        r'(validate|validation|validator|isValid)',
        r'(required|pattern|minLength|maxLength)',
        r'(formik|yup|joi|zod|react-hook-form)',
        r'(error\s*message|validation\s*message|hint)'
    ],
    'user_journey': [
        r'(step|wizard|stepper|multi-step)',
        r'(onboarding|tutorial|guide|walkthrough)',
        r'(progress|completion|finished|done)',
        r'(next|previous|back|continue|submit)'
    ],
    'success_feedback': [
        r'(success|complete|finished|done)',
        r'(confirmation|thank\s*you|acknowledgment)',
        r'(toast|notification|alert|message)',
        r'(checkmark|tick|success\s*icon)'
    ]
}

# Patterns for detecting error handling
ERROR_HANDLING_PATTERNS = {
    'error_messages': [
        r'(error\s*message|error\s*text|error\s*state)',
        r'(alert|toast|notification|snackbar)',
        r'(warning|danger|critical|important)',
        r'(try\s*again|retry|recover)'
    ],
    'form_errors': [
        r'(field\s*error|input\s*error|validation\s*error)',
        r'(invalid|invalid\s*input|invalid\s*value)',
        r'(error\s*state|error\s*class|error\s*style)',
        r'(aria-invalid|aria-errormessage)'
    ],
    'empty_states': [
        r'(empty\s*state|no\s*results|no\s*data)',
        r'(not\s*found|404|no\s*match)',
        r'(placeholder|fallback)',
        r'(zero\s*state|initial\s*state)'
    ]
}

# Patterns for detecting accessibility
ACCESSIBILITY_PATTERNS = {
    'aria': [
        r'(aria-[a-z]+)',
        r'(role=)',
        r'(accessible|a11y)',
        r'(screen\s*reader)'
    ],
    'semantic_html': [
        r'(<nav|<header|<main|<footer|<article|<section|<aside)',
        r'(<h1|<h2|<h3|<h4|<h5|<h6)',
        r'(<button|<a\s|<input|<label|<form)',
        r'(<table|<th|<caption)'
    ],
    'keyboard_navigation': [
        r'(tabindex|focus|blur|keydown|keyup|keypress)',
        r'(keyboard|tab\s*key|enter\s*key|escape\s*key)',
        r'(focus\s*trap|focus\s*lock|focus\s*management)',
        r'(keyboard\s*shortcut|hotkey|accelerator)'
    ],
    'color_contrast': [
        r'(contrast|color\s*contrast|background\s*contrast)',
        r'(light\s*mode|dark\s*mode|theme)',
        r'(accessibility\s*theme|high\s*contrast)',
        r'(color\s*blind|deuteranopia|protanopia|tritanopia)'
    ]
}

class UserExperienceMetricsAnalyzer:
    """Analyzes the user experience metrics of the forex trading platform."""
    
    def __init__(self, project_root: Path):
        """
        Initialize the analyzer.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.files = []
        self.html_files = []
        self.css_files = []
        self.js_files = []
        self.template_files = []
        
        self.ui_responsiveness = defaultdict(list)
        self.workflow_completion = defaultdict(list)
        self.error_handling = defaultdict(list)
        self.accessibility = defaultdict(list)
        
    def find_files(self) -> None:
        """Find all relevant files in the project."""
        logger.info(f"Finding files in {self.project_root}...")
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.project_root)
                
                # Skip files in excluded directories
                if any(part in EXCLUDE_DIRS for part in Path(rel_path).parts):
                    continue
                
                # Categorize files by extension
                ext = os.path.splitext(file)[1].lower()
                if ext in ALL_EXTENSIONS:
                    self.files.append(file_path)
                    
                    if ext in HTML_EXTENSIONS:
                        self.html_files.append(file_path)
                    elif ext in CSS_EXTENSIONS:
                        self.css_files.append(file_path)
                    elif ext in JS_EXTENSIONS:
                        self.js_files.append(file_path)
                    elif ext in TEMPLATE_EXTENSIONS:
                        self.template_files.append(file_path)
        
        logger.info(f"Found {len(self.files)} files to analyze")
        logger.info(f"HTML files: {len(self.html_files)}")
        logger.info(f"CSS files: {len(self.css_files)}")
        logger.info(f"JS files: {len(self.js_files)}")
        logger.info(f"Template files: {len(self.template_files)}")
    
    def analyze_html_accessibility(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze HTML file for accessibility features.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Dictionary with accessibility analysis
        """
        result = {
            'aria_attributes': 0,
            'semantic_elements': 0,
            'form_labels': 0,
            'alt_texts': 0,
            'issues': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Count ARIA attributes
            elements_with_aria = soup.select('[aria-*]')
            result['aria_attributes'] = len(elements_with_aria)
            
            # Count semantic elements
            semantic_tags = ['header', 'nav', 'main', 'footer', 'article', 'section', 'aside', 
                            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'figure', 'figcaption']
            semantic_elements = soup.find_all(semantic_tags)
            result['semantic_elements'] = len(semantic_elements)
            
            # Check form labels
            inputs = soup.find_all('input')
            labeled_inputs = 0
            for input_el in inputs:
                if input_el.get('id') and soup.find('label', attrs={'for': input_el['id']}):
                    labeled_inputs += 1
                elif input_el.parent.name == 'label':
                    labeled_inputs += 1
                else:
                    result['issues'].append(f"Input without label: {input_el}")
            result['form_labels'] = labeled_inputs
            
            # Check alt texts
            images = soup.find_all('img')
            images_with_alt = 0
            for img in images:
                if img.get('alt'):
                    images_with_alt += 1
                else:
                    result['issues'].append(f"Image without alt text: {img}")
            result['alt_texts'] = images_with_alt
        
        except Exception as e:
            logger.error(f"Error analyzing HTML accessibility in {file_path}: {e}")
        
        return result
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single file for user experience metrics.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with analysis results
        """
        result = {
            'file_path': file_path,
            'ui_responsiveness': defaultdict(list),
            'workflow_completion': defaultdict(list),
            'error_handling': defaultdict(list),
            'accessibility': defaultdict(list),
            'html_accessibility': None
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for UI responsiveness patterns
            for category, patterns in UI_RESPONSIVENESS_PATTERNS.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content.count('\n', 0, match.start()) + 1
                        result['ui_responsiveness'][category].append({
                            'pattern': pattern,
                            'match': match.group(0),
                            'line': line_num,
                            'file': file_path
                        })
            
            # Check for workflow completion patterns
            for category, patterns in WORKFLOW_COMPLETION_PATTERNS.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content.count('\n', 0, match.start()) + 1
                        result['workflow_completion'][category].append({
                            'pattern': pattern,
                            'match': match.group(0),
                            'line': line_num,
                            'file': file_path
                        })
            
            # Check for error handling patterns
            for category, patterns in ERROR_HANDLING_PATTERNS.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content.count('\n', 0, match.start()) + 1
                        result['error_handling'][category].append({
                            'pattern': pattern,
                            'match': match.group(0),
                            'line': line_num,
                            'file': file_path
                        })
            
            # Check for accessibility patterns
            for category, patterns in ACCESSIBILITY_PATTERNS.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content.count('\n', 0, match.start()) + 1
                        result['accessibility'][category].append({
                            'pattern': pattern,
                            'match': match.group(0),
                            'line': line_num,
                            'file': file_path
                        })
            
            # Perform detailed HTML accessibility analysis for HTML files
            ext = os.path.splitext(file_path)[1].lower()
            if ext in HTML_EXTENSIONS:
                result['html_accessibility'] = self.analyze_html_accessibility(file_path)
        
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
        
        return result
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the user experience metrics.
        
        Returns:
            Analysis results
        """
        logger.info("Starting user experience metrics analysis...")
        
        # Find all files
        self.find_files()
        
        # Analyze files
        logger.info("Analyzing files...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(self.analyze_file, file): file for file in self.files}
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    
                    # Merge results
                    for category, findings in result['ui_responsiveness'].items():
                        self.ui_responsiveness[category].extend(findings)
                    
                    for category, findings in result['workflow_completion'].items():
                        self.workflow_completion[category].extend(findings)
                    
                    for category, findings in result['error_handling'].items():
                        self.error_handling[category].extend(findings)
                    
                    for category, findings in result['accessibility'].items():
                        self.accessibility[category].extend(findings)
                
                except Exception as e:
                    logger.error(f"Error processing result for {file}: {e}")
        
        # Calculate statistics
        ui_responsiveness_stats = {category: len(findings) for category, findings in self.ui_responsiveness.items()}
        workflow_completion_stats = {category: len(findings) for category, findings in self.workflow_completion.items()}
        error_handling_stats = {category: len(findings) for category, findings in self.error_handling.items()}
        accessibility_stats = {category: len(findings) for category, findings in self.accessibility.items()}
        
        # Generate summary
        summary = {
            'ui_responsiveness': dict(self.ui_responsiveness),
            'workflow_completion': dict(self.workflow_completion),
            'error_handling': dict(self.error_handling),
            'accessibility': dict(self.accessibility),
            'stats': {
                'ui_responsiveness': ui_responsiveness_stats,
                'workflow_completion': workflow_completion_stats,
                'error_handling': error_handling_stats,
                'accessibility': accessibility_stats,
                'html_files': len(self.html_files),
                'css_files': len(self.css_files),
                'js_files': len(self.js_files),
                'template_files': len(self.template_files),
                'total_files_analyzed': len(self.files)
            }
        }
        
        logger.info("User experience metrics analysis complete")
        return summary

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze forex trading platform user experience metrics")
    parser.add_argument(
        "--project-root", 
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--output-dir", 
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output files"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze user experience metrics
    analyzer = UserExperienceMetricsAnalyzer(Path(args.project_root))
    results = analyzer.analyze()
    
    # Save results
    output_path = os.path.join(args.output_dir, "user_experience_metrics_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"User experience metrics analysis saved to {output_path}")

if __name__ == "__main__":
    main()
