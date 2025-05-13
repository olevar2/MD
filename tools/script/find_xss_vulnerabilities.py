#!/usr/bin/env python3
"""
Find potential XSS vulnerabilities in Python files.
"""

import os
import re
from typing import List, Tuple

def find_xss_vulnerabilities(directories: List[str]) -> List[Tuple[str, int, str, str]]:
    """
    Find potential XSS vulnerabilities in Python files.
    
    Args:
        directories: List of directories to search
    
    Returns:
        List of tuples (file_path, line_number, vulnerability_type, line)
    """
    results = []
    
    # Patterns to look for
    patterns = [
        (r'render_template\(.*\{\{.*\}\}', 'Potential XSS in template rendering'),
        (r'render\(.*\{\{.*\}\}', 'Potential XSS in template rendering'),
        (r'render_to_string\(.*\{\{.*\}\}', 'Potential XSS in template rendering'),
        (r'html\s*=', 'Potential XSS in HTML assignment'),
        (r'innerHTML\s*=', 'Potential XSS in innerHTML assignment'),
        (r'document\.write\(', 'Potential XSS in document.write'),
        (r'\.html\(', 'Potential XSS in jQuery html()'),
        (r'dangerouslySetInnerHTML', 'Potential XSS in React dangerouslySetInnerHTML'),
        (r'@Html\.Raw\(', 'Potential XSS in ASP.NET Html.Raw'),
        (r'mark_safe\(', 'Potential XSS in Django mark_safe'),
        (r'safe\s+filter', 'Potential XSS in template safe filter'),
        (r'v-html', 'Potential XSS in Vue v-html'),
        (r'\.parseHTML\(', 'Potential XSS in parseHTML'),
        (r'\.insertAdjacentHTML\(', 'Potential XSS in insertAdjacentHTML'),
        (r'eval\(', 'Potential XSS in eval'),
        (r'Function\(', 'Potential XSS in Function constructor'),
        (r'setTimeout\(.*,', 'Potential XSS in setTimeout'),
        (r'setInterval\(.*,', 'Potential XSS in setInterval'),
        (r'location\.href\s*=', 'Potential XSS in location.href assignment'),
        (r'location\.replace\(', 'Potential XSS in location.replace'),
        (r'location\.assign\(', 'Potential XSS in location.assign'),
        (r'\.src\s*=', 'Potential XSS in src assignment'),
        (r'\.setAttribute\([\'"]src[\'"]\s*,', 'Potential XSS in setAttribute for src'),
        (r'\.setAttribute\([\'"]href[\'"]\s*,', 'Potential XSS in setAttribute for href'),
        (r'\.href\s*=', 'Potential XSS in href assignment'),
        (r'<script>.*</script>', 'Potential XSS in script tags'),
        (r'<iframe>.*</iframe>', 'Potential XSS in iframe tags'),
        (r'<img.*onerror=', 'Potential XSS in img onerror'),
        (r'<a.*href=', 'Potential XSS in a href'),
        (r'<input.*value=', 'Potential XSS in input value'),
        (r'<textarea>.*</textarea>', 'Potential XSS in textarea'),
        (r'<div.*>.*</div>', 'Potential XSS in div'),
        (r'<span.*>.*</span>', 'Potential XSS in span'),
        (r'<p.*>.*</p>', 'Potential XSS in p'),
        (r'<h[1-6].*>.*</h[1-6]>', 'Potential XSS in heading'),
        (r'<button.*>.*</button>', 'Potential XSS in button'),
        (r'<select.*>.*</select>', 'Potential XSS in select'),
        (r'<option.*>.*</option>', 'Potential XSS in option'),
        (r'<label.*>.*</label>', 'Potential XSS in label'),
        (r'<form.*>.*</form>', 'Potential XSS in form'),
        (r'<table.*>.*</table>', 'Potential XSS in table'),
        (r'<tr.*>.*</tr>', 'Potential XSS in tr'),
        (r'<td.*>.*</td>', 'Potential XSS in td'),
        (r'<th.*>.*</th>', 'Potential XSS in th'),
        (r'<ul.*>.*</ul>', 'Potential XSS in ul'),
        (r'<ol.*>.*</ol>', 'Potential XSS in ol'),
        (r'<li.*>.*</li>', 'Potential XSS in li'),
        (r'<a.*>.*</a>', 'Potential XSS in a'),
        (r'<img.*>', 'Potential XSS in img'),
        (r'<input.*>', 'Potential XSS in input'),
        (r'<textarea.*>.*</textarea>', 'Potential XSS in textarea'),
        (r'<div.*>', 'Potential XSS in div'),
        (r'<span.*>', 'Potential XSS in span'),
        (r'<p.*>', 'Potential XSS in p'),
        (r'<h[1-6].*>', 'Potential XSS in heading'),
        (r'<button.*>', 'Potential XSS in button'),
        (r'<select.*>', 'Potential XSS in select'),
        (r'<option.*>', 'Potential XSS in option'),
        (r'<label.*>', 'Potential XSS in label'),
        (r'<form.*>', 'Potential XSS in form'),
        (r'<table.*>', 'Potential XSS in table'),
        (r'<tr.*>', 'Potential XSS in tr'),
        (r'<td.*>', 'Potential XSS in td'),
        (r'<th.*>', 'Potential XSS in th'),
        (r'<ul.*>', 'Potential XSS in ul'),
        (r'<ol.*>', 'Potential XSS in ol'),
        (r'<li.*>', 'Potential XSS in li'),
        (r'<a.*>', 'Potential XSS in a'),
    ]
    
    # Exclusion patterns
    exclusion_patterns = [
        r'escape\(',
        r'escape_html\(',
        r'html\.escape\(',
        r'html_escape\(',
        r'sanitize\(',
        r'sanitize_html\(',
        r'html_sanitize\(',
        r'bleach\.',
        r'clean\(',
        r'linkify\(',
        r'strip_tags\(',
        r'strip_html\(',
        r'html_strip\(',
        r'cgi\.escape\(',
        r'html_safe\(',
        r'safe_html\(',
        r'htmlspecialchars\(',
        r'htmlentities\(',
        r'encodeURIComponent\(',
        r'encodeURI\(',
        r'escape_once\(',
        r'h\(',
        r'html_safe\(',
        r'raw\(',
        r'safe\(',
        r'sanitize\(',
        r'strip_tags\(',
        r'xss\(',
        r'xssfilter\(',
        r'xss_clean\(',
        r'xss_filter\(',
        r'xss_sanitize\(',
        r'xss_safe\(',
        r'xss_secure\(',
        r'xss_strip\(',
        r'xss_remove\(',
        r'xss_escape\(',
        r'xss_clean_recursive\(',
        r'xss_sanitize_recursive\(',
        r'xss_filter_recursive\(',
        r'xss_remove_recursive\(',
        r'xss_escape_recursive\(',
        r'xss_strip_recursive\(',
        r'xss_safe_recursive\(',
        r'xss_secure_recursive\(',
        r'xss_clean_array\(',
        r'xss_sanitize_array\(',
        r'xss_filter_array\(',
        r'xss_remove_array\(',
        r'xss_escape_array\(',
        r'xss_strip_array\(',
        r'xss_safe_array\(',
        r'xss_secure_array\(',
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
            
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith(('.py', '.html', '.js', '.jsx', '.ts', '.tsx', '.vue', '.cshtml', '.aspx', '.php', '.jsp', '.jspx', '.tag', '.tpl', '.tmpl', '.hbs', '.handlebars', '.mustache', '.ejs', '.erb', '.haml', '.slim', '.pug', '.jade', '.liquid', '.twig', '.jinja', '.jinja2', '.j2', '.njk', '.nunjucks', '.nunjs', '.nunj', '.nj', '.nk', '.nk2', '.nunj2', '.nunjucks2', '.nunjs2', '.nunj2', '.nj2', '.nk2', '.nk22')):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                            for pattern, vulnerability_type in patterns:
                                for match in re.finditer(pattern, content):
                                    # Check if match contains exclusion patterns
                                    excluded = False
                                    for exclusion in exclusion_patterns:
                                        if re.search(exclusion, content[max(0, match.start() - 50):min(len(content), match.end() + 50)], re.IGNORECASE):
                                            excluded = True
                                            break
                                            
                                    if not excluded:
                                        line_num = content[:match.start()].count('\n') + 1
                                        line = content.splitlines()[line_num - 1]
                                        results.append((file_path, line_num, vulnerability_type, line.strip()))
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    
    return results

if __name__ == "__main__":
    # Focus on specific directories
    directories = [
        "ui-service",
        "api-gateway",
        "trading-gateway-service",
        "analysis-engine-service",
        "data-management-service",
        "feature-store-service",
        "ml-integration-service",
        "ml-workbench-service",
        "strategy-execution-engine",
        "portfolio-management-service",
        "risk-management-service",
        "monitoring-alerting-service",
        "model-registry-service",
        "common-js-lib"
    ]
    
    results = find_xss_vulnerabilities(directories)
    
    print(f"Found {len(results)} potential XSS vulnerabilities:")
    
    # Group by file
    files = {}
    for file_path, line_number, vulnerability_type, line in results:
        if file_path not in files:
            files[file_path] = []
        files[file_path].append((line_number, vulnerability_type, line))
    
    # Print results
    for file_path, vulnerabilities in files.items():
        print(f"\n{file_path}:")
        for line_number, vulnerability_type, line in vulnerabilities:
            print(f"  Line {line_number}: {vulnerability_type}")
            print(f"  {line}")
            print("  " + "-" * 50)