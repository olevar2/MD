"""
Comprehensive test reporting for E2E tests.
Generates detailed reports with failure analysis and metrics.
"""
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class TestReporter:
    """
    Generates detailed reports for E2E tests.
    Provides comprehensive test result reporting with failure analysis.
    """
    
    def __init__(self, report_dir: Optional[str] = None):
        """
        Initialize the test reporter.
        
        Args:
            report_dir: Directory to store test reports
        """
        if report_dir is None:
            # Use default reports directory if not specified
            base_dir = Path(__file__).parent.parent.parent
            report_dir = str(base_dir / "test_reports" / "e2e")
            
        self.report_dir = report_dir
        self.results: List[Dict[str, Any]] = []
        self.test_run_id = f"run_{int(time.time())}"
        
        # Create report directory
        os.makedirs(self.report_dir, exist_ok=True)
        logger.info(f"Test reports will be saved to: {self.report_dir}")
        
        # Create artifacts directory
        self.artifacts_dir = os.path.join(self.report_dir, self.test_run_id, "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
    def add_test_result(
        self,
        test_name: str,
        status: str,
        duration: float,
        environment: str,
        logs: Dict[str, str] = None,
        artifacts: Dict[str, Any] = None,
        error_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a test result to the report.
        
        Args:
            test_name: Name of the test
            status: Test status (PASS, FAIL, SKIP)
            duration: Test duration in seconds
            environment: Test environment mode
            logs: Dictionary of service logs
            artifacts: Dictionary of test artifacts
            error_details: Detailed error information for failed tests
        """
        timestamp = datetime.now().isoformat()
        
        result = {
            "test_name": test_name,
            "status": status,
            "duration": duration,
            "timestamp": timestamp,
            "environment": environment,
        }
        
        # Process logs
        if logs:
            # Store logs in artifacts directory
            log_paths = {}
            for service_name, log_content in logs.items():
                log_path = f"{self.artifacts_dir}/{test_name}_{service_name}.log"
                with open(log_path, "w") as f:
                    f.write(log_content)
                log_paths[service_name] = os.path.relpath(log_path, self.report_dir)
            result["logs"] = log_paths
        
        # Process artifacts
        if artifacts:
            artifact_paths = {}
            for artifact_name, artifact_content in artifacts.items():
                if isinstance(artifact_content, str):
                    # String artifact
                    artifact_path = f"{self.artifacts_dir}/{test_name}_{artifact_name}.txt"
                    with open(artifact_path, "w") as f:
                        f.write(artifact_content)
                elif isinstance(artifact_content, dict) or isinstance(artifact_content, list):
                    # JSON-serializable artifact
                    artifact_path = f"{self.artifacts_dir}/{test_name}_{artifact_name}.json"
                    with open(artifact_path, "w") as f:
                        json.dump(artifact_content, f, indent=2)
                elif isinstance(artifact_content, bytes):
                    # Binary artifact
                    artifact_path = f"{self.artifacts_dir}/{test_name}_{artifact_name}.bin"
                    with open(artifact_path, "wb") as f:
                        f.write(artifact_content)
                elif hasattr(artifact_content, "read"):
                    # File-like artifact
                    artifact_path = f"{self.artifacts_dir}/{test_name}_{artifact_name}"
                    with open(artifact_path, "wb") as f:
                        shutil.copyfileobj(artifact_content, f)
                else:
                    logger.warning(f"Unsupported artifact type for {artifact_name}: {type(artifact_content)}")
                    continue
                    
                artifact_paths[artifact_name] = os.path.relpath(artifact_path, self.report_dir)
                
            result["artifacts"] = artifact_paths
        
        # Add error details for failed tests
        if status == "FAIL" and error_details:
            result["error_details"] = error_details
            
            # Perform failure analysis if error details available
            result["failure_analysis"] = self._analyze_failure(error_details, logs or {})
        
        # Store the result
        self.results.append(result)
        
        # Create an individual test report
        self._generate_test_report(result)
        
        # Update the overall report summary
        self._generate_summary_report()
        
    def _analyze_failure(
        self, 
        error_details: Dict[str, Any],
        logs: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze test failure to determine root cause.
        
        Args:
            error_details: Detailed error information
            logs: Service logs
            
        Returns:
            Dictionary with failure analysis results
        """
        analysis = {
            "possible_causes": [],
            "suggested_actions": [],
            "related_services": []
        }
        
        # Extract error message and stack trace
        error_msg = error_details.get("message", "")
        stack_trace = error_details.get("stack_trace", "")
        
        # Look for common error patterns in the error message and stack trace
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            analysis["possible_causes"].append("Service response timeout")
            analysis["suggested_actions"].append("Check service performance and resource usage")
            
        if "connection refused" in error_msg.lower():
            analysis["possible_causes"].append("Service unavailable or not running")
            analysis["suggested_actions"].append("Verify service is running and accessible")
            
        if "permission denied" in error_msg.lower() or "unauthorized" in error_msg.lower():
            analysis["possible_causes"].append("Authentication or authorization issue")
            analysis["suggested_actions"].append("Check credentials and permissions")
            
        # Search for service names in the error message and stack trace
        service_names = [
            "market_data_provider",
            "exchange_connector",
            "order_service",
            "position_service",
            "portfolio_service",
            "strategy_execution_engine",
            "risk_management_service",
            "notification_service",
            "ml_integration_service"
        ]
        
        for service in service_names:
            if service in error_msg.lower() or service in stack_trace.lower():
                analysis["related_services"].append(service)
                
        # Analyze logs for errors around the failure time
        for service_name, log_content in logs.items():
            if "error" in log_content.lower() or "exception" in log_content.lower():
                if service_name not in analysis["related_services"]:
                    analysis["related_services"].append(service_name)
        
        # If we couldn't determine specific causes, add generic recommendations
        if not analysis["possible_causes"]:
            analysis["possible_causes"].append("Unknown error")
            analysis["suggested_actions"].append("Check service logs for detailed error messages")
            analysis["suggested_actions"].append("Verify test environment configuration")
            
        return analysis
        
    def _generate_test_report(self, result: Dict[str, Any]) -> None:
        """
        Generate a report for an individual test.
        
        Args:
            result: Test result data
        """
        test_name = result["test_name"]
        status = result["status"]
        
        # Create report directory
        report_dir = os.path.join(self.report_dir, self.test_run_id)
        os.makedirs(report_dir, exist_ok=True)
        
        # Save detailed report
        report_path = os.path.join(report_dir, f"{test_name}.json")
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Test report for {test_name} saved to {report_path}")
        
    def _generate_summary_report(self) -> None:
        """Generate a summary report of all test results."""
        report_dir = os.path.join(self.report_dir, self.test_run_id)
        os.makedirs(report_dir, exist_ok=True)
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        skipped = sum(1 for r in self.results if r["status"] == "SKIP")
        
        total_duration = sum(r["duration"] for r in self.results)
        
        # Group tests by status
        passed_tests = [r["test_name"] for r in self.results if r["status"] == "PASS"]
        failed_tests = [r["test_name"] for r in self.results if r["status"] == "FAIL"]
        skipped_tests = [r["test_name"] for r in self.results if r["status"] == "SKIP"]
        
        # Create summary report
        summary = {
            "test_run_id": self.test_run_id,
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": passed / total_tests if total_tests > 0 else 0,
            "total_duration": total_duration,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "environment": self.results[0]["environment"] if self.results else "unknown"
        }
        
        # Save summary report
        summary_path = os.path.join(report_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        # Generate HTML report
        self._generate_html_report(summary, report_dir)
        
        logger.info(f"Test summary report saved to {summary_path}")
        logger.info(f"Pass rate: {summary['pass_rate'] * 100:.2f}% ({passed}/{total_tests})")
        
    def _generate_html_report(self, summary: Dict[str, Any], report_dir: str) -> None:
        """
        Generate an HTML report from the test results.
        
        Args:
            summary: Summary statistics
            report_dir: Directory to save the report
        """
        # Create a basic HTML report
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>Forex Trading Platform E2E Test Report</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        h1 { color: #333; }",
            "        .summary { margin-bottom: 20px; }",
            "        .pass { color: green; }",
            "        .fail { color: red; }",
            "        .skip { color: orange; }",
            "        table { border-collapse: collapse; width: 100%; }",
            "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "        tr:nth-child(even) { background-color: #f2f2f2; }",
            "        th { background-color: #4CAF50; color: white; }",
            "    </style>",
            "</head>",
            "<body>",
            f"    <h1>E2E Test Report - {summary['test_run_id']}</h1>",
            "    <div class='summary'>",
            f"        <p><strong>Environment:</strong> {summary['environment']}</p>",
            f"        <p><strong>Total Tests:</strong> {summary['total_tests']}</p>",
            f"        <p class='pass'><strong>Passed:</strong> {summary['passed']}</p>",
            f"        <p class='fail'><strong>Failed:</strong> {summary['failed']}</p>",
            f"        <p class='skip'><strong>Skipped:</strong> {summary['skipped']}</p>",
            f"        <p><strong>Pass Rate:</strong> {summary['pass_rate'] * 100:.2f}%</p>",
            f"        <p><strong>Total Duration:</strong> {summary['total_duration']:.2f} seconds</p>",
            "    </div>"
        ]
        
        # Add detailed results table
        html.extend([
            "    <h2>Test Details</h2>",
            "    <table>",
            "        <tr>",
            "            <th>Test Name</th>",
            "            <th>Status</th>",
            "            <th>Duration (s)</th>",
            "            <th>Artifacts</th>",
            "        </tr>"
        ])
        
        # Add rows for each test
        for result in self.results:
            status_class = "pass" if result["status"] == "PASS" else "fail" if result["status"] == "FAIL" else "skip"
            artifacts_links = []
            
            if "artifacts" in result:
                for name, path in result["artifacts"].items():
                    artifacts_links.append(f"<a href='{path}'>{name}</a>")
                    
            if "logs" in result:
                for name, path in result["logs"].items():
                    artifacts_links.append(f"<a href='{path}'>{name} logs</a>")
                    
            artifacts_html = ", ".join(artifacts_links) if artifacts_links else "None"
            
            html.append(f"        <tr>")
            html.append(f"            <td>{result['test_name']}</td>")
            html.append(f"            <td class='{status_class}'>{result['status']}</td>")
            html.append(f"            <td>{result['duration']:.2f}</td>")
            html.append(f"            <td>{artifacts_html}</td>")
            html.append(f"        </tr>")
            
        html.extend([
            "    </table>",
            "</body>",
            "</html>"
        ])
        
        # Save HTML report
        html_path = os.path.join(report_dir, "report.html")
        with open(html_path, "w") as f:
            f.write("\n".join(html))
            
        logger.info(f"HTML test report saved to {html_path}")
        
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a summary of test results.
        
        Returns:
            Summary statistics
        """
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        skipped = sum(1 for r in self.results if r["status"] == "SKIP")
        
        return {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": passed / total_tests if total_tests > 0 else 0
        }
