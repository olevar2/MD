"""
Phase 3 Comprehensive Test Runner
Coordinates and executes all system testing components
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any
import json
import os

from stress_testing.stress_test_coordinator import StressTestCoordinator, run_stress_test_suite
from integration_testing.integration_test_coordinator import IntegrationTestCoordinator, run_integration_tests
from system_validation.system_validator import SystemValidator, run_system_validation

logger = logging.getLogger(__name__)

class Phase3TestRunner:
    """
    Coordinates and runs all Phase 3 system tests
    """
    
    def __init__(self, output_dir: str = None):
        """Initialize the test runner"""
        self.output_dir = output_dir or os.path.join("test_results", f"phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 3 tests"""
        logger.info("Starting Phase 3 comprehensive system testing")
        start_time = datetime.now()
        
        try:
            # Run stress tests
            logger.info("Running stress tests...")
            stress_results = await run_stress_test_suite()
            
            # Run integration tests
            logger.info("Running integration tests...")
            integration_results = await run_integration_tests()
            
            # Run system validation
            logger.info("Running system validation...")
            validation_results = await run_system_validation()
            
            # Generate comprehensive report
            report = self._generate_report(
                start_time=start_time,
                end_time=datetime.now(),
                stress_results=stress_results,
                integration_results=integration_results,
                validation_results=validation_results
            )
            
            # Save detailed results
            self._save_results(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error during Phase 3 testing: {str(e)}")
            raise
            
    def _generate_report(self, 
                        start_time: datetime,
                        end_time: datetime,
                        stress_results: List[Any],
                        integration_results: List[Any],
                        validation_results: List[Any]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        # Calculate stress test metrics
        stress_passed = sum(1 for r in stress_results if r.passed)
        stress_total = len(stress_results)
        
        # Calculate integration test metrics
        integration_passed = sum(1 for r in integration_results if r.status == "passed")
        integration_total = len(integration_results)
        
        # Calculate validation metrics
        validation_passed = sum(1 for r in validation_results if r.passed)
        validation_total = len(validation_results)
        
        report = {
            "summary": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "total_tests": stress_total + integration_total + validation_total,
                "total_passed": stress_passed + integration_passed + validation_passed,
                "success_rate": (stress_passed + integration_passed + validation_passed) / 
                               (stress_total + integration_total + validation_total)
            },
            "stress_testing": {
                "total": stress_total,
                "passed": stress_passed,
                "success_rate": stress_passed / stress_total if stress_total > 0 else 0,
                "results": [self._format_stress_result(r) for r in stress_results]
            },
            "integration_testing": {
                "total": integration_total,
                "passed": integration_passed,
                "success_rate": integration_passed / integration_total if integration_total > 0 else 0,
                "results": [self._format_integration_result(r) for r in integration_results]
            },
            "system_validation": {
                "total": validation_total,
                "passed": validation_passed,
                "success_rate": validation_passed / validation_total if validation_total > 0 else 0,
                "results": [self._format_validation_result(r) for r in validation_results]
            }
        }
        
        return report
        
    def _format_stress_result(self, result: Any) -> Dict[str, Any]:
        """Format stress test result for reporting"""
        return {
            "scenario_name": result.scenario_name,
            "passed": result.passed,
            "metrics": result.metrics,
            "bottlenecks": result.bottlenecks,
            "failure_reasons": result.failure_reasons if not result.passed else []
        }
        
    def _format_integration_result(self, result: Any) -> Dict[str, Any]:
        """Format integration test result for reporting"""
        return {
            "test_name": result.test_name,
            "component": result.component_name,
            "status": result.status,
            "metrics": result.metrics,
            "error": result.error_message if result.status == "failed" else None
        }
        
    def _format_validation_result(self, result: Any) -> Dict[str, Any]:
        """Format validation result for reporting"""
        return {
            "validation_type": result.validation_type,
            "passed": result.passed,
            "metrics": result.metrics,
            "issues": result.issues,
            "recommendations": result.recommendations
        }
        
    def _save_results(self, report: Dict[str, Any]):
        """Save test results to files"""
        # Save summary report
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(report["summary"], f, indent=2)
            
        # Save detailed results
        details_path = os.path.join(self.output_dir, "detailed_results.json")
        with open(details_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Test results saved to {self.output_dir}")

async def main():
    """Main entry point for Phase 3 testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    runner = Phase3TestRunner()
    
    try:
        results = await runner.run_all_tests()
        
        # Log summary
        summary = results["summary"]
        logger.info(f"Phase 3 testing completed:")
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Passed tests: {summary['total_passed']}")
        logger.info(f"Success rate: {summary['success_rate']*100:.1f}%")
        logger.info(f"Duration: {summary['duration_seconds']:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Phase 3 testing failed: {str(e)}")
        raise

if __name__ == '__main__':
    asyncio.run(main())
