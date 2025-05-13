"""
Tests for the Integrated Recovery System.
"""
import unittest
from datetime import datetime
import tempfile
import shutil
from pathlib import Path
import json
from core.integrated_recovery import (
    IntegratedRecoverySystem,
    RecoveryStrategy,
    RecoveryPriority,
    RecoveryResult
)

class TestIntegratedRecoverySystem(unittest.TestCase):
    """Test suite for the IntegratedRecoverySystem"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.recovery_system = IntegratedRecoverySystem(state_dir=self.temp_dir)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir)

    def test_state_recovery(self):
        """Test state-based recovery."""
        component = "trading_engine"
        
        # Create a mock error and context
        error = Exception("State corruption detected")
        context = {'is_critical': True}
        
        # Set up initial component state
        self.recovery_system.component_states[component] = {
            'version': '1.0',
            'timestamp': datetime.utcnow().isoformat(),
            'checksum': 'abc123',
            'status': 'running'
        }
        
        # Save the state
        self.recovery_system._save_component_state(component)
        
        # Attempt recovery
        result = self.recovery_system.attempt_recovery(
            component=component,
            error=error,
            context=context
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.strategy, RecoveryStrategy.STATE_RECOVERY)
        self.assertIn('Successfully recovered state', result.message)

    def test_component_restart(self):
        """Test component restart recovery."""
        component = "data_pipeline"
        error = Exception("Connection timeout")
        
        result = self.recovery_system.attempt_recovery(
            component=component,
            error=error,
            context={'is_critical': False}
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.strategy, RecoveryStrategy.COMPONENT_RESTART)
        self.assertIn('Successfully restarted', result.message)

    def test_data_sync_recovery(self):
        """Test data synchronization recovery."""
        component = "feature_store"
        error = Exception("Data inconsistency detected")
        
        result = self.recovery_system.attempt_recovery(
            component=component,
            error=error,
            context={'is_critical': False},
            sync_source="backup_store"
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.strategy, RecoveryStrategy.DATA_SYNC)
        self.assertIn('Successfully synchronized', result.message)

    def test_fallback_recovery(self):
        """Test fallback recovery."""
        component = "monitoring"
        error = Exception("Non-critical error")
        fallback_config = {
            'mode': 'minimal',
            'features': ['essential_only']
        }
        
        result = self.recovery_system.attempt_recovery(
            component=component,
            error=error,
            context={'is_critical': False},
            fallback_config=fallback_config
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.strategy, RecoveryStrategy.FALLBACK)
        self.assertIn('Successfully applied fallback', result.message)

    def test_recovery_priority_assessment(self):
        """Test recovery priority assessment."""
        # Test critical error
        priority = self.recovery_system._assess_recovery_priority(
            "trading_engine",
            Exception("data_loss detected"),
            {'is_critical': True}
        )
        self.assertEqual(priority, RecoveryPriority.CRITICAL)
        
        # Test high priority component
        priority = self.recovery_system._assess_recovery_priority(
            "risk_manager",
            Exception("general error"),
            None
        )
        self.assertEqual(priority, RecoveryPriority.HIGH)
        
        # Test low priority component
        priority = self.recovery_system._assess_recovery_priority(
            "monitoring",
            Exception("general error"),
            None
        )
        self.assertEqual(priority, RecoveryPriority.MEDIUM)

    def test_recovery_strategy_determination(self):
        """Test recovery strategy determination."""
        # Test state corruption error
        strategy = self.recovery_system._determine_recovery_strategy(
            "trading_engine",
            Exception("state corruption detected"),
            RecoveryPriority.CRITICAL
        )
        self.assertEqual(strategy, RecoveryStrategy.STATE_RECOVERY)
        
        # Test connection error
        strategy = self.recovery_system._determine_recovery_strategy(
            "data_pipeline",
            Exception("connection timeout"),
            RecoveryPriority.HIGH
        )
        self.assertEqual(strategy, RecoveryStrategy.COMPONENT_RESTART)

    def test_state_persistence(self):
        """Test component state persistence."""
        component = "test_component"
        state = {
            'version': '1.0',
            'timestamp': datetime.utcnow().isoformat(),
            'checksum': 'abc123',
            'status': 'running'
        }
        
        # Save state
        self.recovery_system.component_states[component] = state
        self.recovery_system._save_component_state(component)
        
        # Load state
        loaded_state = self.recovery_system._load_component_state(component)
        self.assertEqual(loaded_state['version'], state['version'])
        self.assertEqual(loaded_state['status'], state['status'])

    def test_recovery_summary(self):
        """Test recovery summary generation."""
        # Perform multiple recoveries
        components = ['trading_engine', 'data_pipeline', 'monitoring']
        for component in components:
            self.recovery_system.attempt_recovery(
                component=component,
                error=Exception(f"Test error for {component}"),
                context={'is_critical': False}
            )
        
        summary = self.recovery_system.get_recovery_summary()
        self.assertIn('total_recoveries', summary)
        self.assertIn('success_rate', summary)
        self.assertIn('by_strategy', summary)
        self.assertIn('by_component', summary)
        self.assertEqual(summary['total_recoveries'], len(components))
