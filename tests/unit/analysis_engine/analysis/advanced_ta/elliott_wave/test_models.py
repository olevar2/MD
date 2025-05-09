"""
Unit tests for Elliott Wave models module.
"""

import unittest
from analysis_engine.analysis.advanced_ta.elliott_wave.models import (
    WaveType, WavePosition, WaveDegree
)


class TestElliottWaveModels(unittest.TestCase):
    """Test cases for Elliott Wave models."""
    
    def test_wave_type_enum(self):
        """Test WaveType enum values."""
        self.assertEqual(WaveType.IMPULSE.value, "impulse")
        self.assertEqual(WaveType.CORRECTION.value, "correction")
        self.assertEqual(WaveType.DIAGONAL.value, "diagonal")
        self.assertEqual(WaveType.EXTENSION.value, "extension")
        self.assertEqual(WaveType.TRIANGLE.value, "triangle")
        self.assertEqual(WaveType.UNKNOWN.value, "unknown")
    
    def test_wave_position_enum(self):
        """Test WavePosition enum values."""
        # Impulse wave positions
        self.assertEqual(WavePosition.ONE.value, "1")
        self.assertEqual(WavePosition.TWO.value, "2")
        self.assertEqual(WavePosition.THREE.value, "3")
        self.assertEqual(WavePosition.FOUR.value, "4")
        self.assertEqual(WavePosition.FIVE.value, "5")
        
        # Corrective wave positions
        self.assertEqual(WavePosition.A.value, "A")
        self.assertEqual(WavePosition.B.value, "B")
        self.assertEqual(WavePosition.C.value, "C")
        self.assertEqual(WavePosition.D.value, "D")
        self.assertEqual(WavePosition.E.value, "E")
        
        # Sub-waves
        self.assertEqual(WavePosition.SUB_ONE.value, "i")
        self.assertEqual(WavePosition.SUB_TWO.value, "ii")
        self.assertEqual(WavePosition.SUB_THREE.value, "iii")
        self.assertEqual(WavePosition.SUB_FOUR.value, "iv")
        self.assertEqual(WavePosition.SUB_FIVE.value, "v")
        
        # Sub-corrective waves
        self.assertEqual(WavePosition.SUB_A.value, "a")
        self.assertEqual(WavePosition.SUB_B.value, "b")
        self.assertEqual(WavePosition.SUB_C.value, "c")
        self.assertEqual(WavePosition.SUB_D.value, "d")
        self.assertEqual(WavePosition.SUB_E.value, "e")
    
    def test_wave_degree_enum(self):
        """Test WaveDegree enum values."""
        self.assertEqual(WaveDegree.GRAND_SUPERCYCLE.value, "Grand Supercycle")
        self.assertEqual(WaveDegree.SUPERCYCLE.value, "Supercycle")
        self.assertEqual(WaveDegree.CYCLE.value, "Cycle")
        self.assertEqual(WaveDegree.PRIMARY.value, "Primary")
        self.assertEqual(WaveDegree.INTERMEDIATE.value, "Intermediate")
        self.assertEqual(WaveDegree.MINOR.value, "Minor")
        self.assertEqual(WaveDegree.MINUTE.value, "Minute")
        self.assertEqual(WaveDegree.MINUETTE.value, "Minuette")
        self.assertEqual(WaveDegree.SUBMINUETTE.value, "Subminuette")


if __name__ == "__main__":
    unittest.main()