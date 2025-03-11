"""
Tests for utility helper functions.
"""

import unittest
import sys
from pathlib import Path
import time

# Add src directory to path for imports
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.helpers import format_time, time_function


class TestHelperFunctions(unittest.TestCase):
    """Test cases for helper functions."""

    def test_format_time_microseconds(self):
        """Test formatting time in microseconds."""
        self.assertEqual(format_time(0.0005), "500.00 µs")

    def test_format_time_milliseconds(self):
        """Test formatting time in milliseconds."""
        self.assertEqual(format_time(0.5), "500.00 ms")

    def test_format_time_seconds(self):
        """Test formatting time in seconds."""
        self.assertEqual(format_time(2.5), "2.50 s")

    def test_time_function_decorator(self):
        """Test the time_function decorator."""
        
        @time_function
        def slow_function():
            time.sleep(0.1)
            return "done"
        
        result = slow_function()
        self.assertEqual(result, "done")


if __name__ == "__main__":
    unittest.main()