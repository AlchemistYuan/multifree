"""
Unit and regression test for the multifree package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import multifree


def test_multifree_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "multifree" in sys.modules
