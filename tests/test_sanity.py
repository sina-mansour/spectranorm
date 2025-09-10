"""
tests/test_sanity.py

Sanity tests for the Spectranorm package.
"""

from spectranorm import snm

# ----- Basic import / attribute tests -----


def test_snm_import() -> None:
    # Check that the main classes can be imported
    assert hasattr(snm, "SpectralNormativeModel")
    assert hasattr(snm, "DirectNormativeModel")
