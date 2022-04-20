"""Generative neural network models for multi-level free energy calculations"""

# Add imports here
from .multifree import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
