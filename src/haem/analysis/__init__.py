"""Analysis modules for HAEM meteorological assessment."""

from haem.analysis.model_analyzer import ModelAnalyzer
from haem.analysis.pattern_detector import PatternDetector
from haem.analysis.comparison import MultiModelComparator
from haem.analysis.interpreter import PhysicalInterpreter
from haem.analysis.confidence import ConfidenceAssessor

__all__ = [
    "ModelAnalyzer",
    "PatternDetector",
    "MultiModelComparator",
    "PhysicalInterpreter",
    "ConfidenceAssessor",
]
