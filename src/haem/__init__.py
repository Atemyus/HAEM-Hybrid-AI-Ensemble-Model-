"""
HAEM - Hybrid AI-NWP Ensemble Model
====================================

A sophisticated meteorological analysis engine combining multiple NWP models
with AI-based validation and ensemble forecasting capabilities.

This system generates physically consistent, explainable, and graphically
plottable meteorological forecasts derived from:
- Multi-model numerical data (ECMWF, GFS, UKMO, GEM, ICON)
- Independent AI-based meteorological reasoning
- Dynamic confidence scoring
- Ensemble mathematical aggregation
"""

__version__ = "1.0.0"
__author__ = "HAEM Team"

from haem.engine import WeatherAnalysisEngine

__all__ = ["WeatherAnalysisEngine", "__version__"]
