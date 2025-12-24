"""Tests for HAEM analysis modules."""

import numpy as np
import pytest
from datetime import datetime

from haem.models.meteorological import (
    NWPModel,
    ForecastRun,
    GeopotentialField,
    PressureField,
    TemperatureField,
    PressureLevel,
    ModelData,
    SynopticPatternType,
)
from haem.analysis.pattern_detector import PatternDetector
from haem.analysis.confidence import ConfidenceAssessor
from haem.analysis.interpreter import PhysicalInterpreter


class TestPatternDetector:
    """Tests for PatternDetector."""

    @pytest.fixture
    def detector(self):
        """Create pattern detector."""
        return PatternDetector()

    @pytest.fixture
    def sample_z500_with_trough(self):
        """Create Z500 field with a clear trough pattern."""
        lats = np.linspace(30, 70, 81)
        lons = np.linspace(-60, 40, 201)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Base height with meridional gradient
        data = 560 - (lat_grid - 50) * 0.5

        # Add trough at 0°E
        trough_lon = 0
        data -= 8 * np.exp(-((lon_grid - trough_lon) ** 2) / 200)

        return GeopotentialField(
            level=PressureLevel.L500,
            valid_time=datetime(2024, 1, 15, 12, 0, 0),
            forecast_hour=24,
            lats=lats,
            lons=lons,
            data=data,
            source_model=NWPModel.ECMWF,
            run_time=datetime(2024, 1, 14, 12, 0, 0),
        )

    def test_detect_patterns(self, detector, sample_z500_with_trough):
        """Test pattern detection."""
        patterns = detector.detect_all_patterns(z500=sample_z500_with_trough)

        # Should detect some patterns
        assert len(patterns) > 0

    def test_empty_input(self, detector):
        """Test with no input fields."""
        patterns = detector.detect_all_patterns()
        assert patterns == []


class TestConfidenceAssessor:
    """Tests for ConfidenceAssessor."""

    @pytest.fixture
    def assessor(self):
        """Create confidence assessor."""
        return ConfidenceAssessor()

    @pytest.fixture
    def sample_model_data(self):
        """Create sample model data."""
        run = ForecastRun(
            model=NWPModel.ECMWF,
            init_time=datetime(2024, 1, 15, 12, 0, 0),
        )

        lats = np.linspace(30, 60, 31)
        lons = np.linspace(-30, 30, 61)
        data = np.random.uniform(540, 580, (31, 61))

        model_data = ModelData(run=run)
        model_data.z500[24] = GeopotentialField(
            level=PressureLevel.L500,
            valid_time=datetime(2024, 1, 16, 12, 0, 0),
            forecast_hour=24,
            lats=lats,
            lons=lons,
            data=data,
            source_model=NWPModel.ECMWF,
            run_time=run.init_time,
        )

        return model_data

    def test_assess_confidence(self, assessor, sample_model_data):
        """Test confidence assessment."""
        confidence = assessor.assess_confidence(
            model_data=sample_model_data,
            forecast_hour=24,
            patterns=[],
        )

        assert 0 <= confidence.overall_score <= 100
        assert 0 <= confidence.temporal_consistency <= 100
        assert 0 <= confidence.spatial_coherence <= 100

    def test_assess_reliability(self, assessor, sample_model_data):
        """Test reliability assessment."""
        reliability = assessor.assess_reliability(
            model=NWPModel.ECMWF,
            model_data=sample_model_data,
            forecast_hour=24,
        )

        assert 0 <= reliability.reliability_score <= 100
        assert reliability.model == NWPModel.ECMWF
        assert len(reliability.strengths) > 0


class TestPhysicalInterpreter:
    """Tests for PhysicalInterpreter."""

    @pytest.fixture
    def interpreter(self):
        """Create physical interpreter."""
        return PhysicalInterpreter()

    @pytest.fixture
    def sample_fields(self):
        """Create sample atmospheric fields."""
        lats = np.linspace(30, 70, 81)
        lons = np.linspace(-60, 40, 201)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        z500_data = 560 - (lat_grid - 50) * 0.5
        t850_data = 280 - (lat_grid - 50) * 0.3

        run_time = datetime(2024, 1, 14, 12, 0, 0)
        valid_time = datetime(2024, 1, 15, 12, 0, 0)

        z500 = GeopotentialField(
            level=PressureLevel.L500,
            valid_time=valid_time,
            forecast_hour=24,
            lats=lats,
            lons=lons,
            data=z500_data,
            source_model=NWPModel.ECMWF,
            run_time=run_time,
        )

        t850 = TemperatureField(
            level=PressureLevel.L850,
            valid_time=valid_time,
            forecast_hour=24,
            lats=lats,
            lons=lons,
            data=t850_data,
            source_model=NWPModel.ECMWF,
            run_time=run_time,
        )

        return z500, t850

    def test_interpret(self, interpreter, sample_fields):
        """Test physical interpretation."""
        z500, t850 = sample_fields

        interpretation = interpreter.interpret(
            z500=z500,
            t850=t850,
            patterns=[],
            forecast_hour=24,
        )

        assert interpretation.dominant_pattern is not None
        assert interpretation.primary_mechanism is not None
        assert len(interpretation.causal_explanation) > 0
        assert len(interpretation.evolution_description) > 0
