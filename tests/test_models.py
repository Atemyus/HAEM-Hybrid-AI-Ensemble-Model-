"""Tests for HAEM meteorological models."""

import numpy as np
import pytest
from datetime import datetime

from haem.models.meteorological import (
    NWPModel,
    ForecastRun,
    GeopotentialField,
    TemperatureField,
    PressureField,
    SynopticPattern,
    SynopticPatternType,
    PressureLevel,
    ModelData,
)


class TestNWPModel:
    """Tests for NWPModel enum."""

    def test_model_values(self):
        """Test that all expected models are present."""
        assert NWPModel.ECMWF.value == "ecmwf"
        assert NWPModel.GFS.value == "gfs"
        assert NWPModel.UKMO.value == "ukmo"
        assert NWPModel.GEM.value == "gem"
        assert NWPModel.ICON.value == "icon"

    def test_full_name(self):
        """Test full name property."""
        assert "European" in NWPModel.ECMWF.full_name
        assert "Global Forecast" in NWPModel.GFS.full_name

    def test_resolution(self):
        """Test resolution property."""
        assert "km" in NWPModel.ECMWF.resolution
        assert "km" in NWPModel.GFS.resolution


class TestForecastRun:
    """Tests for ForecastRun model."""

    def test_valid_init_time(self):
        """Test valid synoptic init times."""
        run = ForecastRun(
            model=NWPModel.ECMWF,
            init_time=datetime(2024, 1, 15, 12, 0, 0),
        )
        assert run.init_time.hour == 12

    def test_invalid_init_time(self):
        """Test that non-synoptic times are rejected."""
        with pytest.raises(ValueError):
            ForecastRun(
                model=NWPModel.ECMWF,
                init_time=datetime(2024, 1, 15, 15, 0, 0),  # 15Z is invalid
            )

    def test_get_lead_time(self):
        """Test lead time calculation."""
        run = ForecastRun(
            model=NWPModel.GFS,
            init_time=datetime(2024, 1, 15, 0, 0, 0),
        )
        valid_time = datetime(2024, 1, 16, 12, 0, 0)
        assert run.get_lead_time(valid_time) == 36


class TestGeopotentialField:
    """Tests for GeopotentialField."""

    @pytest.fixture
    def sample_field(self):
        """Create a sample geopotential field."""
        lats = np.linspace(30, 60, 31)
        lons = np.linspace(-30, 30, 61)
        data = np.random.uniform(540, 580, (31, 61))

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

    def test_shape(self, sample_field):
        """Test shape property."""
        assert sample_field.shape == (31, 61)

    def test_domain(self, sample_field):
        """Test domain property."""
        domain = sample_field.domain
        assert domain["lat_min"] == 30
        assert domain["lat_max"] == 60
        assert domain["lon_min"] == -30
        assert domain["lon_max"] == 30

    def test_in_meters(self, sample_field):
        """Test conversion to meters."""
        meters = sample_field.in_meters
        assert np.allclose(meters, sample_field.data * 10)

    def test_interpolate_to_point(self, sample_field):
        """Test point interpolation."""
        value = sample_field.interpolate_to_point(45.0, 0.0)
        assert 540 <= value <= 580


class TestPressureField:
    """Tests for PressureField."""

    @pytest.fixture
    def sample_slp(self):
        """Create a sample SLP field with a low and high."""
        lats = np.linspace(30, 60, 31)
        lons = np.linspace(-30, 30, 61)

        # Base pressure
        data = np.ones((31, 61)) * 1013.0

        # Add a low (center at lat=45, lon=0)
        lat_idx, lon_idx = 15, 30
        for i in range(31):
            for j in range(61):
                dist = np.sqrt((i - lat_idx)**2 + (j - lon_idx)**2)
                data[i, j] -= 20 * np.exp(-dist**2 / 50)

        # Add a high (center at lat=50, lon=20)
        lat_idx, lon_idx = 20, 50
        for i in range(31):
            for j in range(61):
                dist = np.sqrt((i - lat_idx)**2 + (j - lon_idx)**2)
                data[i, j] += 15 * np.exp(-dist**2 / 50)

        return PressureField(
            valid_time=datetime(2024, 1, 15, 12, 0, 0),
            forecast_hour=24,
            lats=lats,
            lons=lons,
            data=data,
            source_model=NWPModel.ECMWF,
            run_time=datetime(2024, 1, 14, 12, 0, 0),
        )

    def test_find_pressure_centers(self, sample_slp):
        """Test pressure center detection."""
        centers = sample_slp.find_pressure_centers(
            threshold_low=1010,
            threshold_high=1020,
        )

        assert "lows" in centers
        assert "highs" in centers
        # Should find at least one low
        assert len(centers["lows"]) >= 1


class TestSynopticPattern:
    """Tests for SynopticPattern."""

    def test_pattern_creation(self):
        """Test creating a synoptic pattern."""
        pattern = SynopticPattern(
            pattern_type=SynopticPatternType.TROUGH,
            center_lat=50.0,
            center_lon=-10.0,
            intensity=8.5,
            extent_km=1200.0,
            valid_time=datetime(2024, 1, 15, 12, 0, 0),
        )

        assert pattern.pattern_type == SynopticPatternType.TROUGH
        assert pattern.center_lat == 50.0
        assert "trough" in pattern.description.lower()

    def test_pattern_with_movement(self):
        """Test pattern with movement attributes."""
        pattern = SynopticPattern(
            pattern_type=SynopticPatternType.CUT_OFF_LOW,
            center_lat=45.0,
            center_lon=5.0,
            intensity=12.0,
            extent_km=800.0,
            valid_time=datetime(2024, 1, 15, 12, 0, 0),
            movement_speed_kmh=15.0,
            movement_direction_deg=90.0,
        )

        assert pattern.movement_speed_kmh == 15.0
        assert pattern.movement_direction_deg == 90.0


class TestModelData:
    """Tests for ModelData container."""

    @pytest.fixture
    def sample_model_data(self):
        """Create sample model data."""
        run = ForecastRun(
            model=NWPModel.ECMWF,
            init_time=datetime(2024, 1, 15, 12, 0, 0),
            forecast_hours=[0, 6, 12, 24],
        )

        lats = np.linspace(30, 60, 31)
        lons = np.linspace(-30, 30, 61)

        model_data = ModelData(run=run)

        for hour in [0, 6, 12, 24]:
            data = np.random.uniform(540, 580, (31, 61))
            model_data.z500[hour] = GeopotentialField(
                level=PressureLevel.L500,
                valid_time=run.init_time,
                forecast_hour=hour,
                lats=lats,
                lons=lons,
                data=data,
                source_model=NWPModel.ECMWF,
                run_time=run.init_time,
            )

        return model_data

    def test_get_field(self, sample_model_data):
        """Test field retrieval."""
        field = sample_model_data.get_field("z500", 12)
        assert field is not None
        assert field.forecast_hour == 12

    def test_available_hours(self, sample_model_data):
        """Test available hours listing."""
        hours = sample_model_data.available_hours("z500")
        assert hours == [0, 6, 12, 24]

    def test_missing_field(self, sample_model_data):
        """Test handling of missing fields."""
        field = sample_model_data.get_field("z500", 48)
        assert field is None
