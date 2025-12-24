"""Tests for HAEM ensemble integration."""

import numpy as np
import pytest
from datetime import datetime

from haem.models.meteorological import (
    NWPModel,
    ForecastRun,
    GeopotentialField,
    PressureLevel,
    ModelData,
)
from haem.models.ensemble import (
    EnsembleConfig,
    WeightedModel,
)
from haem.ensemble.weighting import WeightCalculator


class TestWeightCalculator:
    """Tests for WeightCalculator."""

    @pytest.fixture
    def config(self):
        """Create ensemble config."""
        return EnsembleConfig()

    @pytest.fixture
    def calculator(self, config):
        """Create weight calculator."""
        return WeightCalculator(config)

    @pytest.fixture
    def sample_model_data_dict(self):
        """Create sample model data for multiple models."""
        lats = np.linspace(30, 60, 31)
        lons = np.linspace(-30, 30, 61)

        model_data = {}

        for model in [NWPModel.ECMWF, NWPModel.GFS, NWPModel.UKMO]:
            run = ForecastRun(
                model=model,
                init_time=datetime(2024, 1, 15, 12, 0, 0),
            )

            data = ModelData(run=run)

            # Slightly different data for each model
            field_data = np.random.uniform(540, 580, (31, 61))

            data.z500[24] = GeopotentialField(
                level=PressureLevel.L500,
                valid_time=datetime(2024, 1, 16, 12, 0, 0),
                forecast_hour=24,
                lats=lats,
                lons=lons,
                data=field_data,
                source_model=model,
                run_time=run.init_time,
            )

            model_data[model] = data

        return model_data

    def test_historical_weight(self, calculator):
        """Test historical weight calculation."""
        weight = calculator._get_historical_weight(NWPModel.ECMWF, 24)

        assert 0 < weight <= 1.0
        # ECMWF should have high weight
        assert weight > 0.8

    def test_historical_weight_decay(self, calculator):
        """Test that weights decay with forecast hour."""
        weight_24h = calculator._get_historical_weight(NWPModel.GFS, 24)
        weight_168h = calculator._get_historical_weight(NWPModel.GFS, 168)

        assert weight_24h > weight_168h


class TestWeightedModel:
    """Tests for WeightedModel."""

    def test_combined_weight(self):
        """Test combined weight calculation."""
        wm = WeightedModel(
            model=NWPModel.ECMWF,
            historical_skill_weight=0.9,
            run_consistency_weight=0.85,
            ai_confidence_weight=0.8,
        )

        # Combined weight should be geometric mean
        expected = (0.9 * 0.85 * 0.8) ** (1/3)
        assert abs(wm.combined_weight - expected) < 0.001

    def test_low_weight_combination(self):
        """Test combined weight with low values."""
        wm = WeightedModel(
            model=NWPModel.GEM,
            historical_skill_weight=0.5,
            run_consistency_weight=0.6,
            ai_confidence_weight=0.5,
        )

        assert 0 < wm.combined_weight < 1.0
