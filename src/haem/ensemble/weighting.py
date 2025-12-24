"""
Ensemble Weight Calculation Module.

This module implements dynamic weight calculation for ensemble members
based on:
- Historical model skill (verification-derived)
- Run-to-run consistency
- AI-derived confidence scores
"""

import logging
from typing import Optional

import numpy as np

from haem.models.meteorological import ModelData, NWPModel
from haem.models.analysis import AnalysisResult
from haem.models.ensemble import EnsembleConfig, WeightedModel

logger = logging.getLogger(__name__)


class WeightCalculator:
    """
    Calculates ensemble weights for each model.

    Weight formula:
    W_combined = (W_historical ^ α) × (W_consistency ^ β) × (W_ai ^ γ)

    Where α, β, γ are tunable exponents (default 1.0 each).
    """

    # Default historical skill weights based on verification studies
    DEFAULT_HISTORICAL_WEIGHTS = {
        NWPModel.ECMWF: 0.95,  # Typically best overall skill
        NWPModel.GFS: 0.85,    # Good skill, occasional phase errors
        NWPModel.UKMO: 0.88,   # Strong European performance
        NWPModel.GEM: 0.82,    # Solid but sometimes lags
        NWPModel.ICON: 0.87,   # Good, especially Europe
    }

    # Lead time skill decay by model
    SKILL_DECAY_RATES = {
        NWPModel.ECMWF: 0.002,  # Slowest decay
        NWPModel.GFS: 0.003,
        NWPModel.UKMO: 0.003,
        NWPModel.GEM: 0.0035,
        NWPModel.ICON: 0.003,
    }

    def __init__(
        self,
        config: EnsembleConfig,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ):
        """
        Initialize weight calculator.

        Args:
            config: Ensemble configuration
            alpha: Exponent for historical weight
            beta: Exponent for consistency weight
            gamma: Exponent for AI weight
        """
        self.config = config
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def calculate_weights(
        self,
        model_data_dict: dict[NWPModel, ModelData],
        analysis_results: dict[NWPModel, AnalysisResult],
        forecast_hour: int,
        previous_runs: Optional[dict[NWPModel, ModelData]] = None,
    ) -> list[WeightedModel]:
        """
        Calculate weights for all models.

        Args:
            model_data_dict: Current model data
            analysis_results: Analysis results with confidence scores
            forecast_hour: Forecast lead time
            previous_runs: Previous model runs for consistency

        Returns:
            List of WeightedModel objects
        """
        weighted_models = []

        for model in model_data_dict.keys():
            # Historical skill weight
            historical_weight = self._get_historical_weight(model, forecast_hour)

            # Run consistency weight
            consistency_weight = self._get_consistency_weight(
                model,
                model_data_dict.get(model),
                previous_runs.get(model) if previous_runs else None,
                forecast_hour,
            )

            # AI confidence weight
            ai_weight = self._get_ai_weight(model, analysis_results.get(model))

            # Create weighted model
            wm = WeightedModel(
                model=model,
                historical_skill_weight=historical_weight,
                run_consistency_weight=consistency_weight,
                ai_confidence_weight=ai_weight,
                notes=self._generate_weight_notes(
                    model, historical_weight, consistency_weight, ai_weight
                ),
            )
            weighted_models.append(wm)

        # Normalize combined weights
        total_weight = sum(wm.combined_weight for wm in weighted_models)
        if total_weight > 0:
            # Weights are normalized in the ensemble integration step
            pass

        return weighted_models

    def _get_historical_weight(
        self,
        model: NWPModel,
        forecast_hour: int,
    ) -> float:
        """Get historical skill weight with lead time decay."""
        if not self.config.use_historical_weights:
            return 1.0

        # Base weight from config or defaults
        base_weight = self.config.default_skill_weights.get(
            model.value,
            self.DEFAULT_HISTORICAL_WEIGHTS.get(model, 0.8)
        )

        # Apply lead time decay
        decay_rate = self.SKILL_DECAY_RATES.get(model, 0.003)
        decay_factor = np.exp(-decay_rate * forecast_hour)

        # Don't let weight drop below 50% of base
        adjusted_weight = base_weight * max(0.5, decay_factor)

        return float(adjusted_weight)

    def _get_consistency_weight(
        self,
        model: NWPModel,
        current_data: Optional[ModelData],
        previous_data: Optional[ModelData],
        forecast_hour: int,
    ) -> float:
        """Calculate run-to-run consistency weight."""
        if not self.config.use_run_consistency_weights:
            return 1.0

        if current_data is None or previous_data is None:
            return 0.85  # Default moderate consistency

        # Get fields at forecast hour
        if forecast_hour not in current_data.z500:
            return 0.8

        current_z500 = current_data.z500[forecast_hour].data

        # Find corresponding time in previous run
        hour_diff = 6  # Assume 6h between runs (conservative)
        prev_hour = forecast_hour + hour_diff

        if prev_hour not in previous_data.z500:
            return 0.8

        prev_z500 = previous_data.z500[prev_hour].data

        # Compute correlation
        corr = np.corrcoef(current_z500.ravel(), prev_z500.ravel())[0, 1]

        # Compute RMSE
        rmse = np.sqrt(np.mean((current_z500 - prev_z500) ** 2))

        # Convert to weight (high correlation, low RMSE = high weight)
        corr_weight = max(0, corr)
        rmse_weight = max(0, 1 - rmse / 15)  # 15 dam RMSE = 0 weight

        consistency_weight = (corr_weight + rmse_weight) / 2

        return float(consistency_weight)

    def _get_ai_weight(
        self,
        model: NWPModel,
        analysis: Optional[AnalysisResult],
    ) -> float:
        """Get AI-derived confidence weight."""
        if not self.config.use_ai_weights:
            return 1.0

        if analysis is None:
            return 0.7  # Default moderate weight

        # Use reliability and confidence scores
        reliability_score = analysis.reliability.reliability_score / 100
        confidence_score = analysis.confidence.overall_score / 100

        # Combine reliability and confidence
        ai_weight = (reliability_score * 0.6 + confidence_score * 0.4)

        return float(ai_weight)

    def _generate_weight_notes(
        self,
        model: NWPModel,
        historical: float,
        consistency: float,
        ai: float,
    ) -> str:
        """Generate notes explaining the weight."""
        notes = []

        if historical > 0.9:
            notes.append("High historical skill")
        elif historical < 0.75:
            notes.append("Lower historical skill at this range")

        if consistency > 0.9:
            notes.append("Very consistent with previous run")
        elif consistency < 0.7:
            notes.append("Significant changes from previous run")

        if ai > 0.8:
            notes.append("High AI confidence")
        elif ai < 0.6:
            notes.append("Lower AI confidence")

        return "; ".join(notes) if notes else "Standard weighting applied"


class AdaptiveWeightCalculator(WeightCalculator):
    """
    Adaptive weight calculator that learns from verification.

    Adjusts historical weights based on recent forecast verification.
    """

    def __init__(
        self,
        config: EnsembleConfig,
        learning_rate: float = 0.1,
        verification_window_days: int = 30,
    ):
        super().__init__(config)
        self.learning_rate = learning_rate
        self.verification_window_days = verification_window_days

        # Adaptive weights start from defaults
        self.adaptive_weights = self.DEFAULT_HISTORICAL_WEIGHTS.copy()

    def update_weights_from_verification(
        self,
        verification_scores: dict[NWPModel, float],
    ) -> None:
        """
        Update weights based on verification scores.

        Args:
            verification_scores: Dict mapping model to verification skill score (0-1)
        """
        for model, score in verification_scores.items():
            if model in self.adaptive_weights:
                current = self.adaptive_weights[model]
                # Exponential moving average update
                updated = current * (1 - self.learning_rate) + score * self.learning_rate
                self.adaptive_weights[model] = updated

                logger.info(
                    f"Updated {model.value} weight: {current:.3f} -> {updated:.3f}"
                )

    def _get_historical_weight(
        self,
        model: NWPModel,
        forecast_hour: int,
    ) -> float:
        """Override to use adaptive weights."""
        if not self.config.use_historical_weights:
            return 1.0

        base_weight = self.adaptive_weights.get(
            model,
            self.DEFAULT_HISTORICAL_WEIGHTS.get(model, 0.8)
        )

        # Apply lead time decay
        decay_rate = self.SKILL_DECAY_RATES.get(model, 0.003)
        decay_factor = np.exp(-decay_rate * forecast_hour)
        adjusted_weight = base_weight * max(0.5, decay_factor)

        return float(adjusted_weight)
