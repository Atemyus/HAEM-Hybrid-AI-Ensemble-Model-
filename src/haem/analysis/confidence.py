"""
Confidence and Reliability Assessment Module.

This module provides algorithms for assessing:
- Forecast confidence (0-100)
- Model reliability
- Uncertainty quantification
- Ensemble agreement metrics
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from haem.models.meteorological import (
    ModelData,
    NWPModel,
    SynopticPattern,
    SynopticPatternType,
)
from haem.models.analysis import (
    ConfidenceScore,
    ReliabilityAssessment,
    UncertaintySource,
)

logger = logging.getLogger(__name__)


class ConfidenceAssessor:
    """
    Assesses confidence and reliability of NWP forecasts.

    Uses multiple metrics to quantify:
    - Temporal consistency (run-to-run stability)
    - Spatial coherence (physical plausibility)
    - Ensemble agreement (multi-model consensus)
    - Pattern predictability
    """

    # Historical model skill weights (from verification studies)
    HISTORICAL_SKILL = {
        NWPModel.ECMWF: 0.95,
        NWPModel.GFS: 0.85,
        NWPModel.UKMO: 0.88,
        NWPModel.GEM: 0.82,
        NWPModel.ICON: 0.87,
    }

    # Forecast hour decay rates (confidence decreases with lead time)
    HOUR_DECAY_RATE = 0.003  # ~0.3% per hour

    # Pattern predictability scores
    PATTERN_PREDICTABILITY = {
        SynopticPatternType.ZONAL_FLOW: 85,
        SynopticPatternType.RIDGE: 80,
        SynopticPatternType.TROUGH: 78,
        SynopticPatternType.BLOCKING_HIGH: 70,
        SynopticPatternType.OMEGA_BLOCK: 65,
        SynopticPatternType.CUT_OFF_LOW: 55,
        SynopticPatternType.CYCLONIC_VORTEX: 75,
        SynopticPatternType.ANTICYCLONIC_VORTEX: 80,
        SynopticPatternType.FRONTAL_ZONE: 70,
        SynopticPatternType.MERIDIONAL_FLOW: 60,
    }

    def assess_confidence(
        self,
        model_data: ModelData,
        forecast_hour: int,
        patterns: list[SynopticPattern],
        ensemble_spread: Optional[NDArray[np.float64]] = None,
    ) -> ConfidenceScore:
        """
        Assess forecast confidence for a specific forecast time.

        Args:
            model_data: Model data being assessed
            forecast_hour: Forecast lead time in hours
            patterns: Detected synoptic patterns
            ensemble_spread: Optional ensemble spread field

        Returns:
            ConfidenceScore with component breakdowns
        """
        # Temporal consistency (stability between fields)
        temporal_consistency = self._assess_temporal_consistency(model_data, forecast_hour)

        # Spatial coherence
        spatial_coherence = self._assess_spatial_coherence(model_data, forecast_hour)

        # Physical consistency
        physical_consistency = self._assess_physical_consistency(model_data, forecast_hour)

        # Ensemble agreement (if spread provided)
        if ensemble_spread is not None:
            ensemble_agreement = self._assess_ensemble_agreement(ensemble_spread)
        else:
            # Default moderate agreement
            ensemble_agreement = 70.0

        # Pattern-based adjustment
        pattern_factor = self._get_pattern_predictability(patterns)

        # Lead time decay
        lead_time_factor = max(0.3, 1.0 - self.HOUR_DECAY_RATE * forecast_hour)

        # Calculate overall score
        components = [
            temporal_consistency,
            spatial_coherence,
            physical_consistency,
            ensemble_agreement,
        ]
        base_score = np.mean(components)

        # Apply pattern and lead time factors
        overall_score = base_score * pattern_factor * lead_time_factor

        # Clamp to valid range
        overall_score = max(0, min(100, overall_score))

        return ConfidenceScore(
            overall_score=float(overall_score),
            temporal_consistency=float(temporal_consistency),
            spatial_coherence=float(spatial_coherence),
            physical_consistency=float(physical_consistency),
            ensemble_agreement=float(ensemble_agreement),
        )

    def assess_reliability(
        self,
        model: NWPModel,
        model_data: ModelData,
        forecast_hour: int,
        previous_run: Optional[ModelData] = None,
    ) -> ReliabilityAssessment:
        """
        Assess reliability of a specific model.

        Args:
            model: NWP model being assessed
            model_data: Current model data
            forecast_hour: Forecast lead time
            previous_run: Previous model run for consistency check

        Returns:
            ReliabilityAssessment
        """
        valid_time = model_data.run.init_time + np.timedelta64(forecast_hour, 'h')
        valid_time = datetime.utcfromtimestamp(valid_time.astype('datetime64[s]').astype('int'))

        # Base historical skill
        historical_skill = self.HISTORICAL_SKILL.get(model, 0.8)

        # Run-to-run consistency
        run_consistency = self._assess_run_consistency(
            model_data, previous_run, forecast_hour
        )

        # Identify strengths
        strengths = self._identify_model_strengths(model)

        # Identify weaknesses
        weaknesses = self._identify_model_weaknesses(model, forecast_hour)

        # Identify uncertainty sources
        uncertainty_sources = self._identify_uncertainty_sources(
            model_data, forecast_hour, run_consistency
        )

        # Calculate reliability score
        base_reliability = historical_skill * 100
        consistency_factor = run_consistency
        lead_time_factor = max(0.5, 1.0 - self.HOUR_DECAY_RATE * 0.5 * forecast_hour)

        reliability_score = base_reliability * consistency_factor * lead_time_factor
        reliability_score = max(0, min(100, reliability_score))

        # Generate notes
        notes = self._generate_reliability_notes(
            model, forecast_hour, run_consistency
        )

        return ReliabilityAssessment(
            model=model,
            valid_time=valid_time,
            forecast_hour=forecast_hour,
            reliability_score=float(reliability_score),
            historical_skill=historical_skill,
            run_consistency=run_consistency,
            strengths=strengths,
            weaknesses=weaknesses,
            uncertainty_sources=uncertainty_sources,
            notes=notes,
        )

    def _assess_temporal_consistency(
        self,
        model_data: ModelData,
        forecast_hour: int,
    ) -> float:
        """Assess temporal consistency between adjacent forecast hours."""
        # Check consistency with adjacent hours
        prev_hour = forecast_hour - 3
        next_hour = forecast_hour + 3

        scores = []

        if prev_hour in model_data.z500 and forecast_hour in model_data.z500:
            prev_field = model_data.z500[prev_hour].data
            curr_field = model_data.z500[forecast_hour].data

            # Correlation as consistency metric
            corr = np.corrcoef(prev_field.ravel(), curr_field.ravel())[0, 1]
            scores.append(corr * 100)

            # Change magnitude (should be small)
            max_change = np.max(np.abs(curr_field - prev_field))
            change_score = max(0, 100 - max_change * 5)
            scores.append(change_score)

        if not scores:
            return 75.0  # Default moderate score

        return float(np.mean(scores))

    def _assess_spatial_coherence(
        self,
        model_data: ModelData,
        forecast_hour: int,
    ) -> float:
        """Assess spatial coherence (smoothness and physical plausibility)."""
        if forecast_hour not in model_data.z500:
            return 75.0

        data = model_data.z500[forecast_hour].data

        # Check for excessive gradients
        dy = np.gradient(data, axis=0)
        dx = np.gradient(data, axis=1)
        grad_mag = np.sqrt(dx**2 + dy**2)

        # Median gradient should be reasonable
        median_grad = np.median(grad_mag)

        # Score based on gradient magnitude (reasonable is 0.05-0.15)
        if median_grad < 0.05:
            grad_score = 95  # Very smooth
        elif median_grad < 0.1:
            grad_score = 90
        elif median_grad < 0.15:
            grad_score = 80
        elif median_grad < 0.2:
            grad_score = 70
        else:
            grad_score = 50  # Too rough

        # Check for NaN/Inf
        finite_fraction = np.sum(np.isfinite(data)) / data.size
        finite_score = finite_fraction * 100

        # Check for outliers
        z_scores = np.abs((data - np.nanmean(data)) / (np.nanstd(data) + 1e-10))
        outlier_fraction = np.sum(z_scores > 4) / data.size
        outlier_score = (1 - outlier_fraction) * 100

        return float(np.mean([grad_score, finite_score, outlier_score]))

    def _assess_physical_consistency(
        self,
        model_data: ModelData,
        forecast_hour: int,
    ) -> float:
        """Assess physical consistency between fields."""
        scores = []

        # Check thermal wind consistency (T850 and Z500/Z850 thickness)
        if (forecast_hour in model_data.z500 and
                forecast_hour in model_data.z850 and
                forecast_hour in model_data.t850):

            z500 = model_data.z500[forecast_hour].data
            z850 = model_data.z850[forecast_hour].data
            t850 = model_data.t850[forecast_hour].data

            # Thickness (500-850 hPa)
            thickness = z500 - z850

            # Thickness should correlate with mean layer temperature
            corr = np.corrcoef(thickness.ravel(), t850.ravel())[0, 1]
            thermal_wind_score = max(0, corr * 100)
            scores.append(thermal_wind_score)

        # Check hydrostatic consistency (SLP and Z fields)
        if forecast_hour in model_data.slp and forecast_hour in model_data.z850:
            slp = model_data.slp[forecast_hour].data
            z850 = model_data.z850[forecast_hour].data

            # SLP and Z850 should be inversely related
            corr = np.corrcoef(slp.ravel(), z850.ravel())[0, 1]
            # For hydrostatic consistency, we expect negative correlation
            hydrostatic_score = max(0, -corr * 50 + 50)
            scores.append(hydrostatic_score)

        if not scores:
            return 75.0

        return float(np.mean(scores))

    def _assess_ensemble_agreement(
        self,
        spread: NDArray[np.float64],
    ) -> float:
        """Assess ensemble agreement from spread field."""
        # Lower spread = higher agreement
        mean_spread = np.nanmean(spread)
        max_spread = np.nanmax(spread)

        # Normalize by typical spread values (for Z500 in dam)
        typical_spread = 5.0  # dam

        normalized_spread = mean_spread / typical_spread

        # Convert to agreement score
        agreement = max(0, 100 * (1 - normalized_spread / 2))

        return float(agreement)

    def _get_pattern_predictability(
        self,
        patterns: list[SynopticPattern],
    ) -> float:
        """Get pattern-based predictability factor."""
        if not patterns:
            return 1.0  # Neutral factor

        # Weight by pattern intensity
        weighted_sum = 0.0
        total_weight = 0.0

        for pattern in patterns:
            score = self.PATTERN_PREDICTABILITY.get(pattern.pattern_type, 75)
            weight = pattern.intensity
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 1.0

        avg_predictability = weighted_sum / total_weight

        # Convert to factor (70 is neutral, >70 increases, <70 decreases)
        return avg_predictability / 70.0

    def _assess_run_consistency(
        self,
        current_data: ModelData,
        previous_data: Optional[ModelData],
        forecast_hour: int,
    ) -> float:
        """Assess consistency with previous model run."""
        if previous_data is None:
            return 0.85  # Default moderate consistency

        # Find matching valid time in previous run
        current_init = current_data.run.init_time
        prev_init = previous_data.run.init_time

        hour_diff = int((current_init - prev_init) / np.timedelta64(1, 'h'))
        prev_hour = forecast_hour + hour_diff

        if prev_hour not in previous_data.z500 or forecast_hour not in current_data.z500:
            return 0.8

        curr_z500 = current_data.z500[forecast_hour].data
        prev_z500 = previous_data.z500[prev_hour].data

        # Pattern correlation
        corr = np.corrcoef(curr_z500.ravel(), prev_z500.ravel())[0, 1]

        # RMSE
        rmse = np.sqrt(np.mean((curr_z500 - prev_z500) ** 2))

        # Convert to consistency score
        corr_score = max(0, corr)
        rmse_score = max(0, 1 - rmse / 10)  # 10 dam RMSE = 0 score

        return float((corr_score + rmse_score) / 2)

    def _identify_model_strengths(self, model: NWPModel) -> list[str]:
        """Identify known strengths of a model."""
        strengths = {
            NWPModel.ECMWF: [
                "Highest overall forecast skill",
                "Excellent ensemble system (ENS)",
                "Strong tropical cyclone prediction",
                "Good representation of blocking",
            ],
            NWPModel.GFS: [
                "Long forecast range (384h)",
                "Frequent update cycle (6-hourly)",
                "Good synoptic-scale representation",
                "Large ensemble (GEFS)",
            ],
            NWPModel.UKMO: [
                "Strong European regional performance",
                "Good mesoscale representation",
                "Excellent UK/Atlantic sector",
            ],
            NWPModel.GEM: [
                "Good Arctic representation",
                "Strong Canadian sector performance",
                "Reasonable ensemble system",
            ],
            NWPModel.ICON: [
                "High resolution capability",
                "Good convection representation",
                "Strong European performance",
            ],
        }
        return strengths.get(model, ["General purpose global model"])

    def _identify_model_weaknesses(
        self,
        model: NWPModel,
        forecast_hour: int,
    ) -> list[str]:
        """Identify known weaknesses of a model."""
        weaknesses = {
            NWPModel.ECMWF: [
                "Can underestimate convective precipitation",
            ],
            NWPModel.GFS: [
                "Can have phase speed errors in synoptic systems",
                "Sometimes over-amplifies ridges",
                "Tropical cyclone intensity biases",
            ],
            NWPModel.UKMO: [
                "Limited forecast range",
                "Sometimes struggles with blocking",
            ],
            NWPModel.GEM: [
                "Can lag European models in skill",
                "Sometimes less reliable in extended range",
            ],
            NWPModel.ICON: [
                "Relatively shorter operational history",
                "Can differ from consensus in some situations",
            ],
        }

        result = weaknesses.get(model, [])

        # Add forecast-range specific weaknesses
        if forecast_hour > 168:
            result.append("Extended range: all models show reduced skill")
        elif forecast_hour > 120:
            result.append("Medium range: pattern errors may be significant")

        return result

    def _identify_uncertainty_sources(
        self,
        model_data: ModelData,
        forecast_hour: int,
        run_consistency: float,
    ) -> list[UncertaintySource]:
        """Identify sources of forecast uncertainty."""
        sources = []

        # Run consistency implies model disagreement
        if run_consistency < 0.8:
            sources.append(UncertaintySource.TIMING_UNCERTAINTY)

        if run_consistency < 0.7:
            sources.append(UncertaintySource.PATTERN_EVOLUTION)

        # Lead time-based sources
        if forecast_hour > 120:
            sources.append(UncertaintySource.CHAOTIC_DYNAMICS)

        if forecast_hour > 72:
            sources.append(UncertaintySource.INITIAL_CONDITIONS)

        # Check data quality
        if model_data.data_quality_score < 0.9:
            sources.append(UncertaintySource.DATA_SPARSE_REGION)

        return sources

    def _generate_reliability_notes(
        self,
        model: NWPModel,
        forecast_hour: int,
        run_consistency: float,
    ) -> str:
        """Generate human-readable reliability notes."""
        notes = []

        # Model-specific notes
        if model == NWPModel.ECMWF:
            notes.append("ECMWF typically provides highest skill reference")
        elif model == NWPModel.GFS:
            if forecast_hour > 168:
                notes.append("GFS extended range: use with caution")

        # Consistency notes
        if run_consistency > 0.9:
            notes.append("Excellent run-to-run consistency")
        elif run_consistency > 0.8:
            notes.append("Good run-to-run consistency")
        elif run_consistency > 0.7:
            notes.append("Moderate run-to-run consistency")
        else:
            notes.append("Significant changes from previous run - low confidence")

        return ". ".join(notes)
