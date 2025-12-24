"""
Ensemble integration models for multi-model aggregation.

This module defines structures for:
- Ensemble configuration
- Weighted model combinations
- Final ensemble field outputs
- Ensemble result summaries
"""

from datetime import datetime
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, computed_field

from haem.models.meteorological import NWPModel, SynopticPatternType
from haem.models.analysis import ConfidenceScore, PhysicalInterpretation


class WeightedModel(BaseModel):
    """A single model with its ensemble weights."""

    model: NWPModel

    # Base weights
    historical_skill_weight: float = Field(
        ge=0.0, le=1.0,
        description="Weight based on historical verification skill"
    )
    run_consistency_weight: float = Field(
        ge=0.0, le=1.0,
        description="Weight based on run-to-run consistency"
    )
    ai_confidence_weight: float = Field(
        ge=0.0, le=1.0,
        description="Weight from AI meteorological assessment"
    )

    # Computed combined weight
    @computed_field
    @property
    def combined_weight(self) -> float:
        """Calculate combined weight from all components."""
        # Geometric mean of weights for balanced combination
        weights = [
            self.historical_skill_weight,
            self.run_consistency_weight,
            self.ai_confidence_weight,
        ]
        return float(np.prod(weights) ** (1.0 / len(weights)))

    # Optional model-specific notes
    notes: str = Field(default="")


class EnsembleConfig(BaseModel):
    """Configuration for ensemble generation."""

    # Model selection
    models: list[NWPModel] = Field(
        default_factory=lambda: [
            NWPModel.ECMWF,
            NWPModel.GFS,
            NWPModel.UKMO,
            NWPModel.GEM,
            NWPModel.ICON,
        ],
        description="Models to include in ensemble"
    )

    # Weighting scheme
    use_historical_weights: bool = Field(
        default=True,
        description="Apply historical skill weights"
    )
    use_run_consistency_weights: bool = Field(
        default=True,
        description="Apply run-to-run consistency weights"
    )
    use_ai_weights: bool = Field(
        default=True,
        description="Apply AI-derived confidence weights"
    )

    # Aggregation method
    aggregation_method: str = Field(
        default="weighted_mean",
        description="Method: 'weighted_mean', 'weighted_median', 'bayesian'"
    )

    # Spatial smoothing
    apply_smoothing: bool = Field(
        default=True,
        description="Apply spatial smoothing to ensemble"
    )
    smoothing_sigma: float = Field(
        default=1.0,
        description="Gaussian smoothing sigma in grid points"
    )

    # Outlier handling
    remove_outliers: bool = Field(
        default=True,
        description="Remove outlier models from ensemble"
    )
    outlier_threshold: float = Field(
        default=2.5,
        description="Standard deviations for outlier detection"
    )

    # Default historical skill weights (can be overridden)
    default_skill_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "ecmwf": 0.95,  # Typically highest skill
            "gfs": 0.85,
            "ukmo": 0.88,
            "gem": 0.82,
            "icon": 0.87,
        }
    )


class EnsembleField(BaseModel):
    """Ensemble-combined atmospheric field."""

    model_config = {"arbitrary_types_allowed": True}

    field_name: str
    units: str
    valid_time: datetime
    forecast_hour: int
    run_time: datetime

    # Spatial grid
    lats: NDArray[np.float64]
    lons: NDArray[np.float64]

    # Ensemble statistics
    ensemble_mean: NDArray[np.float64] = Field(
        description="Weighted ensemble mean field"
    )
    ensemble_spread: NDArray[np.float64] = Field(
        description="Ensemble spread (std dev)"
    )
    ensemble_min: NDArray[np.float64] = Field(
        description="Minimum across ensemble"
    )
    ensemble_max: NDArray[np.float64] = Field(
        description="Maximum across ensemble"
    )

    # Probability fields (optional)
    probability_above_threshold: Optional[NDArray[np.float64]] = Field(
        default=None,
        description="Probability of exceeding a threshold"
    )
    threshold_value: Optional[float] = Field(default=None)

    # Contributing models
    contributing_models: list[NWPModel]
    model_weights: dict[str, float] = Field(
        description="Normalized weights used for each model"
    )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the data arrays."""
        return self.ensemble_mean.shape

    @property
    def normalized_spread(self) -> NDArray[np.float64]:
        """Return spread normalized by the mean (coefficient of variation)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = self.ensemble_spread / np.abs(self.ensemble_mean)
            return np.where(np.isfinite(result), result, 0.0)

    def get_confidence_field(self) -> NDArray[np.float64]:
        """
        Generate a spatial confidence field based on ensemble spread.

        Low spread = high confidence, high spread = low confidence.
        Returns values 0-100.
        """
        # Normalize spread to 0-1 range using percentiles
        spread_pct = (
            (self.ensemble_spread - np.nanmin(self.ensemble_spread)) /
            (np.nanpercentile(self.ensemble_spread, 95) - np.nanmin(self.ensemble_spread) + 1e-10)
        )
        spread_pct = np.clip(spread_pct, 0, 1)
        # Invert so low spread = high confidence
        return (1 - spread_pct) * 100


class EnsembleScenario(BaseModel):
    """A distinct forecast scenario from ensemble clustering."""

    scenario_id: int
    probability: float = Field(ge=0.0, le=1.0)
    description: str
    supporting_models: list[NWPModel]
    dominant_pattern: SynopticPatternType
    key_features: list[str]


class EnsembleResult(BaseModel):
    """Complete ensemble analysis result."""

    model_config = {"arbitrary_types_allowed": True}

    run_time: datetime
    valid_time: datetime
    forecast_hour: int

    # Configuration used
    config: EnsembleConfig

    # Model weights
    model_weights: list[WeightedModel]

    # Ensemble fields
    z500: Optional[EnsembleField] = None
    z850: Optional[EnsembleField] = None
    t850: Optional[EnsembleField] = None
    slp: Optional[EnsembleField] = None

    # Physical interpretation of ensemble
    physical_interpretation: PhysicalInterpretation

    # Confidence assessment
    confidence: ConfidenceScore

    # Scenario analysis
    scenarios: list[EnsembleScenario] = Field(default_factory=list)
    dominant_scenario_probability: float = Field(ge=0.0, le=1.0)

    # Summary
    synoptic_summary: str
    key_findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # Final confidence score
    final_confidence_score: float = Field(
        ge=0.0, le=100.0,
        description="Final ensemble confidence score (0-100)"
    )

    def get_weight_summary(self) -> str:
        """Generate summary of model weights used."""
        lines = ["Model Weights Used:"]
        for mw in sorted(self.model_weights, key=lambda x: x.combined_weight, reverse=True):
            lines.append(
                f"  {mw.model.value.upper():6s}: "
                f"Historical={mw.historical_skill_weight:.2f} "
                f"Consistency={mw.run_consistency_weight:.2f} "
                f"AI={mw.ai_confidence_weight:.2f} "
                f"=> Combined={mw.combined_weight:.3f}"
            )
        return "\n".join(lines)

    def to_report(self) -> str:
        """Generate complete ensemble analysis report."""
        lines = [
            "=" * 70,
            "HAEM ENSEMBLE ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Run Time: {self.run_time.strftime('%Y-%m-%d %H:%MZ')}",
            f"Valid Time: {self.valid_time.strftime('%Y-%m-%d %H:%MZ')} (+{self.forecast_hour}h)",
            f"Models Used: {', '.join(m.model.value.upper() for m in self.model_weights)}",
            "",
            "-" * 70,
            "SYNOPTIC SUMMARY",
            "-" * 70,
            self.synoptic_summary,
            "",
            "-" * 70,
            "PHYSICAL INTERPRETATION",
            "-" * 70,
            f"Dominant Pattern: {self.physical_interpretation.dominant_pattern.value}",
            f"Primary Mechanism: {self.physical_interpretation.primary_mechanism.value}",
            "",
            self.physical_interpretation.causal_explanation,
            "",
            "-" * 70,
            self.get_weight_summary(),
            "",
            "-" * 70,
            "CONFIDENCE ASSESSMENT",
            "-" * 70,
            f"Overall Confidence: {self.confidence.overall_score:.0f}/100 ({self.confidence.level.value})",
            f"  - Temporal Consistency: {self.confidence.temporal_consistency:.0f}/100",
            f"  - Spatial Coherence: {self.confidence.spatial_coherence:.0f}/100",
            f"  - Physical Consistency: {self.confidence.physical_consistency:.0f}/100",
            f"  - Ensemble Agreement: {self.confidence.ensemble_agreement:.0f}/100",
            "",
        ]

        if self.scenarios:
            lines.extend([
                "-" * 70,
                "SCENARIO ANALYSIS",
                "-" * 70,
            ])
            for scenario in sorted(self.scenarios, key=lambda s: s.probability, reverse=True):
                lines.append(
                    f"Scenario {scenario.scenario_id} ({scenario.probability*100:.0f}%): "
                    f"{scenario.description}"
                )
                lines.append(f"  Supported by: {', '.join(m.value.upper() for m in scenario.supporting_models)}")

        lines.extend([
            "",
            "-" * 70,
            "KEY FINDINGS",
            "-" * 70,
        ])
        for finding in self.key_findings:
            lines.append(f"  • {finding}")

        if self.warnings:
            lines.extend([
                "",
                "-" * 70,
                "WARNINGS",
                "-" * 70,
            ])
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        lines.extend([
            "",
            "=" * 70,
            f"FINAL ENSEMBLE CONFIDENCE SCORE: {self.final_confidence_score:.0f}/100",
            "=" * 70,
        ])

        return "\n".join(lines)
