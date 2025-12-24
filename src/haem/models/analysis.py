"""
Analysis result models for meteorological assessment.

This module defines structures for:
- Individual model analysis results
- Confidence and reliability scoring
- Physical-dynamical interpretations
- Multi-model comparisons
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from haem.models.meteorological import NWPModel, SynopticPatternType


class ConfidenceLevel(str, Enum):
    """Qualitative confidence levels."""

    VERY_LOW = "very_low"      # 0-20
    LOW = "low"                # 20-40
    MODERATE = "moderate"      # 40-60
    HIGH = "high"              # 60-80
    VERY_HIGH = "very_high"    # 80-100


class UncertaintySource(str, Enum):
    """Sources of forecast uncertainty."""

    INITIAL_CONDITIONS = "initial_conditions"
    MODEL_PHYSICS = "model_physics"
    BOUNDARY_CONDITIONS = "boundary_conditions"
    RESOLUTION_LIMITATIONS = "resolution_limitations"
    CHAOTIC_DYNAMICS = "chaotic_dynamics"
    DATA_SPARSE_REGION = "data_sparse_region"
    CONVECTIVE_PROCESSES = "convective_processes"
    OROGRAPHIC_EFFECTS = "orographic_effects"
    MODEL_DISAGREEMENT = "model_disagreement"
    TIMING_UNCERTAINTY = "timing_uncertainty"
    PATTERN_EVOLUTION = "pattern_evolution"


class PhysicalMechanism(str, Enum):
    """Physical mechanisms driving atmospheric phenomena."""

    ROSSBY_WAVE_PROPAGATION = "rossby_wave_propagation"
    BAROCLINIC_INSTABILITY = "baroclinic_instability"
    BAROTROPIC_INSTABILITY = "barotropic_instability"
    DIABATIC_HEATING = "diabatic_heating"
    OROGRAPHIC_FORCING = "orographic_forcing"
    JET_STREAM_DYNAMICS = "jet_stream_dynamics"
    POTENTIAL_VORTICITY_ADVECTION = "potential_vorticity_advection"
    WARM_AIR_ADVECTION = "warm_air_advection"
    COLD_AIR_ADVECTION = "cold_air_advection"
    CYCLOGENESIS = "cyclogenesis"
    ANTICYCLOGENESIS = "anticyclogenesis"
    FRONTOGENESIS = "frontogenesis"
    FRONTOLYSIS = "frontolysis"
    BLOCKING_DEVELOPMENT = "blocking_development"
    WAVE_BREAKING = "wave_breaking"
    CUT_OFF_FORMATION = "cut_off_formation"
    MERIDIONAL_HEAT_TRANSPORT = "meridional_heat_transport"


class ConfidenceScore(BaseModel):
    """Quantified confidence score with breakdown."""

    overall_score: float = Field(ge=0.0, le=100.0, description="Overall confidence 0-100")

    # Component scores
    temporal_consistency: float = Field(
        ge=0.0, le=100.0,
        description="Consistency between consecutive model runs"
    )
    spatial_coherence: float = Field(
        ge=0.0, le=100.0,
        description="Spatial smoothness and physical plausibility"
    )
    physical_consistency: float = Field(
        ge=0.0, le=100.0,
        description="Adherence to physical constraints"
    )
    ensemble_agreement: float = Field(
        ge=0.0, le=100.0,
        description="Agreement with other ensemble members/models"
    )

    # Qualitative level
    @property
    def level(self) -> ConfidenceLevel:
        """Derive qualitative confidence level from score."""
        if self.overall_score < 20:
            return ConfidenceLevel.VERY_LOW
        elif self.overall_score < 40:
            return ConfidenceLevel.LOW
        elif self.overall_score < 60:
            return ConfidenceLevel.MODERATE
        elif self.overall_score < 80:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH


class ReliabilityAssessment(BaseModel):
    """Reliability assessment for a single NWP model."""

    model: NWPModel
    valid_time: datetime
    forecast_hour: int

    # Core reliability metrics
    reliability_score: float = Field(
        ge=0.0, le=100.0,
        description="Overall reliability score"
    )

    # Historical skill weights (from verification)
    historical_skill: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Historical skill weight based on past verification"
    )

    # Run-to-run consistency
    run_consistency: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Consistency with previous model runs"
    )

    # Identified strengths
    strengths: list[str] = Field(default_factory=list)

    # Identified weaknesses
    weaknesses: list[str] = Field(default_factory=list)

    # Uncertainty sources
    uncertainty_sources: list[UncertaintySource] = Field(default_factory=list)

    # Notes
    notes: str = Field(default="")

    @property
    def weight(self) -> float:
        """Calculate ensemble weight from reliability metrics."""
        return (self.reliability_score / 100.0) * self.historical_skill * self.run_consistency


class PhysicalInterpretation(BaseModel):
    """Physical-dynamical interpretation of atmospheric patterns."""

    valid_time: datetime
    forecast_hour: int

    # Pattern identification
    dominant_pattern: SynopticPatternType
    secondary_patterns: list[SynopticPatternType] = Field(default_factory=list)

    # Driving mechanisms
    primary_mechanism: PhysicalMechanism
    contributing_mechanisms: list[PhysicalMechanism] = Field(default_factory=list)

    # Jet stream analysis
    jet_position: Optional[str] = Field(
        default=None,
        description="Description of jet stream position and characteristics"
    )
    jet_intensity: Optional[str] = Field(
        default=None,
        description="Jet stream intensity assessment"
    )

    # Air mass analysis
    air_mass_description: str = Field(
        default="",
        description="Description of relevant air masses and their origins"
    )
    thermal_advection: str = Field(
        default="",
        description="Description of thermal advection patterns"
    )

    # Temporal evolution
    evolution_description: str = Field(
        description="Physical explanation of pattern evolution"
    )
    expected_changes: list[str] = Field(
        default_factory=list,
        description="Expected changes in coming forecast period"
    )

    # Anomaly assessment
    is_anomalous: bool = Field(
        default=False,
        description="Whether pattern is anomalous relative to climatology"
    )
    anomaly_description: Optional[str] = Field(
        default=None,
        description="Description of anomalous features"
    )

    # Causal chain
    causal_explanation: str = Field(
        description="Complete causal chain explaining the pattern"
    )


class ModelComparison(BaseModel):
    """Comparison between multiple NWP models."""

    valid_time: datetime
    forecast_hour: int
    models_compared: list[NWPModel]

    # Agreement metrics
    overall_agreement: float = Field(
        ge=0.0, le=100.0,
        description="Overall agreement score across models"
    )

    # Field-specific agreement
    z500_agreement: float = Field(ge=0.0, le=100.0)
    slp_agreement: float = Field(ge=0.0, le=100.0)
    t850_agreement: float = Field(ge=0.0, le=100.0)

    # Pattern agreement
    pattern_consensus: bool = Field(
        description="Whether models agree on dominant synoptic pattern"
    )
    consensus_pattern: Optional[SynopticPatternType] = Field(default=None)

    # Divergences
    major_divergences: list[str] = Field(
        default_factory=list,
        description="Descriptions of significant model divergences"
    )
    divergence_locations: list[tuple[float, float]] = Field(
        default_factory=list,
        description="Lat/lon locations of major divergences"
    )

    # Timing differences
    timing_spread_hours: float = Field(
        default=0.0,
        description="Spread in timing of key features across models"
    )

    # Scenarios
    dominant_scenario: str = Field(
        description="Description of the most likely scenario"
    )
    alternative_scenarios: list[str] = Field(
        default_factory=list,
        description="Descriptions of alternative scenarios from outlier models"
    )


class AnalysisResult(BaseModel):
    """Complete analysis result for a single model at a forecast time."""

    model: NWPModel
    run_time: datetime
    valid_time: datetime
    forecast_hour: int

    # Synoptic summary
    synoptic_summary: str = Field(
        description="Concise synoptic summary"
    )

    # Detailed interpretation
    physical_interpretation: PhysicalInterpretation

    # Confidence assessment
    confidence: ConfidenceScore

    # Reliability assessment
    reliability: ReliabilityAssessment

    # Key findings
    key_findings: list[str] = Field(
        default_factory=list,
        description="Bullet points of key findings"
    )

    # Warnings/caveats
    warnings: list[str] = Field(
        default_factory=list,
        description="Important caveats or warnings"
    )

    def to_summary(self) -> str:
        """Generate a text summary of the analysis."""
        lines = [
            f"=== {self.model.full_name} Analysis ===",
            f"Run: {self.run_time.strftime('%Y-%m-%d %H:%MZ')}",
            f"Valid: {self.valid_time.strftime('%Y-%m-%d %H:%MZ')} (+{self.forecast_hour}h)",
            "",
            "SYNOPTIC SUMMARY:",
            self.synoptic_summary,
            "",
            f"CONFIDENCE: {self.confidence.overall_score:.0f}/100 ({self.confidence.level.value})",
            f"RELIABILITY: {self.reliability.reliability_score:.0f}/100",
            "",
            "KEY FINDINGS:",
        ]
        for finding in self.key_findings:
            lines.append(f"  • {finding}")

        if self.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)
