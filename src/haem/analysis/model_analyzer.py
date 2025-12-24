"""
Individual Model Analysis Engine.

This module provides comprehensive analysis of individual NWP model outputs,
including synoptic pattern identification, physical interpretation, and
reliability assessment.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from haem.models.meteorological import (
    AtmosphericField,
    GeopotentialField,
    ModelData,
    NWPModel,
    PressureField,
    SynopticPattern,
    SynopticPatternType,
    TemperatureField,
)
from haem.models.analysis import (
    AnalysisResult,
    ConfidenceScore,
    PhysicalInterpretation,
    PhysicalMechanism,
    ReliabilityAssessment,
    UncertaintySource,
)
from haem.analysis.pattern_detector import PatternDetector
from haem.analysis.interpreter import PhysicalInterpreter
from haem.analysis.confidence import ConfidenceAssessor

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """
    Comprehensive analyzer for individual NWP model outputs.

    Performs:
    - Synoptic pattern identification
    - Physical-dynamical interpretation
    - Confidence and reliability assessment
    - Key finding extraction
    """

    def __init__(self):
        self.pattern_detector = PatternDetector()
        self.interpreter = PhysicalInterpreter()
        self.confidence_assessor = ConfidenceAssessor()

    def analyze(
        self,
        model_data: ModelData,
        forecast_hour: int,
        previous_run: Optional[ModelData] = None,
    ) -> AnalysisResult:
        """
        Perform comprehensive analysis of a model at a specific forecast time.

        Args:
            model_data: Complete model data
            forecast_hour: Forecast hour to analyze
            previous_run: Previous model run for consistency assessment

        Returns:
            Complete AnalysisResult
        """
        model = model_data.run.model
        run_time = model_data.run.init_time
        valid_time = run_time + np.timedelta64(forecast_hour, 'h')
        valid_time = datetime.utcfromtimestamp(valid_time.astype('datetime64[s]').astype('int'))

        logger.info(f"Analyzing {model.value} +{forecast_hour}h (valid {valid_time})")

        # Get fields for this forecast hour
        z500 = model_data.z500.get(forecast_hour)
        z850 = model_data.z850.get(forecast_hour)
        t850 = model_data.t850.get(forecast_hour)
        slp = model_data.slp.get(forecast_hour)

        # Detect synoptic patterns
        patterns = self.pattern_detector.detect_all_patterns(
            z500=z500,
            slp=slp,
            t850=t850,
        )

        # Store patterns in model_data
        model_data.patterns[forecast_hour] = patterns

        # Generate physical interpretation
        interpretation = self.interpreter.interpret(
            z500=z500,
            z850=z850,
            t850=t850,
            slp=slp,
            patterns=patterns,
            forecast_hour=forecast_hour,
        )

        # Assess confidence and reliability
        confidence = self.confidence_assessor.assess_confidence(
            model_data=model_data,
            forecast_hour=forecast_hour,
            patterns=patterns,
        )

        reliability = self.confidence_assessor.assess_reliability(
            model=model,
            model_data=model_data,
            forecast_hour=forecast_hour,
            previous_run=previous_run,
        )

        # Generate synoptic summary
        synoptic_summary = self._generate_synoptic_summary(
            patterns=patterns,
            interpretation=interpretation,
            z500=z500,
            slp=slp,
        )

        # Extract key findings
        key_findings = self._extract_key_findings(
            patterns=patterns,
            interpretation=interpretation,
            confidence=confidence,
        )

        # Generate warnings
        warnings = self._generate_warnings(
            confidence=confidence,
            reliability=reliability,
            forecast_hour=forecast_hour,
        )

        return AnalysisResult(
            model=model,
            run_time=run_time,
            valid_time=valid_time,
            forecast_hour=forecast_hour,
            synoptic_summary=synoptic_summary,
            physical_interpretation=interpretation,
            confidence=confidence,
            reliability=reliability,
            key_findings=key_findings,
            warnings=warnings,
        )

    def _generate_synoptic_summary(
        self,
        patterns: list[SynopticPattern],
        interpretation: PhysicalInterpretation,
        z500: Optional[GeopotentialField],
        slp: Optional[PressureField],
    ) -> str:
        """Generate a concise synoptic summary."""
        summary_parts = []

        # Dominant pattern
        if patterns:
            dominant = patterns[0]
            summary_parts.append(
                f"A {dominant.description} is centered near {dominant.center_lat:.0f}°N, "
                f"{abs(dominant.center_lon):.0f}°{'W' if dominant.center_lon < 0 else 'E'}"
            )

        # Flow regime
        if interpretation.jet_position:
            summary_parts.append(f"The jet stream {interpretation.jet_position}")

        # Air mass characteristics
        if interpretation.air_mass_description:
            summary_parts.append(interpretation.air_mass_description)

        # SLP features
        if slp is not None:
            centers = slp.find_pressure_centers()
            if centers["lows"]:
                deepest = min(centers["lows"], key=lambda x: x[2])
                summary_parts.append(
                    f"Surface low pressure ({deepest[2]:.0f} hPa) near "
                    f"{deepest[0]:.0f}°N, {abs(deepest[1]):.0f}°{'W' if deepest[1] < 0 else 'E'}"
                )

        return ". ".join(summary_parts) + "." if summary_parts else "Insufficient data for summary."

    def _extract_key_findings(
        self,
        patterns: list[SynopticPattern],
        interpretation: PhysicalInterpretation,
        confidence: ConfidenceScore,
    ) -> list[str]:
        """Extract bullet-point key findings."""
        findings = []

        # Pattern-based findings
        for pattern in patterns[:3]:  # Top 3 patterns
            findings.append(
                f"{pattern.pattern_type.value.replace('_', ' ').title()} "
                f"identified at {pattern.center_lat:.0f}°N, {pattern.center_lon:.0f}°E"
            )

        # Mechanism findings
        findings.append(
            f"Primary driver: {interpretation.primary_mechanism.value.replace('_', ' ')}"
        )

        # Anomaly findings
        if interpretation.is_anomalous:
            findings.append(f"Anomalous pattern: {interpretation.anomaly_description}")

        # Thermal advection
        if "warm" in interpretation.thermal_advection.lower():
            findings.append("Warm air advection detected in analyzed region")
        elif "cold" in interpretation.thermal_advection.lower():
            findings.append("Cold air advection detected in analyzed region")

        return findings

    def _generate_warnings(
        self,
        confidence: ConfidenceScore,
        reliability: ReliabilityAssessment,
        forecast_hour: int,
    ) -> list[str]:
        """Generate analysis warnings and caveats."""
        warnings = []

        # Low confidence warnings
        if confidence.overall_score < 40:
            warnings.append(
                f"Low confidence ({confidence.overall_score:.0f}/100) - "
                "multiple scenarios possible"
            )

        # Reliability warnings
        if reliability.reliability_score < 50:
            warnings.append(
                f"Reduced reliability for {reliability.model.value.upper()} at this range"
            )

        # Long-range warnings
        if forecast_hour > 168:
            warnings.append(
                "Extended range forecast - increased uncertainty expected"
            )
        elif forecast_hour > 120:
            warnings.append(
                "Medium-range forecast - pattern evolution less certain"
            )

        # Uncertainty source warnings
        for source in reliability.uncertainty_sources:
            if source == UncertaintySource.MODEL_DISAGREEMENT:
                warnings.append("Significant model disagreement on key features")
            elif source == UncertaintySource.CHAOTIC_DYNAMICS:
                warnings.append("Chaotic dynamics limit predictability")
            elif source == UncertaintySource.CONVECTIVE_PROCESSES:
                warnings.append("Convective-scale processes add uncertainty")

        return warnings

    def analyze_temporal_evolution(
        self,
        model_data: ModelData,
        start_hour: int = 0,
        end_hour: int = 168,
        step: int = 24,
    ) -> list[AnalysisResult]:
        """
        Analyze temporal evolution of a model forecast.

        Args:
            model_data: Complete model data
            start_hour: Starting forecast hour
            end_hour: Ending forecast hour
            step: Hour step for analysis

        Returns:
            List of AnalysisResults for each time step
        """
        results = []
        hours = list(range(start_hour, end_hour + 1, step))

        for hour in hours:
            if hour in model_data.z500:  # Check data availability
                result = self.analyze(model_data, hour)
                results.append(result)

        return results

    def compare_runs(
        self,
        current_run: ModelData,
        previous_run: ModelData,
        forecast_hour: int,
    ) -> dict:
        """
        Compare current run with previous run for consistency.

        Args:
            current_run: Current model run
            previous_run: Previous model run
            forecast_hour: Forecast hour to compare

        Returns:
            Dict with comparison metrics
        """
        # Get valid time from current run
        valid_time = current_run.run.init_time + np.timedelta64(forecast_hour, 'h')

        # Find corresponding hour in previous run
        prev_offset = (current_run.run.init_time - previous_run.run.init_time)
        prev_hour = forecast_hour + int(prev_offset.astype('timedelta64[h]').astype(int))

        if prev_hour not in previous_run.z500:
            return {"consistent": None, "message": "Previous run data not available"}

        # Compare Z500 fields
        current_z500 = current_run.z500[forecast_hour].data
        prev_z500 = previous_run.z500[prev_hour].data

        # Compute differences
        rmse = float(np.sqrt(np.mean((current_z500 - prev_z500) ** 2)))
        max_diff = float(np.max(np.abs(current_z500 - prev_z500)))
        pattern_corr = float(np.corrcoef(current_z500.ravel(), prev_z500.ravel())[0, 1])

        # Determine consistency
        is_consistent = pattern_corr > 0.95 and rmse < 5.0

        return {
            "consistent": is_consistent,
            "rmse_dam": rmse,
            "max_diff_dam": max_diff,
            "pattern_correlation": pattern_corr,
            "valid_time": str(valid_time),
        }
