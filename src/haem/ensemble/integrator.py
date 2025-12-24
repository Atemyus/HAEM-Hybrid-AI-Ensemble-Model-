"""
Ensemble Integration Module.

This module implements the core ensemble aggregation algorithm:

Z_final(x,y,t) = Σ [ Z_model_i(x,y,t) × W_model_i × W_AI_i ]

Where:
- Z_model_i are the original NWP fields
- W_model_i are historical model skill weights
- W_AI_i are AI confidence and coherence weights

The final product maintains physical plausibility and spatial continuity.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from haem.models.meteorological import (
    GeopotentialField,
    ModelData,
    NWPModel,
    PressureField,
    TemperatureField,
)
from haem.models.analysis import (
    AnalysisResult,
    ConfidenceScore,
    PhysicalInterpretation,
    PhysicalMechanism,
)
from haem.models.ensemble import (
    EnsembleConfig,
    EnsembleField,
    EnsembleResult,
    EnsembleScenario,
    WeightedModel,
)
from haem.analysis.pattern_detector import PatternDetector
from haem.analysis.interpreter import PhysicalInterpreter
from haem.analysis.confidence import ConfidenceAssessor
from haem.ensemble.weighting import WeightCalculator

logger = logging.getLogger(__name__)


class EnsembleIntegrator:
    """
    Integrates multiple NWP model outputs into a weighted ensemble.

    The integration process:
    1. Calculate weights for each model (historical + AI-derived)
    2. Perform weighted aggregation of fields
    3. Apply physical consistency checks
    4. Compute ensemble statistics (mean, spread)
    5. Generate physical interpretation
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        """
        Initialize ensemble integrator.

        Args:
            config: Ensemble configuration (uses defaults if None)
        """
        self.config = config or EnsembleConfig()
        self.weight_calculator = WeightCalculator(self.config)
        self.pattern_detector = PatternDetector()
        self.interpreter = PhysicalInterpreter()
        self.confidence_assessor = ConfidenceAssessor()

    def integrate(
        self,
        model_data_dict: dict[NWPModel, ModelData],
        analysis_results: dict[NWPModel, AnalysisResult],
        forecast_hour: int,
        run_time: Optional[datetime] = None,
    ) -> EnsembleResult:
        """
        Perform ensemble integration for a specific forecast time.

        Args:
            model_data_dict: Dict mapping model to ModelData
            analysis_results: Analysis results from each model
            forecast_hour: Forecast hour to integrate
            run_time: Model run time (inferred from data if None)

        Returns:
            EnsembleResult with integrated fields and analysis
        """
        if not model_data_dict:
            raise ValueError("No model data provided for integration")

        # Determine run time
        if run_time is None:
            first_model = next(iter(model_data_dict.values()))
            run_time = first_model.run.init_time

        valid_time = run_time + np.timedelta64(forecast_hour, 'h')
        valid_time = datetime.utcfromtimestamp(valid_time.astype('datetime64[s]').astype('int'))

        logger.info(f"Integrating ensemble for +{forecast_hour}h (valid {valid_time})")

        # Calculate weights for each model
        model_weights = self.weight_calculator.calculate_weights(
            model_data_dict=model_data_dict,
            analysis_results=analysis_results,
            forecast_hour=forecast_hour,
        )

        # Integrate each field
        z500_ensemble = self._integrate_field(
            model_data_dict, model_weights, forecast_hour, "z500"
        )
        z850_ensemble = self._integrate_field(
            model_data_dict, model_weights, forecast_hour, "z850"
        )
        t850_ensemble = self._integrate_field(
            model_data_dict, model_weights, forecast_hour, "t850"
        )
        slp_ensemble = self._integrate_field(
            model_data_dict, model_weights, forecast_hour, "slp"
        )

        # Detect patterns in ensemble mean
        patterns = []
        if z500_ensemble is not None:
            # Create a temporary GeopotentialField for pattern detection
            temp_z500 = GeopotentialField(
                level=None,
                valid_time=valid_time,
                forecast_hour=forecast_hour,
                lats=z500_ensemble.lats,
                lons=z500_ensemble.lons,
                data=z500_ensemble.ensemble_mean,
                source_model=NWPModel.ECMWF,  # Placeholder
                run_time=run_time,
            )
            patterns = self.pattern_detector.detect_all_patterns(z500=temp_z500)

        # Generate physical interpretation
        interpretation = self._generate_interpretation(
            z500_ensemble, t850_ensemble, slp_ensemble, patterns, forecast_hour
        )

        # Assess ensemble confidence
        confidence = self._assess_ensemble_confidence(
            z500_ensemble, slp_ensemble, model_weights, forecast_hour
        )

        # Identify scenarios
        scenarios = self._identify_scenarios(
            model_data_dict, analysis_results, forecast_hour
        )

        # Generate summary
        synoptic_summary = self._generate_ensemble_summary(
            patterns, interpretation, confidence
        )

        # Key findings
        key_findings = self._extract_key_findings(
            patterns, interpretation, model_weights
        )

        # Warnings
        warnings = self._generate_warnings(
            confidence, model_weights, forecast_hour
        )

        # Final confidence score
        final_score = self._calculate_final_score(confidence, scenarios)

        return EnsembleResult(
            run_time=run_time,
            valid_time=valid_time,
            forecast_hour=forecast_hour,
            config=self.config,
            model_weights=model_weights,
            z500=z500_ensemble,
            z850=z850_ensemble,
            t850=t850_ensemble,
            slp=slp_ensemble,
            physical_interpretation=interpretation,
            confidence=confidence,
            scenarios=scenarios,
            dominant_scenario_probability=(
                scenarios[0].probability if scenarios else 1.0
            ),
            synoptic_summary=synoptic_summary,
            key_findings=key_findings,
            warnings=warnings,
            final_confidence_score=final_score,
        )

    def _integrate_field(
        self,
        model_data_dict: dict[NWPModel, ModelData],
        model_weights: list[WeightedModel],
        forecast_hour: int,
        field_name: str,
    ) -> Optional[EnsembleField]:
        """Integrate a single field type across all models."""
        # Collect fields and weights
        fields = []
        weights = []
        contributing_models = []
        lats = None
        lons = None

        weight_dict = {mw.model: mw.combined_weight for mw in model_weights}

        for model, data in model_data_dict.items():
            field_dict = getattr(data, field_name)
            if forecast_hour in field_dict:
                field = field_dict[forecast_hour]
                fields.append(field.data)
                weights.append(weight_dict.get(model, 0.5))
                contributing_models.append(model)

                if lats is None:
                    lats = field.lats
                    lons = field.lons

        if not fields:
            return None

        # Stack and compute weighted statistics
        stacked = np.stack(fields, axis=0)
        weights_array = np.array(weights)

        # Normalize weights
        weights_norm = weights_array / np.sum(weights_array)

        # Remove outliers if configured
        if self.config.remove_outliers and len(fields) >= 3:
            stacked, weights_norm = self._remove_outliers(stacked, weights_norm)

        # Weighted mean
        ensemble_mean = np.average(stacked, axis=0, weights=weights_norm)

        # Apply smoothing if configured
        if self.config.apply_smoothing:
            ensemble_mean = gaussian_filter(
                ensemble_mean,
                sigma=self.config.smoothing_sigma,
                mode="nearest"
            )

        # Ensemble spread (weighted std)
        deviations = stacked - ensemble_mean[np.newaxis, :, :]
        weighted_var = np.average(deviations**2, axis=0, weights=weights_norm)
        ensemble_spread = np.sqrt(weighted_var)

        # Min/max
        ensemble_min = np.min(stacked, axis=0)
        ensemble_max = np.max(stacked, axis=0)

        # Get run time from first model
        first_model = next(iter(model_data_dict.values()))
        run_time = first_model.run.init_time
        valid_time = run_time + np.timedelta64(forecast_hour, 'h')
        valid_time = datetime.utcfromtimestamp(valid_time.astype('datetime64[s]').astype('int'))

        # Get units based on field
        units = {"z500": "dam", "z850": "dam", "t850": "K", "slp": "hPa"}

        return EnsembleField(
            field_name=field_name,
            units=units.get(field_name, ""),
            valid_time=valid_time,
            forecast_hour=forecast_hour,
            run_time=run_time,
            lats=lats,
            lons=lons,
            ensemble_mean=ensemble_mean,
            ensemble_spread=ensemble_spread,
            ensemble_min=ensemble_min,
            ensemble_max=ensemble_max,
            contributing_models=contributing_models,
            model_weights={m.value: w for m, w in zip(contributing_models, weights_norm)},
        )

    def _remove_outliers(
        self,
        stacked: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Remove outlier models from ensemble."""
        # Compute mean across models
        mean_field = np.mean(stacked, axis=0)

        # Compute RMSE of each model from mean
        rmses = []
        for i in range(stacked.shape[0]):
            rmse = np.sqrt(np.mean((stacked[i] - mean_field) ** 2))
            rmses.append(rmse)

        rmses = np.array(rmses)
        median_rmse = np.median(rmses)
        mad = np.median(np.abs(rmses - median_rmse))

        # Flag outliers (more than threshold MADs from median)
        threshold = self.config.outlier_threshold
        outlier_mask = np.abs(rmses - median_rmse) > threshold * mad * 1.4826

        if np.sum(~outlier_mask) < 2:
            # Keep at least 2 models
            return stacked, weights

        # Remove outliers
        stacked_filtered = stacked[~outlier_mask]
        weights_filtered = weights[~outlier_mask]

        # Renormalize weights
        weights_filtered = weights_filtered / np.sum(weights_filtered)

        return stacked_filtered, weights_filtered

    def _generate_interpretation(
        self,
        z500: Optional[EnsembleField],
        t850: Optional[EnsembleField],
        slp: Optional[EnsembleField],
        patterns: list,
        forecast_hour: int,
    ) -> PhysicalInterpretation:
        """Generate physical interpretation of ensemble."""
        valid_time = z500.valid_time if z500 else datetime.utcnow()

        # Create temporary fields for interpreter
        temp_z500 = None
        temp_t850 = None
        temp_slp = None

        if z500 is not None:
            temp_z500 = GeopotentialField(
                level=None,
                valid_time=valid_time,
                forecast_hour=forecast_hour,
                lats=z500.lats,
                lons=z500.lons,
                data=z500.ensemble_mean,
                source_model=NWPModel.ECMWF,
                run_time=z500.run_time,
            )

        if t850 is not None:
            temp_t850 = TemperatureField(
                level=None,
                valid_time=valid_time,
                forecast_hour=forecast_hour,
                lats=t850.lats,
                lons=t850.lons,
                data=t850.ensemble_mean,
                source_model=NWPModel.ECMWF,
                run_time=t850.run_time,
            )

        if slp is not None:
            temp_slp = PressureField(
                valid_time=valid_time,
                forecast_hour=forecast_hour,
                lats=slp.lats,
                lons=slp.lons,
                data=slp.ensemble_mean,
                source_model=NWPModel.ECMWF,
                run_time=slp.run_time,
            )

        return self.interpreter.interpret(
            z500=temp_z500,
            t850=temp_t850,
            slp=temp_slp,
            patterns=patterns,
            forecast_hour=forecast_hour,
        )

    def _assess_ensemble_confidence(
        self,
        z500: Optional[EnsembleField],
        slp: Optional[EnsembleField],
        model_weights: list[WeightedModel],
        forecast_hour: int,
    ) -> ConfidenceScore:
        """Assess overall ensemble confidence."""
        scores = []

        # Spread-based confidence
        if z500 is not None:
            mean_spread = np.nanmean(z500.ensemble_spread)
            # Lower spread = higher confidence
            spread_score = max(0, 100 - mean_spread * 10)
            scores.append(spread_score)

        # Weight distribution (more even = less confident single best)
        weight_values = [mw.combined_weight for mw in model_weights]
        weight_std = np.std(weight_values)
        weight_spread_score = 50 + weight_std * 50  # Higher spread = more differentiation
        scores.append(min(100, weight_spread_score))

        # Lead time decay
        lead_time_factor = max(0.3, 1.0 - 0.003 * forecast_hour)
        lead_time_score = lead_time_factor * 100
        scores.append(lead_time_score)

        # Model agreement (from weights)
        ai_confidence_mean = np.mean([mw.ai_confidence_weight for mw in model_weights])
        agreement_score = ai_confidence_mean * 100
        scores.append(agreement_score)

        overall = np.mean(scores)

        return ConfidenceScore(
            overall_score=float(overall),
            temporal_consistency=float(scores[0] if len(scores) > 0 else 70),
            spatial_coherence=float(scores[1] if len(scores) > 1 else 70),
            physical_consistency=float(scores[2] if len(scores) > 2 else 70),
            ensemble_agreement=float(scores[3] if len(scores) > 3 else 70),
        )

    def _identify_scenarios(
        self,
        model_data_dict: dict[NWPModel, ModelData],
        analysis_results: dict[NWPModel, AnalysisResult],
        forecast_hour: int,
    ) -> list[EnsembleScenario]:
        """Identify distinct forecast scenarios from model clustering."""
        scenarios = []

        # Simple approach: group by dominant pattern
        pattern_groups: dict = {}

        for model, analysis in analysis_results.items():
            if analysis.physical_interpretation:
                pattern = analysis.physical_interpretation.dominant_pattern
                if pattern not in pattern_groups:
                    pattern_groups[pattern] = []
                pattern_groups[pattern].append(model)

        # Convert to scenarios
        total_models = len(analysis_results)
        scenario_id = 1

        for pattern, models in sorted(
            pattern_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        ):
            probability = len(models) / total_models

            # Generate description
            if pattern.value == "blocking_high" or pattern.value == "omega_block":
                desc = "Blocking scenario with persistent conditions"
            elif pattern.value == "trough":
                desc = "Progressive trough scenario with active weather"
            elif pattern.value == "ridge":
                desc = "Ridge-dominated scenario with quieter conditions"
            elif pattern.value == "cut_off_low":
                desc = "Cut-off low scenario with slow-moving system"
            else:
                desc = f"{pattern.value.replace('_', ' ').title()} scenario"

            scenarios.append(EnsembleScenario(
                scenario_id=scenario_id,
                probability=probability,
                description=desc,
                supporting_models=models,
                dominant_pattern=pattern,
                key_features=[f"Supported by {len(models)} model(s)"],
            ))
            scenario_id += 1

        return scenarios

    def _generate_ensemble_summary(
        self,
        patterns: list,
        interpretation: PhysicalInterpretation,
        confidence: ConfidenceScore,
    ) -> str:
        """Generate ensemble synoptic summary."""
        parts = []

        # Dominant pattern
        if patterns:
            dominant = patterns[0]
            parts.append(
                f"The ensemble depicts a {dominant.description} "
                f"centered near {dominant.center_lat:.0f}°N"
            )

        # Flow regime
        if interpretation.jet_position:
            parts.append(f"The jet stream {interpretation.jet_position}")

        # Confidence statement
        if confidence.overall_score > 75:
            parts.append("Model agreement is strong, indicating high forecast confidence")
        elif confidence.overall_score > 50:
            parts.append("Model agreement is moderate, with some spread in solutions")
        else:
            parts.append("Significant model spread exists, indicating lower confidence")

        return ". ".join(parts) + "."

    def _extract_key_findings(
        self,
        patterns: list,
        interpretation: PhysicalInterpretation,
        model_weights: list[WeightedModel],
    ) -> list[str]:
        """Extract key ensemble findings."""
        findings = []

        # Pattern findings
        for pattern in patterns[:2]:
            findings.append(
                f"Ensemble mean shows {pattern.pattern_type.value.replace('_', ' ')}"
            )

        # Mechanism
        findings.append(
            f"Primary driving mechanism: "
            f"{interpretation.primary_mechanism.value.replace('_', ' ')}"
        )

        # Weight distribution
        highest_weight = max(model_weights, key=lambda x: x.combined_weight)
        findings.append(
            f"Highest weighted model: {highest_weight.model.value.upper()} "
            f"(weight={highest_weight.combined_weight:.2f})"
        )

        return findings

    def _generate_warnings(
        self,
        confidence: ConfidenceScore,
        model_weights: list[WeightedModel],
        forecast_hour: int,
    ) -> list[str]:
        """Generate ensemble warnings."""
        warnings = []

        if confidence.overall_score < 40:
            warnings.append(
                f"Low ensemble confidence ({confidence.overall_score:.0f}/100) - "
                "consider multiple scenarios"
            )

        if forecast_hour > 168:
            warnings.append(
                "Extended range (>168h): Ensemble spread typically increases"
            )

        # Check for low agreement
        if confidence.ensemble_agreement < 50:
            warnings.append(
                "Significant model disagreement detected"
            )

        return warnings

    def _calculate_final_score(
        self,
        confidence: ConfidenceScore,
        scenarios: list[EnsembleScenario],
    ) -> float:
        """Calculate final ensemble confidence score."""
        base_score = confidence.overall_score

        # Adjust based on scenario distribution
        if scenarios:
            dominant_prob = scenarios[0].probability
            # Higher dominant probability = more confidence
            scenario_factor = 0.7 + 0.3 * dominant_prob
            base_score *= scenario_factor

        return float(max(0, min(100, base_score)))
