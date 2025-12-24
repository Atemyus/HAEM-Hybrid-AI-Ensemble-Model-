"""
Multi-Model Comparison Module.

This module provides algorithms for comparing multiple NWP models
and synthesizing their outputs into coherent scenarios.

Features:
- Field-by-field agreement metrics
- Scenario identification
- Divergence analysis
- Timing spread assessment
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from haem.models.meteorological import (
    GeopotentialField,
    ModelData,
    NWPModel,
    SynopticPattern,
    SynopticPatternType,
)
from haem.models.analysis import ModelComparison

logger = logging.getLogger(__name__)


class MultiModelComparator:
    """
    Compares and synthesizes multiple NWP model outputs.

    Provides:
    - Agreement metrics across models
    - Scenario clustering
    - Divergence identification
    - Consensus pattern determination
    """

    def __init__(self, agreement_threshold: float = 0.7):
        """
        Initialize comparator.

        Args:
            agreement_threshold: Threshold for determining agreement (0-1)
        """
        self.agreement_threshold = agreement_threshold

    def compare(
        self,
        model_data_dict: dict[NWPModel, ModelData],
        forecast_hour: int,
    ) -> ModelComparison:
        """
        Compare all models at a specific forecast time.

        Args:
            model_data_dict: Dict mapping model to ModelData
            forecast_hour: Forecast hour to compare

        Returns:
            ModelComparison object
        """
        models = list(model_data_dict.keys())

        if len(models) < 2:
            raise ValueError("Need at least 2 models to compare")

        # Get valid time from first model
        first_model = models[0]
        valid_time = model_data_dict[first_model].run.init_time + np.timedelta64(forecast_hour, 'h')
        valid_time = datetime.utcfromtimestamp(valid_time.astype('datetime64[s]').astype('int'))

        # Compute field-specific agreements
        z500_agreement = self._compute_field_agreement(
            model_data_dict, forecast_hour, "z500"
        )
        slp_agreement = self._compute_field_agreement(
            model_data_dict, forecast_hour, "slp"
        )
        t850_agreement = self._compute_field_agreement(
            model_data_dict, forecast_hour, "t850"
        )

        # Overall agreement
        overall_agreement = np.mean([z500_agreement, slp_agreement, t850_agreement])

        # Check pattern consensus
        pattern_consensus, consensus_pattern = self._check_pattern_consensus(
            model_data_dict, forecast_hour
        )

        # Identify major divergences
        major_divergences, divergence_locations = self._identify_divergences(
            model_data_dict, forecast_hour
        )

        # Compute timing spread
        timing_spread = self._compute_timing_spread(model_data_dict, forecast_hour)

        # Generate scenario descriptions
        dominant_scenario, alternative_scenarios = self._generate_scenarios(
            model_data_dict, forecast_hour, z500_agreement
        )

        return ModelComparison(
            valid_time=valid_time,
            forecast_hour=forecast_hour,
            models_compared=models,
            overall_agreement=float(overall_agreement),
            z500_agreement=float(z500_agreement),
            slp_agreement=float(slp_agreement),
            t850_agreement=float(t850_agreement),
            pattern_consensus=pattern_consensus,
            consensus_pattern=consensus_pattern,
            major_divergences=major_divergences,
            divergence_locations=divergence_locations,
            timing_spread_hours=timing_spread,
            dominant_scenario=dominant_scenario,
            alternative_scenarios=alternative_scenarios,
        )

    def _compute_field_agreement(
        self,
        model_data_dict: dict[NWPModel, ModelData],
        forecast_hour: int,
        field_name: str,
    ) -> float:
        """Compute agreement metric for a specific field across models."""
        fields = []

        for model, data in model_data_dict.items():
            field_dict = getattr(data, field_name)
            if forecast_hour in field_dict:
                fields.append(field_dict[forecast_hour].data)

        if len(fields) < 2:
            return 100.0  # Can't compare with less than 2

        # Compute pairwise correlations
        correlations = []
        for i in range(len(fields)):
            for j in range(i + 1, len(fields)):
                corr = np.corrcoef(fields[i].ravel(), fields[j].ravel())[0, 1]
                correlations.append(corr)

        # Mean correlation as agreement
        mean_corr = np.mean(correlations)

        # Convert to 0-100 scale
        return float(max(0, min(100, (mean_corr + 1) / 2 * 100)))

    def _check_pattern_consensus(
        self,
        model_data_dict: dict[NWPModel, ModelData],
        forecast_hour: int,
    ) -> tuple[bool, Optional[SynopticPatternType]]:
        """Check if models agree on the dominant synoptic pattern."""
        pattern_counts: dict[SynopticPatternType, int] = {}

        for model, data in model_data_dict.items():
            if forecast_hour in data.patterns:
                patterns = data.patterns[forecast_hour]
                if patterns:
                    dominant = patterns[0].pattern_type
                    pattern_counts[dominant] = pattern_counts.get(dominant, 0) + 1

        if not pattern_counts:
            return False, None

        # Find most common pattern
        most_common = max(pattern_counts, key=pattern_counts.get)
        count = pattern_counts[most_common]
        total = sum(pattern_counts.values())

        # Consensus if majority agree
        is_consensus = count >= total * 0.6

        return is_consensus, most_common if is_consensus else None

    def _identify_divergences(
        self,
        model_data_dict: dict[NWPModel, ModelData],
        forecast_hour: int,
    ) -> tuple[list[str], list[tuple[float, float]]]:
        """Identify regions and features where models diverge."""
        divergences = []
        locations = []

        # Collect Z500 fields
        z500_fields: dict[NWPModel, NDArray] = {}
        lats = None
        lons = None

        for model, data in model_data_dict.items():
            if forecast_hour in data.z500:
                field = data.z500[forecast_hour]
                z500_fields[model] = field.data
                if lats is None:
                    lats = field.lats
                    lons = field.lons

        if len(z500_fields) < 2 or lats is None:
            return divergences, locations

        # Stack all fields
        stacked = np.stack(list(z500_fields.values()), axis=0)

        # Compute spread (std across models)
        spread = np.std(stacked, axis=0)

        # Find high-spread regions
        threshold = np.percentile(spread, 90)
        high_spread_mask = spread > threshold

        # Find connected regions of high spread
        from scipy.ndimage import label

        labeled, num_features = label(high_spread_mask)

        for region_id in range(1, num_features + 1):
            region_mask = labeled == region_id
            if np.sum(region_mask) < 50:  # Skip small regions
                continue

            # Find center
            region_indices = np.where(region_mask)
            center_i = int(np.mean(region_indices[0]))
            center_j = int(np.mean(region_indices[1]))

            center_lat = float(lats[center_i])
            center_lon = float(lons[center_j])

            # Characterize divergence
            region_spread = np.mean(spread[region_mask])
            divergences.append(
                f"Model spread of {region_spread:.1f} dam near "
                f"{center_lat:.0f}°N, {abs(center_lon):.0f}°{'W' if center_lon < 0 else 'E'}"
            )
            locations.append((center_lat, center_lon))

        # Also check for specific pattern divergences
        if len(divergences) == 0:
            # Check if any model is an outlier
            model_means = {m: np.mean(f) for m, f in z500_fields.items()}
            overall_mean = np.mean(list(model_means.values()))
            overall_std = np.std(list(model_means.values()))

            for model, mean in model_means.items():
                if abs(mean - overall_mean) > 2 * overall_std:
                    divergences.append(
                        f"{model.value.upper()} shows systematic bias "
                        f"({mean - overall_mean:+.1f} dam vs consensus)"
                    )

        return divergences, locations

    def _compute_timing_spread(
        self,
        model_data_dict: dict[NWPModel, ModelData],
        forecast_hour: int,
    ) -> float:
        """Estimate timing spread in key feature evolution."""
        # Simple approach: look at gradient of Z500 to estimate feature speed
        speeds = []

        for model, data in model_data_dict.items():
            if forecast_hour in data.z500 and (forecast_hour + 6) in data.z500:
                current = data.z500[forecast_hour].data
                future = data.z500[forecast_hour + 6].data

                # Estimate eastward movement from correlation lag
                # This is a simplified approach
                change = future - current
                mean_change = np.mean(np.abs(change))
                speeds.append(mean_change)

        if len(speeds) < 2:
            return 0.0

        # Spread in speeds indicates timing uncertainty
        speed_std = np.std(speeds)

        # Convert to approximate hours of timing uncertainty
        # (rough heuristic)
        timing_spread = speed_std * 3  # hours

        return float(timing_spread)

    def _generate_scenarios(
        self,
        model_data_dict: dict[NWPModel, ModelData],
        forecast_hour: int,
        agreement: float,
    ) -> tuple[str, list[str]]:
        """Generate scenario descriptions based on model clustering."""
        models = list(model_data_dict.keys())

        if agreement > 85:
            # High agreement - single scenario
            dominant = self._describe_consensus_scenario(model_data_dict, forecast_hour)
            return dominant, []

        # Cluster models by Z500 similarity
        z500_fields = []
        model_order = []

        for model, data in model_data_dict.items():
            if forecast_hour in data.z500:
                z500_fields.append(data.z500[forecast_hour].data.ravel())
                model_order.append(model)

        if len(z500_fields) < 3:
            dominant = self._describe_consensus_scenario(model_data_dict, forecast_hour)
            return dominant, []

        # Hierarchical clustering
        z500_array = np.array(z500_fields)
        distances = pdist(z500_array, metric="correlation")
        linkage_matrix = linkage(distances, method="ward")

        # Cut into 2-3 clusters
        n_clusters = 2 if agreement > 60 else 3
        clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

        # Generate scenarios per cluster
        scenarios = []
        cluster_sizes = []

        for cluster_id in range(1, n_clusters + 1):
            cluster_models = [
                model_order[i]
                for i in range(len(model_order))
                if clusters[i] == cluster_id
            ]
            if cluster_models:
                scenario = self._describe_cluster_scenario(
                    model_data_dict, forecast_hour, cluster_models
                )
                scenarios.append(scenario)
                cluster_sizes.append(len(cluster_models))

        # Sort by cluster size (largest is dominant)
        sorted_scenarios = [
            s for _, s in sorted(
                zip(cluster_sizes, scenarios),
                reverse=True
            )
        ]

        dominant = sorted_scenarios[0] if sorted_scenarios else "No clear scenario identified"
        alternatives = sorted_scenarios[1:] if len(sorted_scenarios) > 1 else []

        return dominant, alternatives

    def _describe_consensus_scenario(
        self,
        model_data_dict: dict[NWPModel, ModelData],
        forecast_hour: int,
    ) -> str:
        """Describe the consensus scenario when models agree."""
        # Average Z500 pattern description
        z500_mean = None
        count = 0

        for model, data in model_data_dict.items():
            if forecast_hour in data.z500:
                field = data.z500[forecast_hour].data
                if z500_mean is None:
                    z500_mean = field.copy()
                else:
                    z500_mean += field
                count += 1

        if z500_mean is not None and count > 0:
            z500_mean /= count
            mean_height = np.mean(z500_mean)

            if mean_height > 555:
                pattern_desc = "amplified ridge-dominated pattern"
            elif mean_height < 545:
                pattern_desc = "active trough-dominated pattern"
            else:
                pattern_desc = "near-normal zonal flow pattern"

            return (
                f"Models show strong consensus on {pattern_desc}. "
                f"High confidence in synoptic evolution."
            )

        return "Consensus scenario based on model agreement"

    def _describe_cluster_scenario(
        self,
        model_data_dict: dict[NWPModel, ModelData],
        forecast_hour: int,
        cluster_models: list[NWPModel],
    ) -> str:
        """Describe a scenario from a cluster of models."""
        model_names = [m.value.upper() for m in cluster_models]
        model_str = ", ".join(model_names)

        # Characterize this cluster's solution
        z500_fields = []
        for model in cluster_models:
            if model in model_data_dict:
                data = model_data_dict[model]
                if forecast_hour in data.z500:
                    z500_fields.append(data.z500[forecast_hour].data)

        if z500_fields:
            cluster_mean = np.mean(z500_fields, axis=0)
            mean_height = np.mean(cluster_mean)

            if mean_height > 555:
                char = "more amplified/ridging"
            elif mean_height < 545:
                char = "more progressive/troughing"
            else:
                char = "near-normal"

            return f"{model_str} favor {char} solution"

        return f"Scenario supported by {model_str}"

    def compute_ensemble_statistics(
        self,
        model_data_dict: dict[NWPModel, ModelData],
        forecast_hour: int,
        field_name: str = "z500",
    ) -> dict:
        """
        Compute ensemble statistics across models.

        Returns dict with mean, spread, min, max, and percentiles.
        """
        fields = []

        for model, data in model_data_dict.items():
            field_dict = getattr(data, field_name)
            if forecast_hour in field_dict:
                fields.append(field_dict[forecast_hour].data)

        if not fields:
            return {}

        stacked = np.stack(fields, axis=0)

        return {
            "mean": np.mean(stacked, axis=0),
            "std": np.std(stacked, axis=0),
            "min": np.min(stacked, axis=0),
            "max": np.max(stacked, axis=0),
            "p10": np.percentile(stacked, 10, axis=0),
            "p25": np.percentile(stacked, 25, axis=0),
            "p50": np.percentile(stacked, 50, axis=0),
            "p75": np.percentile(stacked, 75, axis=0),
            "p90": np.percentile(stacked, 90, axis=0),
            "n_models": len(fields),
        }
