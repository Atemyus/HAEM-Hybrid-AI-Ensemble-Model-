"""
Physical-Dynamical Interpretation Engine.

This module provides causal explanations for atmospheric patterns
using principles of synoptic and dynamic meteorology.

All interpretations are grounded in:
- Quasi-geostrophic theory
- Potential vorticity dynamics
- Atmospheric thermodynamics
- Rossby wave theory
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from haem.models.meteorological import (
    GeopotentialField,
    PressureField,
    SynopticPattern,
    SynopticPatternType,
    TemperatureField,
)
from haem.models.analysis import (
    PhysicalInterpretation,
    PhysicalMechanism,
)

logger = logging.getLogger(__name__)


class PhysicalInterpreter:
    """
    Generates physical-dynamical interpretations of atmospheric patterns.

    Uses atmospheric dynamics principles to explain:
    - Why patterns exist (causal mechanisms)
    - How patterns will evolve (prognostic reasoning)
    - What phenomena are associated (impacts)
    """

    # Climatological reference values
    CLIMATOLOGY = {
        "z500_mean_45N": 552,  # dam at 45°N
        "z500_std": 8,         # dam typical variability
        "jet_speed_threshold": 30,  # m/s for jet identification
    }

    def interpret(
        self,
        z500: Optional[GeopotentialField] = None,
        z850: Optional[GeopotentialField] = None,
        t850: Optional[TemperatureField] = None,
        slp: Optional[PressureField] = None,
        patterns: Optional[list[SynopticPattern]] = None,
        forecast_hour: int = 0,
    ) -> PhysicalInterpretation:
        """
        Generate comprehensive physical interpretation.

        Args:
            z500: 500 hPa geopotential height
            z850: 850 hPa geopotential height
            t850: 850 hPa temperature
            slp: Mean sea level pressure
            patterns: Detected synoptic patterns
            forecast_hour: Forecast lead time

        Returns:
            PhysicalInterpretation object
        """
        patterns = patterns or []
        valid_time = z500.valid_time if z500 else datetime.utcnow()

        # Identify dominant pattern
        dominant_pattern = self._identify_dominant_pattern(patterns)

        # Identify secondary patterns
        secondary_patterns = [
            p.pattern_type for p in patterns[1:4]
        ] if len(patterns) > 1 else []

        # Determine primary mechanism
        primary_mechanism = self._determine_primary_mechanism(
            dominant_pattern, patterns, z500, t850
        )

        # Determine contributing mechanisms
        contributing_mechanisms = self._determine_contributing_mechanisms(
            patterns, z500, t850, slp
        )

        # Analyze jet stream
        jet_position, jet_intensity = self._analyze_jet_stream(z500)

        # Analyze air masses
        air_mass_description = self._analyze_air_masses(t850, slp)

        # Analyze thermal advection
        thermal_advection = self._analyze_thermal_advection(t850, z500)

        # Describe evolution
        evolution_description = self._describe_evolution(
            dominant_pattern, primary_mechanism, forecast_hour
        )

        # Expected changes
        expected_changes = self._predict_changes(
            patterns, primary_mechanism, forecast_hour
        )

        # Check for anomalous conditions
        is_anomalous, anomaly_description = self._check_anomalies(z500, t850)

        # Build causal explanation
        causal_explanation = self._build_causal_explanation(
            dominant_pattern,
            primary_mechanism,
            contributing_mechanisms,
            jet_position,
            thermal_advection,
        )

        return PhysicalInterpretation(
            valid_time=valid_time,
            forecast_hour=forecast_hour,
            dominant_pattern=dominant_pattern,
            secondary_patterns=secondary_patterns,
            primary_mechanism=primary_mechanism,
            contributing_mechanisms=contributing_mechanisms,
            jet_position=jet_position,
            jet_intensity=jet_intensity,
            air_mass_description=air_mass_description,
            thermal_advection=thermal_advection,
            evolution_description=evolution_description,
            expected_changes=expected_changes,
            is_anomalous=is_anomalous,
            anomaly_description=anomaly_description,
            causal_explanation=causal_explanation,
        )

    def _identify_dominant_pattern(
        self,
        patterns: list[SynopticPattern],
    ) -> SynopticPatternType:
        """Identify the dominant synoptic pattern."""
        if not patterns:
            return SynopticPatternType.ZONAL_FLOW

        # Prioritize certain pattern types
        priority = {
            SynopticPatternType.OMEGA_BLOCK: 10,
            SynopticPatternType.REX_BLOCK: 9,
            SynopticPatternType.BLOCKING_HIGH: 8,
            SynopticPatternType.CUT_OFF_LOW: 7,
            SynopticPatternType.CYCLONIC_VORTEX: 6,
            SynopticPatternType.TROUGH: 5,
            SynopticPatternType.RIDGE: 4,
        }

        # Score by intensity * priority
        scored = [
            (p, p.intensity * priority.get(p.pattern_type, 1))
            for p in patterns
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[0][0].pattern_type

    def _determine_primary_mechanism(
        self,
        dominant_pattern: SynopticPatternType,
        patterns: list[SynopticPattern],
        z500: Optional[GeopotentialField],
        t850: Optional[TemperatureField],
    ) -> PhysicalMechanism:
        """Determine the primary physical mechanism driving the pattern."""
        # Mechanism mapping based on pattern type
        mechanism_map = {
            SynopticPatternType.TROUGH: PhysicalMechanism.ROSSBY_WAVE_PROPAGATION,
            SynopticPatternType.RIDGE: PhysicalMechanism.ROSSBY_WAVE_PROPAGATION,
            SynopticPatternType.CUT_OFF_LOW: PhysicalMechanism.WAVE_BREAKING,
            SynopticPatternType.BLOCKING_HIGH: PhysicalMechanism.BLOCKING_DEVELOPMENT,
            SynopticPatternType.OMEGA_BLOCK: PhysicalMechanism.BLOCKING_DEVELOPMENT,
            SynopticPatternType.CYCLONIC_VORTEX: PhysicalMechanism.CYCLOGENESIS,
            SynopticPatternType.ANTICYCLONIC_VORTEX: PhysicalMechanism.ANTICYCLOGENESIS,
            SynopticPatternType.FRONTAL_ZONE: PhysicalMechanism.BAROCLINIC_INSTABILITY,
            SynopticPatternType.ZONAL_FLOW: PhysicalMechanism.JET_STREAM_DYNAMICS,
            SynopticPatternType.MERIDIONAL_FLOW: PhysicalMechanism.ROSSBY_WAVE_PROPAGATION,
        }

        base_mechanism = mechanism_map.get(
            dominant_pattern,
            PhysicalMechanism.ROSSBY_WAVE_PROPAGATION
        )

        # Check for thermal forcing
        if t850 is not None:
            grad_mag = self._compute_thermal_gradient_magnitude(t850)
            if np.nanmax(grad_mag) > 0.05:  # Strong baroclinicity
                if dominant_pattern in [
                    SynopticPatternType.TROUGH,
                    SynopticPatternType.CYCLONIC_VORTEX
                ]:
                    return PhysicalMechanism.BAROCLINIC_INSTABILITY

        return base_mechanism

    def _determine_contributing_mechanisms(
        self,
        patterns: list[SynopticPattern],
        z500: Optional[GeopotentialField],
        t850: Optional[TemperatureField],
        slp: Optional[PressureField],
    ) -> list[PhysicalMechanism]:
        """Identify contributing physical mechanisms."""
        mechanisms = []

        # Check for thermal advection
        if t850 is not None and z500 is not None:
            advection = self._compute_thermal_advection_type(t850, z500)
            if "warm" in advection.lower():
                mechanisms.append(PhysicalMechanism.WARM_AIR_ADVECTION)
            elif "cold" in advection.lower():
                mechanisms.append(PhysicalMechanism.COLD_AIR_ADVECTION)

        # Check for PV advection (implied by pattern types)
        has_trough = any(
            p.pattern_type == SynopticPatternType.TROUGH for p in patterns
        )
        if has_trough:
            mechanisms.append(PhysicalMechanism.POTENTIAL_VORTICITY_ADVECTION)

        # Check for jet dynamics
        if z500 is not None:
            if self._has_strong_gradient(z500):
                mechanisms.append(PhysicalMechanism.JET_STREAM_DYNAMICS)

        return mechanisms[:3]  # Limit to top 3

    def _analyze_jet_stream(
        self,
        z500: Optional[GeopotentialField],
    ) -> tuple[Optional[str], Optional[str]]:
        """Analyze jet stream position and intensity."""
        if z500 is None:
            return None, None

        data = z500.data
        lats = z500.lats
        lons = z500.lons

        # Compute zonal gradient (proxy for thermal wind / jet)
        dx = np.mean(np.diff(lons)) * 111 * 1000  # meters
        dz_dx = np.gradient(data * 10, dx, axis=1)  # meters

        # Find maximum gradient (jet core)
        max_grad_idx = np.unravel_index(np.argmax(np.abs(dz_dx)), dz_dx.shape)
        jet_lat = lats[max_grad_idx[0]]
        jet_lon = lons[max_grad_idx[1]]
        jet_strength = abs(dz_dx[max_grad_idx])

        # Determine position description
        if jet_lat > 55:
            lat_desc = "is positioned poleward of its climatological position"
        elif jet_lat < 40:
            lat_desc = "is positioned equatorward of its climatological position"
        else:
            lat_desc = "is near its climatological position"

        position = (
            f"{lat_desc} near {jet_lat:.0f}°N, "
            f"{abs(jet_lon):.0f}°{'W' if jet_lon < 0 else 'E'}"
        )

        # Determine intensity
        if jet_strength > 0.0015:
            intensity = "Very strong jet stream with enhanced baroclinicity"
        elif jet_strength > 0.001:
            intensity = "Moderately strong jet stream"
        else:
            intensity = "Weak jet stream conditions"

        return position, intensity

    def _analyze_air_masses(
        self,
        t850: Optional[TemperatureField],
        slp: Optional[PressureField],
    ) -> str:
        """Analyze air mass characteristics and origins."""
        if t850 is None:
            return "Insufficient data for air mass analysis"

        data = t850.data
        lats = t850.lats

        descriptions = []

        # Find warmest and coldest regions
        warm_idx = np.unravel_index(np.argmax(data), data.shape)
        cold_idx = np.unravel_index(np.argmin(data), data.shape)

        warm_t = data[warm_idx] - 273.15
        cold_t = data[cold_idx] - 273.15
        warm_lat = lats[warm_idx[0]]
        cold_lat = lats[cold_idx[0]]

        if warm_t > 15:
            descriptions.append(
                f"Tropical/subtropical air mass ({warm_t:.0f}°C at 850 hPa) "
                f"present in southern domain"
            )
        if cold_t < -10:
            descriptions.append(
                f"Arctic/polar air mass ({cold_t:.0f}°C at 850 hPa) "
                f"present in northern domain"
            )

        # Check temperature contrast
        contrast = warm_t - cold_t
        if contrast > 25:
            descriptions.append(
                "Strong thermal contrast indicates active baroclinic zone"
            )

        return ". ".join(descriptions) if descriptions else "Moderate thermal conditions"

    def _analyze_thermal_advection(
        self,
        t850: Optional[TemperatureField],
        z500: Optional[GeopotentialField],
    ) -> str:
        """Analyze thermal advection patterns."""
        if t850 is None or z500 is None:
            return "Insufficient data for advection analysis"

        return self._compute_thermal_advection_type(t850, z500)

    def _compute_thermal_advection_type(
        self,
        t850: TemperatureField,
        z500: GeopotentialField,
    ) -> str:
        """Compute thermal advection description."""
        t_data = t850.data
        z_data = z500.data
        lats = t850.lats
        lons = t850.lons

        # Compute gradients
        dy = np.mean(np.diff(lats)) * 111  # km
        dx = np.mean(np.diff(lons)) * 111 * np.cos(np.deg2rad(np.mean(lats)))

        dT_dx = np.gradient(t_data, dx, axis=1)
        dT_dy = np.gradient(t_data, dy, axis=0)

        # Approximate geostrophic wind from Z500
        dz_dx = np.gradient(z_data, dx, axis=1)
        dz_dy = np.gradient(z_data, dy, axis=0)

        # Geostrophic wind components (simplified)
        f = 2 * 7.29e-5 * np.sin(np.deg2rad(np.mean(lats)))
        u_g = -dz_dy * 9.8 / f if abs(f) > 1e-10 else 0
        v_g = dz_dx * 9.8 / f if abs(f) > 1e-10 else 0

        # Thermal advection: -V · ∇T
        advection = -(u_g * dT_dx + v_g * dT_dy)

        # Find regions of significant advection
        warm_adv = np.sum(advection > 0.0001)
        cold_adv = np.sum(advection < -0.0001)

        if warm_adv > cold_adv * 1.5:
            return "Predominantly warm air advection across domain"
        elif cold_adv > warm_adv * 1.5:
            return "Predominantly cold air advection across domain"
        else:
            return "Mixed advection patterns with localized warm and cold sectors"

    def _describe_evolution(
        self,
        dominant_pattern: SynopticPatternType,
        primary_mechanism: PhysicalMechanism,
        forecast_hour: int,
    ) -> str:
        """Describe expected pattern evolution."""
        evolutions = {
            SynopticPatternType.TROUGH: (
                "The trough is expected to propagate eastward under Rossby wave dynamics. "
                "Deepening possible if entering a region of positive vorticity advection aloft."
            ),
            SynopticPatternType.RIDGE: (
                "The ridge will amplify or translate eastward depending on upstream forcing. "
                "Blocking potential if amplification continues."
            ),
            SynopticPatternType.CUT_OFF_LOW: (
                "The cut-off low will move slowly or become quasi-stationary. "
                "Gradual weakening expected as it becomes vertically stacked."
            ),
            SynopticPatternType.BLOCKING_HIGH: (
                "The blocking pattern will persist, deflecting transient systems. "
                "Gradual erosion possible from persistent upstream baroclinic forcing."
            ),
            SynopticPatternType.OMEGA_BLOCK: (
                "The omega block will maintain quasi-stationary conditions. "
                "Systems will be forced around the block in a split-flow pattern."
            ),
            SynopticPatternType.CYCLONIC_VORTEX: (
                "The cyclone will intensify if baroclinic support continues, "
                "then occlude and weaken as it becomes barotropic."
            ),
        }

        base = evolutions.get(
            dominant_pattern,
            "Pattern evolution follows standard Rossby wave dynamics."
        )

        if forecast_hour > 120:
            base += " Extended range: increased uncertainty in exact position and intensity."

        return base

    def _predict_changes(
        self,
        patterns: list[SynopticPattern],
        primary_mechanism: PhysicalMechanism,
        forecast_hour: int,
    ) -> list[str]:
        """Predict expected changes."""
        changes = []

        # General changes based on mechanism
        if primary_mechanism == PhysicalMechanism.ROSSBY_WAVE_PROPAGATION:
            changes.append("Eastward propagation of wave pattern at ~10-15° longitude/day")

        if primary_mechanism == PhysicalMechanism.BAROCLINIC_INSTABILITY:
            changes.append("Potential cyclogenesis in baroclinic zone")

        if primary_mechanism == PhysicalMechanism.BLOCKING_DEVELOPMENT:
            changes.append("Reduced progressive weather pattern, persistent conditions")

        # Pattern-specific changes
        for pattern in patterns[:2]:
            if pattern.pattern_type == SynopticPatternType.CUT_OFF_LOW:
                changes.append("Cut-off low: slow movement, localized precipitation")
            elif pattern.pattern_type == SynopticPatternType.FRONTAL_ZONE:
                changes.append("Frontal zone: potential for organized precipitation")

        return changes

    def _check_anomalies(
        self,
        z500: Optional[GeopotentialField],
        t850: Optional[TemperatureField],
    ) -> tuple[bool, Optional[str]]:
        """Check for anomalous conditions relative to climatology."""
        if z500 is None:
            return False, None

        data = z500.data
        lats = z500.lats

        # Simple anomaly check: compare to expected meridional gradient
        mid_lat_mask = (lats > 40) & (lats < 55)
        if not np.any(mid_lat_mask):
            return False, None

        mid_lat_mean = np.mean(data[mid_lat_mask, :])
        clim_mean = self.CLIMATOLOGY["z500_mean_45N"]
        clim_std = self.CLIMATOLOGY["z500_std"]

        anomaly = (mid_lat_mean - clim_mean) / clim_std

        if abs(anomaly) > 2:
            sign = "positive" if anomaly > 0 else "negative"
            return True, f"Significant {sign} height anomaly ({anomaly:.1f}σ) at mid-latitudes"

        return False, None

    def _build_causal_explanation(
        self,
        dominant_pattern: SynopticPatternType,
        primary_mechanism: PhysicalMechanism,
        contributing_mechanisms: list[PhysicalMechanism],
        jet_position: Optional[str],
        thermal_advection: str,
    ) -> str:
        """Build complete causal chain explanation."""
        parts = []

        # Primary mechanism
        mechanism_explanations = {
            PhysicalMechanism.ROSSBY_WAVE_PROPAGATION: (
                "The pattern is driven by Rossby wave propagation, where the conservation "
                "of potential vorticity on the rotating Earth causes westward phase propagation "
                "relative to the mean flow, resulting in net eastward movement of the wave pattern."
            ),
            PhysicalMechanism.BAROCLINIC_INSTABILITY: (
                "Baroclinic instability is converting available potential energy in the "
                "meridional temperature gradient into kinetic energy of the developing system. "
                "This releases energy from the mean state thermal contrast."
            ),
            PhysicalMechanism.BLOCKING_DEVELOPMENT: (
                "The blocking pattern has developed through wave breaking and/or persistent "
                "Rossby wave amplification. The block creates a quasi-stationary high-amplitude "
                "pattern that deflects the normal westerly flow."
            ),
            PhysicalMechanism.WAVE_BREAKING: (
                "Anticyclonic wave breaking has occurred, with the trough wrapping up and "
                "becoming cut off from the main westerly current. This creates an isolated "
                "pool of high potential vorticity at low levels."
            ),
            PhysicalMechanism.CYCLOGENESIS: (
                "Cyclogenesis is occurring through the interaction of upper-level PV "
                "advection and low-level thermal advection, as described by the Sutcliffe "
                "development equations and quasi-geostrophic omega equation."
            ),
        }

        parts.append(mechanism_explanations.get(
            primary_mechanism,
            f"The pattern is governed by {primary_mechanism.value.replace('_', ' ')}."
        ))

        # Jet context
        if jet_position:
            parts.append(f"The jet stream {jet_position}, modulating the wave pattern intensity.")

        # Thermal forcing
        if "warm" in thermal_advection.lower():
            parts.append(
                "Warm air advection is contributing to height rises and potential "
                "ridge amplification or surface pressure falls ahead of the wave."
            )
        elif "cold" in thermal_advection.lower():
            parts.append(
                "Cold air advection is contributing to height falls and potential "
                "trough deepening or surface pressure rises behind the wave."
            )

        return " ".join(parts)

    def _compute_thermal_gradient_magnitude(
        self,
        t850: TemperatureField,
    ) -> NDArray[np.float64]:
        """Compute thermal gradient magnitude."""
        dy = np.mean(np.diff(t850.lats)) * 111
        dx = np.mean(np.diff(t850.lons)) * 111 * np.cos(np.deg2rad(np.mean(t850.lats)))

        dT_dy = np.gradient(t850.data, dy, axis=0)
        dT_dx = np.gradient(t850.data, dx, axis=1)

        return np.sqrt(dT_dx**2 + dT_dy**2)

    def _has_strong_gradient(
        self,
        z500: GeopotentialField,
        threshold: float = 0.08,
    ) -> bool:
        """Check if field has strong gradient (jet indication)."""
        dy = np.mean(np.diff(z500.lats))
        dz_dy = np.gradient(z500.data, dy, axis=0)
        return bool(np.max(np.abs(dz_dy)) > threshold)
