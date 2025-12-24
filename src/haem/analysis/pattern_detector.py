"""
Synoptic Pattern Detection Module.

This module provides algorithms for identifying synoptic-scale
atmospheric patterns from NWP model fields.

Detected patterns include:
- Troughs and ridges
- Cut-off lows
- Blocking highs (omega, Rex)
- Jet streaks
- Frontal zones
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    gaussian_filter,
    label,
    maximum_filter,
    minimum_filter,
)

from haem.models.meteorological import (
    GeopotentialField,
    PressureField,
    SynopticPattern,
    SynopticPatternType,
    TemperatureField,
)

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    Synoptic pattern detection using gradient and curvature analysis.

    Uses objective algorithms to identify:
    - Wave patterns (troughs/ridges) from geopotential curvature
    - Pressure centers from local extrema
    - Blocking patterns from persistence and geometry
    - Frontal zones from thermal gradients
    """

    def __init__(
        self,
        smooth_sigma: float = 2.0,
        min_pattern_size_km: float = 500.0,
    ):
        """
        Initialize pattern detector.

        Args:
            smooth_sigma: Gaussian smoothing sigma for noise reduction
            min_pattern_size_km: Minimum pattern size to detect
        """
        self.smooth_sigma = smooth_sigma
        self.min_pattern_size_km = min_pattern_size_km

    def detect_all_patterns(
        self,
        z500: Optional[GeopotentialField] = None,
        slp: Optional[PressureField] = None,
        t850: Optional[TemperatureField] = None,
    ) -> list[SynopticPattern]:
        """
        Detect all synoptic patterns from available fields.

        Args:
            z500: 500 hPa geopotential height
            slp: Mean sea level pressure
            t850: 850 hPa temperature

        Returns:
            List of detected SynopticPattern objects
        """
        patterns = []

        if z500 is not None:
            patterns.extend(self._detect_wave_patterns(z500))
            patterns.extend(self._detect_cutoff_lows(z500))
            patterns.extend(self._detect_blocking(z500))

        if slp is not None:
            patterns.extend(self._detect_pressure_centers(slp))

        if t850 is not None and z500 is not None:
            patterns.extend(self._detect_frontal_zones(t850, z500))

        # Sort by intensity (strongest first)
        patterns.sort(key=lambda p: p.intensity, reverse=True)

        return patterns

    def _detect_wave_patterns(
        self,
        z500: GeopotentialField,
    ) -> list[SynopticPattern]:
        """
        Detect troughs and ridges from Z500 curvature.

        Troughs: Local minima in meridional direction
        Ridges: Local maxima in meridional direction
        """
        patterns = []

        # Smooth field
        data = gaussian_filter(z500.data, sigma=self.smooth_sigma)
        lats = z500.lats
        lons = z500.lons

        # Compute second derivative in zonal direction (curvature)
        dx = np.mean(np.diff(lons))
        d2z_dx2 = np.gradient(np.gradient(data, dx, axis=1), dx, axis=1)

        # Compute zonal gradient for trough/ridge identification
        dz_dx = np.gradient(data, dx, axis=1)

        # Find zero-crossings of zonal gradient (trough/ridge axes)
        for i in range(1, data.shape[0] - 1):
            for j in range(1, data.shape[1] - 1):
                # Skip if outside mid-latitudes
                if not (30 <= lats[i] <= 65):
                    continue

                # Check for sign change in zonal gradient
                if dz_dx[i, j - 1] * dz_dx[i, j + 1] < 0:
                    lat = lats[i]
                    lon = lons[j]

                    # Determine if trough or ridge based on curvature
                    curvature = d2z_dx2[i, j]

                    if curvature > 0.001:  # Trough (concave up)
                        # Calculate amplitude (difference from zonal mean)
                        zonal_mean = np.mean(data[i, :])
                        amplitude = abs(data[i, j] - zonal_mean)

                        if amplitude > 2:  # Significant amplitude
                            patterns.append(SynopticPattern(
                                pattern_type=SynopticPatternType.TROUGH,
                                center_lat=float(lat),
                                center_lon=float(lon),
                                intensity=float(amplitude),
                                extent_km=1000.0,  # Estimate
                                amplitude=float(amplitude),
                                valid_time=z500.valid_time,
                                associated_phenomena=["cold air advection", "upper divergence"],
                            ))

                    elif curvature < -0.001:  # Ridge (concave down)
                        zonal_mean = np.mean(data[i, :])
                        amplitude = abs(data[i, j] - zonal_mean)

                        if amplitude > 2:
                            patterns.append(SynopticPattern(
                                pattern_type=SynopticPatternType.RIDGE,
                                center_lat=float(lat),
                                center_lon=float(lon),
                                intensity=float(amplitude),
                                extent_km=1000.0,
                                amplitude=float(amplitude),
                                valid_time=z500.valid_time,
                                associated_phenomena=["warm air advection", "subsidence"],
                            ))

        # Merge nearby patterns of same type
        patterns = self._merge_nearby_patterns(patterns, threshold_km=500)

        return patterns

    def _detect_cutoff_lows(
        self,
        z500: GeopotentialField,
    ) -> list[SynopticPattern]:
        """
        Detect cut-off low pressure systems.

        Cut-offs are characterized by:
        - Closed contours at 500 hPa
        - Separation from main westerly flow
        """
        patterns = []

        data = gaussian_filter(z500.data, sigma=self.smooth_sigma)
        lats = z500.lats
        lons = z500.lons

        # Find local minima
        min_filtered = minimum_filter(data, size=7)
        local_min_mask = (data == min_filtered)

        # Check each local minimum for cut-off characteristics
        min_indices = np.where(local_min_mask)
        for i, j in zip(min_indices[0], min_indices[1]):
            lat = lats[i]
            lon = lons[j]

            # Only consider mid-latitudes
            if not (25 <= lat <= 55):
                continue

            center_value = data[i, j]

            # Check for closed contours (all surrounding values higher)
            if i > 3 and i < data.shape[0] - 3 and j > 3 and j < data.shape[1] - 3:
                surrounding = data[i - 3:i + 4, j - 3:j + 4]
                min_surrounding = np.min(surrounding[surrounding != center_value])

                depth = min_surrounding - center_value

                if depth > 3:  # Significant closed low
                    # Check isolation from main flow (latitude check)
                    zonal_mean = np.mean(data[i, :])
                    isolation = zonal_mean - center_value

                    if isolation > 5:  # Well-isolated from mean flow
                        patterns.append(SynopticPattern(
                            pattern_type=SynopticPatternType.CUT_OFF_LOW,
                            center_lat=float(lat),
                            center_lon=float(lon),
                            intensity=float(depth),
                            extent_km=800.0,
                            valid_time=z500.valid_time,
                            associated_phenomena=[
                                "slow movement",
                                "potential precipitation",
                                "atmospheric instability",
                            ],
                        ))

        return patterns

    def _detect_blocking(
        self,
        z500: GeopotentialField,
    ) -> list[SynopticPattern]:
        """
        Detect blocking high pressure patterns.

        Blocking criteria:
        - Reversed meridional gradient (higher heights poleward)
        - Large amplitude and persistence
        """
        patterns = []

        data = gaussian_filter(z500.data, sigma=self.smooth_sigma)
        lats = z500.lats
        lons = z500.lons

        # Compute meridional gradient
        dy = np.mean(np.diff(lats))
        dz_dy = np.gradient(data, dy, axis=0)

        # Look for reversed gradient (dZ/dy > 0, normally negative)
        for i in range(5, data.shape[0] - 5):
            for j in range(data.shape[1]):
                lat = lats[i]

                # Focus on high latitudes where blocking occurs
                if not (50 <= lat <= 70):
                    continue

                # Check for positive meridional gradient (reversed)
                if dz_dy[i, j] > 0.3:
                    # Verify it's a local maximum
                    if i > 0 and i < data.shape[0] - 1:
                        if data[i, j] > data[i - 1, j] and data[i, j] > data[i + 1, j]:
                            # Check amplitude
                            local_mean = np.mean(data[i - 5:i + 6, j])
                            amplitude = data[i, j] - local_mean

                            if amplitude > 4:
                                # Check for omega block pattern
                                # (high surrounded by lows to east and west)
                                is_omega = False
                                if j > 10 and j < data.shape[1] - 10:
                                    west_trough = np.min(data[i, j - 10:j - 3])
                                    east_trough = np.min(data[i, j + 3:j + 10])
                                    if (data[i, j] - west_trough > 3 and
                                            data[i, j] - east_trough > 3):
                                        is_omega = True

                                pattern_type = (
                                    SynopticPatternType.OMEGA_BLOCK if is_omega
                                    else SynopticPatternType.BLOCKING_HIGH
                                )

                                patterns.append(SynopticPattern(
                                    pattern_type=pattern_type,
                                    center_lat=float(lat),
                                    center_lon=float(lons[j]),
                                    intensity=float(amplitude),
                                    extent_km=1500.0,
                                    amplitude=float(amplitude),
                                    valid_time=z500.valid_time,
                                    associated_phenomena=[
                                        "flow blocking",
                                        "persistent weather",
                                        "meridional heat transport",
                                    ],
                                ))

        # Merge nearby blocking patterns
        patterns = self._merge_nearby_patterns(patterns, threshold_km=1000)

        return patterns

    def _detect_pressure_centers(
        self,
        slp: PressureField,
    ) -> list[SynopticPattern]:
        """Detect surface pressure centers (cyclones and anticyclones)."""
        patterns = []

        data = gaussian_filter(slp.data, sigma=self.smooth_sigma)
        lats = slp.lats
        lons = slp.lons

        centers = slp.find_pressure_centers(
            threshold_low=1008.0,
            threshold_high=1020.0,
        )

        for lat, lon, pressure in centers["lows"]:
            # Calculate intensity (depth below 1013)
            intensity = 1013 - pressure

            patterns.append(SynopticPattern(
                pattern_type=SynopticPatternType.CYCLONIC_VORTEX,
                center_lat=lat,
                center_lon=lon,
                intensity=intensity,
                extent_km=600.0,
                valid_time=slp.valid_time,
                associated_phenomena=[
                    "convergence",
                    "ascending motion",
                    "precipitation potential",
                ],
            ))

        for lat, lon, pressure in centers["highs"]:
            intensity = pressure - 1013

            patterns.append(SynopticPattern(
                pattern_type=SynopticPatternType.ANTICYCLONIC_VORTEX,
                center_lat=lat,
                center_lon=lon,
                intensity=intensity,
                extent_km=800.0,
                valid_time=slp.valid_time,
                associated_phenomena=[
                    "divergence",
                    "subsidence",
                    "fair weather",
                ],
            ))

        return patterns

    def _detect_frontal_zones(
        self,
        t850: TemperatureField,
        z500: GeopotentialField,
    ) -> list[SynopticPattern]:
        """
        Detect frontal zones from thermal gradient.

        Fronts identified by strong thermal gradient at 850 hPa.
        """
        patterns = []

        data = gaussian_filter(t850.data, sigma=self.smooth_sigma)
        lats = t850.lats
        lons = t850.lons

        # Compute thermal gradient magnitude
        dy = np.mean(np.diff(lats)) * 111  # km per degree
        dx = np.mean(np.diff(lons)) * 111 * np.cos(np.deg2rad(np.mean(lats)))

        dT_dy = np.gradient(data, dy, axis=0)
        dT_dx = np.gradient(data, dx, axis=1)
        grad_mag = np.sqrt(dT_dx**2 + dT_dy**2)

        # Threshold for frontal zone (K/100km)
        front_threshold = 0.04

        # Find frontal zones
        front_mask = grad_mag > front_threshold

        # Label connected regions
        labeled, num_features = label(front_mask)

        for region_id in range(1, num_features + 1):
            region_mask = labeled == region_id
            region_size = np.sum(region_mask)

            if region_size < 100:  # Skip small regions
                continue

            # Find center of frontal zone
            region_indices = np.where(region_mask)
            center_i = int(np.mean(region_indices[0]))
            center_j = int(np.mean(region_indices[1]))

            # Calculate average gradient in region
            avg_gradient = float(np.mean(grad_mag[region_mask]))

            # Estimate extent
            extent_km = float(np.sqrt(region_size) * dx)

            patterns.append(SynopticPattern(
                pattern_type=SynopticPatternType.FRONTAL_ZONE,
                center_lat=float(lats[center_i]),
                center_lon=float(lons[center_j]),
                intensity=avg_gradient * 100,  # K per 100km
                extent_km=extent_km,
                valid_time=t850.valid_time,
                associated_phenomena=[
                    "thermal contrast",
                    "potential frontogenesis",
                    "baroclinic instability",
                ],
            ))

        return patterns

    def _merge_nearby_patterns(
        self,
        patterns: list[SynopticPattern],
        threshold_km: float = 500,
    ) -> list[SynopticPattern]:
        """Merge nearby patterns of the same type."""
        if len(patterns) <= 1:
            return patterns

        merged = []
        used = set()

        for i, p1 in enumerate(patterns):
            if i in used:
                continue

            # Find all nearby patterns of same type
            group = [p1]
            for j, p2 in enumerate(patterns[i + 1:], start=i + 1):
                if j in used:
                    continue
                if p2.pattern_type != p1.pattern_type:
                    continue

                # Calculate distance
                dlat = p2.center_lat - p1.center_lat
                dlon = p2.center_lon - p1.center_lon
                dist_km = np.sqrt(
                    (dlat * 111)**2 +
                    (dlon * 111 * np.cos(np.deg2rad(p1.center_lat)))**2
                )

                if dist_km < threshold_km:
                    group.append(p2)
                    used.add(j)

            # Merge group into single pattern (use strongest)
            strongest = max(group, key=lambda p: p.intensity)
            merged.append(strongest)

        return merged
