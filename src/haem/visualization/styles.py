"""
Meteorological Visualization Styles.

Defines color maps, contour levels, and styling for professional
meteorological map generation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class MeteoColorMaps:
    """
    Standard meteorological color maps.

    Based on operational meteorological visualization standards.
    """

    @staticmethod
    def z500_colormap() -> list[tuple[float, float, float]]:
        """
        Color map for Z500 geopotential height.

        Blue (low) -> Green -> Yellow -> Orange -> Red (high)
        """
        return [
            (0.1, 0.1, 0.5),   # Deep blue (low heights)
            (0.2, 0.4, 0.8),   # Blue
            (0.3, 0.6, 0.9),   # Light blue
            (0.4, 0.8, 0.4),   # Green
            (0.6, 0.9, 0.3),   # Yellow-green
            (0.9, 0.9, 0.2),   # Yellow
            (0.95, 0.7, 0.1),  # Orange
            (0.9, 0.4, 0.1),   # Dark orange
            (0.8, 0.2, 0.1),   # Red (high heights)
        ]

    @staticmethod
    def temperature_colormap() -> list[tuple[float, float, float]]:
        """
        Color map for temperature fields.

        Purple (cold) -> Blue -> Cyan -> Green -> Yellow -> Orange -> Red (warm)
        """
        return [
            (0.4, 0.0, 0.6),   # Purple (very cold)
            (0.2, 0.2, 0.8),   # Blue
            (0.0, 0.5, 0.9),   # Cyan
            (0.2, 0.7, 0.4),   # Green
            (0.6, 0.8, 0.2),   # Yellow-green
            (0.9, 0.9, 0.1),   # Yellow
            (0.95, 0.6, 0.1),  # Orange
            (0.9, 0.3, 0.1),   # Red-orange
            (0.7, 0.0, 0.0),   # Dark red (very warm)
        ]

    @staticmethod
    def spread_colormap() -> list[tuple[float, float, float]]:
        """
        Color map for ensemble spread.

        White (low) -> Yellow -> Orange -> Red -> Purple (high)
        """
        return [
            (1.0, 1.0, 1.0),   # White (low spread)
            (1.0, 1.0, 0.8),   # Light yellow
            (1.0, 0.9, 0.5),   # Yellow
            (1.0, 0.7, 0.3),   # Orange
            (0.9, 0.4, 0.2),   # Dark orange
            (0.8, 0.2, 0.2),   # Red
            (0.6, 0.0, 0.4),   # Purple (high spread)
        ]

    @staticmethod
    def confidence_colormap() -> list[tuple[float, float, float]]:
        """
        Color map for confidence/probability fields.

        Red (low) -> Orange -> Yellow -> Green (high)
        """
        return [
            (0.8, 0.0, 0.0),   # Red (low confidence)
            (0.9, 0.4, 0.1),   # Orange
            (0.95, 0.8, 0.2),  # Yellow
            (0.6, 0.9, 0.3),   # Yellow-green
            (0.2, 0.8, 0.2),   # Green (high confidence)
        ]


@dataclass
class ContourLevels:
    """Standard contour levels for meteorological fields."""

    # Z500 levels (decameters)
    z500: NDArray[np.float64] = field(
        default_factory=lambda: np.arange(492, 600, 4)  # 4 dam interval
    )
    z500_fill: NDArray[np.float64] = field(
        default_factory=lambda: np.arange(500, 596, 4)
    )

    # Z850 levels (decameters)
    z850: NDArray[np.float64] = field(
        default_factory=lambda: np.arange(120, 165, 3)
    )

    # T850 levels (Celsius)
    t850_celsius: NDArray[np.float64] = field(
        default_factory=lambda: np.arange(-40, 25, 2)
    )

    # SLP levels (hPa)
    slp: NDArray[np.float64] = field(
        default_factory=lambda: np.arange(960, 1050, 4)
    )
    slp_highlight: NDArray[np.float64] = field(
        default_factory=lambda: np.arange(960, 1050, 8)
    )

    # Spread levels (dam for Z500)
    spread: NDArray[np.float64] = field(
        default_factory=lambda: np.array([1, 2, 4, 6, 8, 10, 15, 20])
    )

    # Confidence levels (%)
    confidence: NDArray[np.float64] = field(
        default_factory=lambda: np.arange(0, 110, 10)
    )


class MapProjection(str, Enum):
    """Supported map projections."""

    LAMBERT_CONFORMAL = "lambert"
    POLAR_STEREOGRAPHIC = "polar"
    MERCATOR = "mercator"
    PLATE_CARREE = "plate_carree"


@dataclass
class PlotDomain:
    """Geographic domain for plotting."""

    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    projection: MapProjection = MapProjection.LAMBERT_CONFORMAL

    # Standard latitude(s) for conic projections
    standard_parallels: tuple[float, float] = (45.0, 55.0)
    central_longitude: float = -10.0

    @classmethod
    def europe(cls) -> "PlotDomain":
        """European domain (Meteociel-style)."""
        return cls(
            name="Europe",
            lat_min=30.0,
            lat_max=70.0,
            lon_min=-30.0,
            lon_max=50.0,
            projection=MapProjection.LAMBERT_CONFORMAL,
            standard_parallels=(40.0, 60.0),
            central_longitude=10.0,
        )

    @classmethod
    def north_atlantic(cls) -> "PlotDomain":
        """North Atlantic domain."""
        return cls(
            name="North Atlantic",
            lat_min=20.0,
            lat_max=70.0,
            lon_min=-80.0,
            lon_max=20.0,
            projection=MapProjection.LAMBERT_CONFORMAL,
            standard_parallels=(35.0, 55.0),
            central_longitude=-30.0,
        )

    @classmethod
    def northern_hemisphere(cls) -> "PlotDomain":
        """Northern Hemisphere polar view."""
        return cls(
            name="Northern Hemisphere",
            lat_min=20.0,
            lat_max=90.0,
            lon_min=-180.0,
            lon_max=180.0,
            projection=MapProjection.POLAR_STEREOGRAPHIC,
            central_longitude=0.0,
        )


@dataclass
class PlotStyle:
    """Complete plot styling configuration."""

    # Figure settings
    figure_width: float = 12.0
    figure_height: float = 10.0
    dpi: int = 150

    # Title settings
    title_fontsize: int = 14
    subtitle_fontsize: int = 11

    # Contour settings
    contour_linewidth: float = 1.0
    contour_highlight_linewidth: float = 1.5
    contour_label_fontsize: int = 9

    # Fill settings
    fill_alpha: float = 0.85
    n_colors: int = 256

    # Coastline and borders
    coastline_linewidth: float = 0.8
    coastline_color: str = "black"
    border_linewidth: float = 0.5
    border_color: str = "gray"

    # Grid lines
    show_gridlines: bool = True
    gridline_linewidth: float = 0.3
    gridline_color: str = "gray"
    gridline_alpha: float = 0.5

    # Colorbar
    colorbar_shrink: float = 0.7
    colorbar_pad: float = 0.02
    colorbar_label_fontsize: int = 10

    # Pressure center markers
    low_marker_color: str = "red"
    high_marker_color: str = "blue"
    center_marker_size: int = 10

    # Domain
    domain: PlotDomain = field(default_factory=PlotDomain.europe)

    @classmethod
    def meteociel_style(cls) -> "PlotStyle":
        """Meteociel-like styling."""
        return cls(
            figure_width=14.0,
            figure_height=10.0,
            dpi=120,
            domain=PlotDomain.europe(),
            contour_linewidth=0.8,
            fill_alpha=0.9,
        )

    @classmethod
    def publication_style(cls) -> "PlotStyle":
        """High-quality publication styling."""
        return cls(
            figure_width=10.0,
            figure_height=8.0,
            dpi=300,
            contour_linewidth=0.6,
            coastline_linewidth=0.5,
            fill_alpha=0.8,
        )

    @classmethod
    def presentation_style(cls) -> "PlotStyle":
        """Large format for presentations."""
        return cls(
            figure_width=16.0,
            figure_height=12.0,
            dpi=100,
            title_fontsize=18,
            contour_linewidth=1.2,
            contour_label_fontsize=11,
        )
