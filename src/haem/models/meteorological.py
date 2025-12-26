"""
Meteorological data models for NWP fields and synoptic structures.

This module defines the core data structures for representing:
- NWP model outputs
- Atmospheric fields (geopotential, temperature, pressure)
- Synoptic patterns and structures
- Forecast run metadata
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator
from numpy.typing import NDArray


class NWPModel(str, Enum):
    """Supported Numerical Weather Prediction models."""

    ECMWF = "ecmwf"  # European Centre for Medium-Range Weather Forecasts
    GFS = "gfs"      # Global Forecast System (NCEP)
    UKMO = "ukmo"    # UK Met Office
    GEM = "gem"      # Global Environmental Multiscale (Canada)
    ICON = "icon"    # Icosahedral Nonhydrostatic (DWD)

    @property
    def full_name(self) -> str:
        """Return the full name of the model."""
        names = {
            "ecmwf": "European Centre for Medium-Range Weather Forecasts",
            "gfs": "Global Forecast System",
            "ukmo": "UK Met Office Unified Model",
            "gem": "Global Environmental Multiscale Model",
            "icon": "Icosahedral Nonhydrostatic Model",
        }
        return names[self.value]

    @property
    def resolution(self) -> str:
        """Return nominal horizontal resolution."""
        resolutions = {
            "ecmwf": "~9 km (HRES)",
            "gfs": "~13 km",
            "ukmo": "~10 km",
            "gem": "~15 km",
            "icon": "~13 km",
        }
        return resolutions[self.value]


class PressureLevel(int, Enum):
    """Standard pressure levels in hPa."""

    SURFACE = 1013
    L1000 = 1000
    L925 = 925
    L850 = 850
    L700 = 700
    L500 = 500
    L300 = 300
    L250 = 250
    L200 = 200


class SynopticPatternType(str, Enum):
    """Types of synoptic-scale atmospheric patterns."""

    TROUGH = "trough"
    RIDGE = "ridge"
    CUT_OFF_LOW = "cut_off_low"
    BLOCKING_HIGH = "blocking_high"
    OMEGA_BLOCK = "omega_block"
    REX_BLOCK = "rex_block"
    ZONAL_FLOW = "zonal_flow"
    MERIDIONAL_FLOW = "meridional_flow"
    SPLIT_FLOW = "split_flow"
    CYCLONIC_VORTEX = "cyclonic_vortex"
    ANTICYCLONIC_VORTEX = "anticyclonic_vortex"
    FRONTAL_ZONE = "frontal_zone"
    JET_STREAK = "jet_streak"


class ForecastRun(BaseModel):
    """Represents a single NWP model forecast run."""

    model: NWPModel
    init_time: datetime = Field(description="Model initialization time (00Z, 06Z, 12Z, 18Z)")
    forecast_hours: list[int] = Field(
        default_factory=lambda: list(range(0, 241, 3)),
        description="Available forecast hours"
    )
    valid_times: list[datetime] = Field(
        default_factory=list,
        description="Valid times corresponding to forecast hours"
    )

    @field_validator("init_time")
    @classmethod
    def validate_synoptic_time(cls, v: datetime) -> datetime:
        """Ensure init time is at synoptic hours (00, 06, 12, 18Z)."""
        if v.hour not in [0, 6, 12, 18]:
            raise ValueError(f"Init time must be at synoptic hours (00, 06, 12, 18Z), got {v.hour}Z")
        return v.replace(minute=0, second=0, microsecond=0)

    def get_lead_time(self, valid_time: datetime) -> int:
        """Calculate forecast lead time in hours."""
        delta = valid_time - self.init_time
        return int(delta.total_seconds() / 3600)


class AtmosphericField(BaseModel):
    """Base class for atmospheric field data."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    units: str
    level: Optional[PressureLevel] = None
    valid_time: datetime
    forecast_hour: int

    # Spatial grid definition
    lats: NDArray[np.float64] = Field(description="Latitude array")
    lons: NDArray[np.float64] = Field(description="Longitude array")
    data: NDArray[np.float64] = Field(description="2D field data [lat, lon]")

    # Metadata
    source_model: NWPModel
    run_time: datetime

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the data array."""
        return self.data.shape

    @property
    def domain(self) -> dict[str, float]:
        """Return the spatial domain bounds."""
        return {
            "lat_min": float(self.lats.min()),
            "lat_max": float(self.lats.max()),
            "lon_min": float(self.lons.min()),
            "lon_max": float(self.lons.max()),
        }

    def interpolate_to_point(self, lat: float, lon: float) -> float:
        """Bilinear interpolation to a specific point."""
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (self.lats, self.lons),
            self.data,
            method="linear",
            bounds_error=False,
            fill_value=np.nan
        )
        return float(interp([[lat, lon]])[0])


class GeopotentialField(AtmosphericField):
    """Geopotential height field (typically Z500 or Z850)."""

    name: str = "geopotential_height"
    units: str = "dam"  # decameters

    @property
    def in_meters(self) -> NDArray[np.float64]:
        """Convert to meters."""
        return self.data * 10.0

    def compute_anomaly(self, climatology: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute anomaly from climatological mean."""
        return self.data - climatology


class TemperatureField(AtmosphericField):
    """Temperature field at a given pressure level."""

    name: str = "temperature"
    units: str = "K"

    @property
    def in_celsius(self) -> NDArray[np.float64]:
        """Convert to degrees Celsius."""
        return self.data - 273.15

    def compute_thermal_advection(
        self,
        u_wind: NDArray[np.float64],
        v_wind: NDArray[np.float64],
        dx: float,
        dy: float
    ) -> NDArray[np.float64]:
        """
        Compute thermal advection: -V · ∇T

        Positive values indicate warm advection, negative indicates cold advection.
        """
        dT_dx = np.gradient(self.data, dx, axis=1)
        dT_dy = np.gradient(self.data, dy, axis=0)
        return -(u_wind * dT_dx + v_wind * dT_dy)


class PressureField(AtmosphericField):
    """Mean Sea Level Pressure (SLP) field."""

    name: str = "mean_sea_level_pressure"
    units: str = "hPa"
    level: Optional[PressureLevel] = None

    def find_pressure_centers(
        self,
        threshold_low: float = 1005.0,
        threshold_high: float = 1025.0
    ) -> dict[str, list[tuple[float, float, float]]]:
        """
        Identify local pressure minima (lows) and maxima (highs).

        Returns dict with 'lows' and 'highs' lists of (lat, lon, pressure).
        """
        from scipy.ndimage import minimum_filter, maximum_filter

        centers: dict[str, list[tuple[float, float, float]]] = {"lows": [], "highs": []}

        # Find local minima (lows)
        min_filtered = minimum_filter(self.data, size=5)
        low_mask = (self.data == min_filtered) & (self.data < threshold_low)
        low_indices = np.where(low_mask)
        for i, j in zip(low_indices[0], low_indices[1]):
            centers["lows"].append((
                float(self.lats[i]),
                float(self.lons[j]),
                float(self.data[i, j])
            ))

        # Find local maxima (highs)
        max_filtered = maximum_filter(self.data, size=5)
        high_mask = (self.data == max_filtered) & (self.data > threshold_high)
        high_indices = np.where(high_mask)
        for i, j in zip(high_indices[0], high_indices[1]):
            centers["highs"].append((
                float(self.lats[i]),
                float(self.lons[j]),
                float(self.data[i, j])
            ))

        return centers


class SynopticPattern(BaseModel):
    """Represents an identified synoptic-scale pattern."""

    pattern_type: SynopticPatternType
    center_lat: float = Field(ge=-90.0, le=90.0)
    center_lon: float = Field(ge=-180.0, le=360.0)
    intensity: float = Field(description="Pattern intensity metric")
    extent_km: float = Field(description="Approximate horizontal extent in km")

    # Physical characteristics
    amplitude: Optional[float] = Field(
        default=None,
        description="Wave amplitude in dam for troughs/ridges"
    )
    wavelength_km: Optional[float] = Field(
        default=None,
        description="Wavelength for wave-like patterns"
    )

    # Temporal characteristics
    valid_time: datetime
    movement_speed_kmh: Optional[float] = Field(
        default=None,
        description="Estimated movement speed"
    )
    movement_direction_deg: Optional[float] = Field(
        default=None,
        description="Movement direction (0-360, meteorological convention)"
    )

    # Associated phenomena
    associated_phenomena: list[str] = Field(default_factory=list)

    @property
    def description(self) -> str:
        """Generate a human-readable description of the pattern."""
        descriptions = {
            SynopticPatternType.TROUGH: "upper-level trough",
            SynopticPatternType.RIDGE: "upper-level ridge",
            SynopticPatternType.CUT_OFF_LOW: "cut-off low pressure system",
            SynopticPatternType.BLOCKING_HIGH: "blocking high pressure system",
            SynopticPatternType.OMEGA_BLOCK: "omega blocking pattern",
            SynopticPatternType.REX_BLOCK: "Rex blocking pattern",
            SynopticPatternType.ZONAL_FLOW: "zonal flow regime",
            SynopticPatternType.MERIDIONAL_FLOW: "meridional flow pattern",
            SynopticPatternType.SPLIT_FLOW: "split flow configuration",
            SynopticPatternType.CYCLONIC_VORTEX: "cyclonic vortex",
            SynopticPatternType.ANTICYCLONIC_VORTEX: "anticyclonic vortex",
            SynopticPatternType.FRONTAL_ZONE: "frontal zone",
            SynopticPatternType.JET_STREAK: "jet streak",
        }
        return descriptions.get(self.pattern_type, str(self.pattern_type))


class SurfaceField(AtmosphericField):
    """Generic surface field (temperature, wind, precipitation, etc.)."""

    name: str = "surface_field"
    units: str = ""
    level: Optional[PressureLevel] = None


class WindField(AtmosphericField):
    """Wind speed field at a given level."""

    name: str = "wind_speed"
    units: str = "m/s"


class PrecipitationField(AtmosphericField):
    """Precipitation field (rain, snow, etc.)."""

    name: str = "precipitation"
    units: str = "mm"
    level: Optional[PressureLevel] = None


class ModelData(BaseModel):
    """Complete data package from a single NWP model run."""

    model_config = {"arbitrary_types_allowed": True}

    run: ForecastRun

    # Core fields (by forecast hour)
    z500: dict[int, GeopotentialField] = Field(
        default_factory=dict,
        description="500 hPa geopotential height by forecast hour"
    )
    z850: dict[int, GeopotentialField] = Field(
        default_factory=dict,
        description="850 hPa geopotential height by forecast hour"
    )
    t850: dict[int, TemperatureField] = Field(
        default_factory=dict,
        description="850 hPa temperature by forecast hour"
    )
    t500: dict[int, TemperatureField] = Field(
        default_factory=dict,
        description="500 hPa temperature by forecast hour"
    )
    slp: dict[int, PressureField] = Field(
        default_factory=dict,
        description="Mean sea level pressure by forecast hour"
    )

    # Additional surface fields
    t2m: dict[int, SurfaceField] = Field(
        default_factory=dict,
        description="2m temperature by forecast hour"
    )
    precip: dict[int, PrecipitationField] = Field(
        default_factory=dict,
        description="Precipitation by forecast hour"
    )
    snow: dict[int, PrecipitationField] = Field(
        default_factory=dict,
        description="Snowfall by forecast hour"
    )
    wind_10m: dict[int, WindField] = Field(
        default_factory=dict,
        description="10m wind speed by forecast hour"
    )
    wind_300: dict[int, WindField] = Field(
        default_factory=dict,
        description="300 hPa wind speed (jet stream) by forecast hour"
    )
    cape: dict[int, SurfaceField] = Field(
        default_factory=dict,
        description="CAPE by forecast hour"
    )

    # Identified patterns
    patterns: dict[int, list[SynopticPattern]] = Field(
        default_factory=dict,
        description="Identified synoptic patterns by forecast hour"
    )

    # Quality metadata
    data_quality_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall data quality score"
    )
    missing_times: list[int] = Field(
        default_factory=list,
        description="Forecast hours with missing data"
    )

    def get_field(
        self,
        field_name: str,
        forecast_hour: int
    ) -> Optional[AtmosphericField]:
        """Retrieve a specific field at a given forecast hour."""
        field_map = {
            "z500": self.z500,
            "z850": self.z850,
            "t850": self.t850,
            "t500": self.t500,
            "slp": self.slp,
            "t2m": self.t2m,
            "precip": self.precip,
            "snow": self.snow,
            "wind_10m": self.wind_10m,
            "wind_300": self.wind_300,
            "cape": self.cape,
        }
        if field_name not in field_map:
            raise ValueError(f"Unknown field: {field_name}")
        return field_map[field_name].get(forecast_hour)

    def available_hours(self, field_name: str = "z500") -> list[int]:
        """Return list of available forecast hours for a field."""
        field_map = {
            "z500": self.z500,
            "z850": self.z850,
            "t850": self.t850,
            "t500": self.t500,
            "slp": self.slp,
            "t2m": self.t2m,
            "precip": self.precip,
            "snow": self.snow,
            "wind_10m": self.wind_10m,
            "wind_300": self.wind_300,
            "cape": self.cape,
        }
        if field_name not in field_map:
            return []
        return sorted(field_map[field_name].keys())
