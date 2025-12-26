"""
Open-Meteo API Integration for HAEM.

Fetches real meteorological data from Open-Meteo's free API.
Supports multiple NWP models: ECMWF, GFS, ICON, GEM, ARPEGE, JMA.

Open-Meteo provides pre-processed data in JSON format, eliminating
the need for GRIB2 processing.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

import aiohttp
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from haem.models.meteorological import (
    NWPModel,
    ForecastRun,
    GeopotentialField,
    TemperatureField,
    PressureField,
    PressureLevel,
    ModelData,
    SurfaceField,
    WindField,
    PrecipitationField,
)

logger = logging.getLogger(__name__)


class OpenMeteoModel(str, Enum):
    """Open-Meteo model identifiers."""

    ECMWF_IFS = "ecmwf_ifs025"         # ECMWF IFS 0.25° (corretto)
    GFS_SEAMLESS = "gfs_seamless"      # GFS with HRRR blend
    GFS_GLOBAL = "gfs_global"          # Pure GFS 0.25°
    ICON_GLOBAL = "icon_global"        # DWD ICON 0.1°
    GEM_GLOBAL = "gem_global"          # Canadian GEM 0.15°
    ARPEGE_WORLD = "arpege_world"      # Météo-France 0.25°
    JMA_SEAMLESS = "jma_seamless"      # Japan Meteorological Agency

    @classmethod
    def from_nwp_model(cls, model: NWPModel) -> "OpenMeteoModel":
        """Convert NWPModel to OpenMeteoModel."""
        mapping = {
            NWPModel.ECMWF: cls.ECMWF_IFS,
            NWPModel.GFS: cls.GFS_GLOBAL,
            NWPModel.ICON: cls.ICON_GLOBAL,
            NWPModel.GEM: cls.GEM_GLOBAL,
            NWPModel.UKMO: cls.ARPEGE_WORLD,  # ARPEGE as substitute
        }
        return mapping.get(model, cls.GFS_GLOBAL)


class OpenMeteoVariable(str, Enum):
    """Available variables from Open-Meteo."""

    # Surface variables
    TEMPERATURE_2M = "temperature_2m"
    RELATIVE_HUMIDITY_2M = "relative_humidity_2m"
    PRECIPITATION = "precipitation"
    RAIN = "rain"
    SNOWFALL = "snowfall"
    SNOW_DEPTH = "snow_depth"
    PRESSURE_MSL = "pressure_msl"
    SURFACE_PRESSURE = "surface_pressure"
    WIND_SPEED_10M = "wind_speed_10m"
    WIND_DIRECTION_10M = "wind_direction_10m"
    WIND_GUSTS_10M = "wind_gusts_10m"
    CAPE = "cape"
    LIFTED_INDEX = "lifted_index"

    # Pressure level variables (need level suffix)
    TEMPERATURE = "temperature"
    GEOPOTENTIAL_HEIGHT = "geopotential_height"
    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    RELATIVE_HUMIDITY = "relative_humidity"


class OpenMeteoConfig(BaseModel):
    """Configuration for Open-Meteo API requests."""

    # API endpoints
    base_url: str = "https://api.open-meteo.com/v1/forecast"
    ensemble_url: str = "https://ensemble-api.open-meteo.com/v1/ensemble"

    # Geographic domain (Europe-Atlantic focus)
    latitude_min: float = 30.0
    latitude_max: float = 70.0
    longitude_min: float = -30.0
    longitude_max: float = 40.0

    # Resolution
    grid_resolution: float = 0.5  # degrees

    # Time settings
    forecast_days: int = 7
    past_days: int = 0

    # Timeout
    timeout_seconds: int = 60


class OpenMeteoFetcher:
    """
    Fetches real meteorological data from Open-Meteo API.

    Retrieves data for multiple models and converts to HAEM format.
    """

    PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300, 250, 200]

    def __init__(self, config: Optional[OpenMeteoConfig] = None):
        self.config = config or OpenMeteoConfig()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _build_grid(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build latitude/longitude grid arrays."""
        lats = np.arange(
            self.config.latitude_min,
            self.config.latitude_max + self.config.grid_resolution,
            self.config.grid_resolution
        )
        lons = np.arange(
            self.config.longitude_min,
            self.config.longitude_max + self.config.grid_resolution,
            self.config.grid_resolution
        )
        return lats, lons

    async def fetch_model_data(
        self,
        model: NWPModel,
        forecast_hours: Optional[list[int]] = None,
    ) -> ModelData:
        """
        Fetch data for a specific NWP model.

        Args:
            model: NWP model to fetch
            forecast_hours: Specific hours to fetch (default: 0-168 by 3h)

        Returns:
            ModelData with all fields populated
        """
        if forecast_hours is None:
            forecast_hours = list(range(0, 169, 3))

        om_model = OpenMeteoModel.from_nwp_model(model)
        logger.info(f"Fetching {model.value} data via Open-Meteo ({om_model.value})")

        lats, lons = self._build_grid()

        # For grid data, we need to make multiple point requests or use a different approach
        # Open-Meteo works best with point forecasts, so we'll sample key points
        # and interpolate for the full grid

        # Sample points across the domain
        sample_lats = np.linspace(self.config.latitude_min, self.config.latitude_max, 15)
        sample_lons = np.linspace(self.config.longitude_min, self.config.longitude_max, 25)

        # Fetch data for all sample points
        all_data = await self._fetch_grid_data(om_model, sample_lats, sample_lons)

        # Create model run
        now = datetime.utcnow()
        run_hour = (now.hour // 6) * 6  # Round to nearest 6h
        run_time = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)

        run = ForecastRun(
            model=model,
            init_time=run_time,
            forecast_hours=forecast_hours,
        )

        model_data = ModelData(run=run)

        # Process and interpolate data for each forecast hour
        for hour_idx, hour in enumerate(forecast_hours):
            if hour_idx >= len(all_data.get('time', [])):
                continue

            valid_time = run_time + timedelta(hours=hour)

            # Interpolate sampled data to full grid - Core synoptic fields
            z500_grid = self._interpolate_to_grid(
                all_data, 'geopotential_height_500hPa', hour_idx,
                sample_lats, sample_lons, lats, lons
            )

            z850_grid = self._interpolate_to_grid(
                all_data, 'geopotential_height_850hPa', hour_idx,
                sample_lats, sample_lons, lats, lons
            )

            t850_grid = self._interpolate_to_grid(
                all_data, 'temperature_850hPa', hour_idx,
                sample_lats, sample_lons, lats, lons
            )

            t500_grid = self._interpolate_to_grid(
                all_data, 'temperature_500hPa', hour_idx,
                sample_lats, sample_lons, lats, lons
            )

            slp_grid = self._interpolate_to_grid(
                all_data, 'pressure_msl', hour_idx,
                sample_lats, sample_lons, lats, lons
            )

            # Additional surface fields
            t2m_grid = self._interpolate_to_grid(
                all_data, 'temperature_2m', hour_idx,
                sample_lats, sample_lons, lats, lons
            )

            precip_grid = self._interpolate_to_grid(
                all_data, 'precipitation', hour_idx,
                sample_lats, sample_lons, lats, lons
            )

            snow_grid = self._interpolate_to_grid(
                all_data, 'snowfall', hour_idx,
                sample_lats, sample_lons, lats, lons
            )

            wind_10m_grid = self._interpolate_to_grid(
                all_data, 'wind_speed_10m', hour_idx,
                sample_lats, sample_lons, lats, lons
            )

            wind_300_grid = self._interpolate_to_grid(
                all_data, 'wind_speed_300hPa', hour_idx,
                sample_lats, sample_lons, lats, lons
            )

            cape_grid = self._interpolate_to_grid(
                all_data, 'cape', hour_idx,
                sample_lats, sample_lons, lats, lons
            )

            # Store core synoptic fields
            if z500_grid is not None:
                model_data.z500[hour] = GeopotentialField(
                    level=PressureLevel.L500,
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=z500_grid / 10.0,  # Convert m to dam
                    source_model=model,
                    run_time=run_time,
                )

            if z850_grid is not None:
                model_data.z850[hour] = GeopotentialField(
                    level=PressureLevel.L850,
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=z850_grid / 10.0,
                    source_model=model,
                    run_time=run_time,
                )

            if t850_grid is not None:
                model_data.t850[hour] = TemperatureField(
                    level=PressureLevel.L850,
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=t850_grid + 273.15,  # Convert C to K
                    source_model=model,
                    run_time=run_time,
                )

            if t500_grid is not None:
                model_data.t500[hour] = TemperatureField(
                    level=PressureLevel.L500,
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=t500_grid + 273.15,  # Convert C to K
                    source_model=model,
                    run_time=run_time,
                )

            if slp_grid is not None:
                model_data.slp[hour] = PressureField(
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=slp_grid,
                    source_model=model,
                    run_time=run_time,
                )

            # Store additional surface fields
            if t2m_grid is not None:
                model_data.t2m[hour] = SurfaceField(
                    name="temperature_2m",
                    units="C",
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=t2m_grid,  # Already in Celsius
                    source_model=model,
                    run_time=run_time,
                )

            if precip_grid is not None:
                model_data.precip[hour] = PrecipitationField(
                    name="precipitation",
                    units="mm",
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=precip_grid,
                    source_model=model,
                    run_time=run_time,
                )

            if snow_grid is not None:
                model_data.snow[hour] = PrecipitationField(
                    name="snowfall",
                    units="cm",
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=snow_grid,
                    source_model=model,
                    run_time=run_time,
                )

            if wind_10m_grid is not None:
                model_data.wind_10m[hour] = WindField(
                    name="wind_speed_10m",
                    units="m/s",
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=wind_10m_grid,
                    source_model=model,
                    run_time=run_time,
                )

            if wind_300_grid is not None:
                model_data.wind_300[hour] = WindField(
                    name="wind_speed_300hPa",
                    units="kt",
                    level=PressureLevel.L300,
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=wind_300_grid * 1.94384,  # Convert m/s to knots
                    source_model=model,
                    run_time=run_time,
                )

            if cape_grid is not None:
                model_data.cape[hour] = SurfaceField(
                    name="cape",
                    units="J/kg",
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=cape_grid,
                    source_model=model,
                    run_time=run_time,
                )

        logger.info(f"Fetched {len(model_data.z500)} timesteps for {model.value}")
        return model_data

    async def _fetch_grid_data(
        self,
        model: OpenMeteoModel,
        lats: NDArray[np.float64],
        lons: NDArray[np.float64],
    ) -> dict:
        """Fetch data for a grid of points."""

        session = await self._get_session()

        # Variables to fetch - Surface
        hourly_vars = [
            "temperature_2m",
            "pressure_msl",
            "precipitation",
            "snowfall",
            "snow_depth",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "cape",
            "lifted_index",
            "freezing_level_height",
        ]

        # Pressure level variables
        pressure_level_vars = []
        for level in [850, 500, 300]:
            pressure_level_vars.extend([
                f"temperature_{level}hPa",
                f"geopotential_height_{level}hPa",
                f"wind_speed_{level}hPa",
                f"wind_direction_{level}hPa",
            ])

        all_vars = hourly_vars + pressure_level_vars

        # Collect data for all grid points
        all_point_data: dict = {var: [] for var in all_vars}
        all_point_data['time'] = None
        all_point_data['lats'] = []
        all_point_data['lons'] = []

        # Create tasks for parallel fetching
        tasks = []
        for lat in lats:
            for lon in lons:
                task = self._fetch_point(session, model, lat, lon, all_vars)
                tasks.append((lat, lon, task))

        # Execute with rate limiting
        for lat, lon, task in tasks:
            try:
                data = await task
                if data and 'hourly' in data:
                    all_point_data['lats'].append(lat)
                    all_point_data['lons'].append(lon)

                    if all_point_data['time'] is None:
                        all_point_data['time'] = data['hourly'].get('time', [])

                    for var in all_vars:
                        values = data['hourly'].get(var, [])
                        all_point_data[var].append(values)

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.05)

            except Exception as e:
                logger.warning(f"Failed to fetch point ({lat}, {lon}): {e}")

        return all_point_data

    async def _fetch_point(
        self,
        session: aiohttp.ClientSession,
        model: OpenMeteoModel,
        lat: float,
        lon: float,
        variables: list[str],
    ) -> Optional[dict]:
        """Fetch data for a single point."""

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(variables),
            "models": model.value,
            "forecast_days": self.config.forecast_days,
        }

        try:
            async with session.get(self.config.base_url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"API returned {response.status} for ({lat}, {lon})")
                    return None
        except Exception as e:
            logger.error(f"Request failed for ({lat}, {lon}): {e}")
            return None

    def _interpolate_to_grid(
        self,
        data: dict,
        variable: str,
        time_idx: int,
        sample_lats: NDArray[np.float64],
        sample_lons: NDArray[np.float64],
        target_lats: NDArray[np.float64],
        target_lons: NDArray[np.float64],
    ) -> Optional[NDArray[np.float64]]:
        """Interpolate sampled data to full grid."""

        if variable not in data or not data[variable]:
            return None

        from scipy.interpolate import griddata

        # Extract values at time_idx for all points
        points = []
        values = []

        for i, (lat, lon) in enumerate(zip(data['lats'], data['lons'])):
            if i < len(data[variable]) and time_idx < len(data[variable][i]):
                val = data[variable][i][time_idx]
                if val is not None:
                    points.append([lon, lat])
                    values.append(val)

        if len(points) < 4:
            return None

        points = np.array(points)
        values = np.array(values)

        # Create target grid
        target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)

        # Interpolate
        try:
            grid = griddata(
                points, values,
                (target_lon_grid, target_lat_grid),
                method='linear',
                fill_value=np.nan
            )

            # Fill NaN with nearest neighbor
            if np.any(np.isnan(grid)):
                grid_nearest = griddata(
                    points, values,
                    (target_lon_grid, target_lat_grid),
                    method='nearest'
                )
                grid = np.where(np.isnan(grid), grid_nearest, grid)

            return grid

        except Exception as e:
            logger.error(f"Interpolation failed for {variable}: {e}")
            return None


class MultiModelOpenMeteoFetcher:
    """Fetches data from multiple models via Open-Meteo."""

    def __init__(
        self,
        models: Optional[list[NWPModel]] = None,
        config: Optional[OpenMeteoConfig] = None,
    ):
        self.models = models or [
            NWPModel.ECMWF,
            NWPModel.GFS,
            NWPModel.ICON,
            NWPModel.GEM,
            NWPModel.UKMO,  # Will use ARPEGE
        ]
        self.fetcher = OpenMeteoFetcher(config)

    async def fetch_all_models(
        self,
        forecast_hours: Optional[list[int]] = None,
    ) -> dict[NWPModel, ModelData]:
        """
        Fetch data from all configured models.

        Args:
            forecast_hours: Hours to fetch

        Returns:
            Dict mapping model to ModelData
        """
        results: dict[NWPModel, ModelData] = {}

        for model in self.models:
            try:
                logger.info(f"Fetching {model.value}...")
                data = await self.fetcher.fetch_model_data(model, forecast_hours)
                results[model] = data
            except Exception as e:
                logger.error(f"Failed to fetch {model.value}: {e}")

        await self.fetcher.close()
        return results


# Simplified fetcher for quick testing
async def fetch_current_data(
    models: list[NWPModel],
    hours: list[int] = [0, 24, 48, 72],
) -> dict[NWPModel, ModelData]:
    """Quick function to fetch current data."""
    fetcher = MultiModelOpenMeteoFetcher(models=models)
    return await fetcher.fetch_all_models(forecast_hours=hours)
