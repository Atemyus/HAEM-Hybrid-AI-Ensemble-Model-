"""
NWP Data Fetching Module.

This module handles fetching numerical weather prediction data from
public sources, primarily Meteociel-style representations.

Supported models:
- ECMWF (European Centre for Medium-Range Weather Forecasts)
- GFS (Global Forecast System - NCEP)
- UKMO (UK Met Office)
- GEM (Canadian Global Environmental Multiscale)
- ICON (DWD Icosahedral Nonhydrostatic)
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import aiohttp
import numpy as np
from numpy.typing import NDArray

from haem.models.meteorological import (
    AtmosphericField,
    ForecastRun,
    GeopotentialField,
    ModelData,
    NWPModel,
    PressureField,
    PressureLevel,
    TemperatureField,
)

logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Exception raised when data fetching fails."""

    pass


class NWPDataFetcher(ABC):
    """Abstract base class for NWP data fetchers."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".haem" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def fetch_model_data(
        self,
        model: NWPModel,
        run_time: datetime,
        forecast_hours: list[int],
        fields: list[str],
    ) -> ModelData:
        """Fetch data for a specific model run."""
        pass

    @abstractmethod
    async def get_latest_run_time(self, model: NWPModel) -> datetime:
        """Get the latest available run time for a model."""
        pass

    def _get_cache_key(
        self,
        model: NWPModel,
        run_time: datetime,
        field: str,
        forecast_hour: int
    ) -> str:
        """Generate a cache key for a specific data request."""
        key_str = f"{model.value}_{run_time.isoformat()}_{field}_{forecast_hour}"
        return hashlib.md5(key_str.encode()).hexdigest()


class MeteocielFetcher(NWPDataFetcher):
    """
    Fetcher for NWP data from Meteociel-style public representations.

    This fetcher simulates data retrieval from public meteorological
    visualization platforms. In production, this would parse actual
    Meteociel pages or use equivalent public APIs.
    """

    # Base URLs for different models (simulation - actual URLs would be configured)
    MODEL_ENDPOINTS = {
        NWPModel.ECMWF: "https://www.meteociel.fr/modeles/ecmwf",
        NWPModel.GFS: "https://www.meteociel.fr/modeles/gfs",
        NWPModel.UKMO: "https://www.meteociel.fr/modeles/ukmo",
        NWPModel.GEM: "https://www.meteociel.fr/modeles/gem",
        NWPModel.ICON: "https://www.meteociel.fr/modeles/icon",
    }

    # Model run frequency (hours between runs)
    RUN_FREQUENCY = {
        NWPModel.ECMWF: 12,  # 00Z and 12Z
        NWPModel.GFS: 6,    # 00Z, 06Z, 12Z, 18Z
        NWPModel.UKMO: 12,
        NWPModel.GEM: 12,
        NWPModel.ICON: 6,
    }

    # Maximum forecast hours
    MAX_FORECAST_HOURS = {
        NWPModel.ECMWF: 240,
        NWPModel.GFS: 384,
        NWPModel.UKMO: 168,
        NWPModel.GEM: 240,
        NWPModel.ICON: 180,
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        timeout: int = 30,
        use_synthetic_data: bool = True,
    ):
        super().__init__(cache_dir)
        self.timeout = timeout
        self.use_synthetic_data = use_synthetic_data
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_latest_run_time(self, model: NWPModel) -> datetime:
        """
        Determine the latest available run time for a model.

        Models typically become available 4-6 hours after init time.
        """
        now = datetime.utcnow()
        freq = self.RUN_FREQUENCY[model]

        # Find the most recent synoptic time
        hour = (now.hour // freq) * freq
        latest = now.replace(hour=hour, minute=0, second=0, microsecond=0)

        # Subtract one cycle to ensure data is available
        # (models take time to produce and disseminate)
        latest -= timedelta(hours=freq)

        # Ensure it's at a valid synoptic hour
        if model in [NWPModel.ECMWF, NWPModel.UKMO, NWPModel.GEM]:
            # These run at 00Z and 12Z only
            if latest.hour not in [0, 12]:
                latest = latest.replace(hour=12 if latest.hour > 12 else 0)
                if latest.hour == 0:
                    latest -= timedelta(hours=12)

        logger.info(f"Latest run time for {model.value}: {latest}")
        return latest

    async def fetch_model_data(
        self,
        model: NWPModel,
        run_time: datetime,
        forecast_hours: Optional[list[int]] = None,
        fields: Optional[list[str]] = None,
    ) -> ModelData:
        """
        Fetch complete model data for a specific run.

        Args:
            model: The NWP model to fetch
            run_time: Model initialization time
            forecast_hours: List of forecast hours to fetch (default: 0-240 by 3h)
            fields: List of fields to fetch (default: z500, z850, t850, slp)

        Returns:
            ModelData object containing all requested fields
        """
        if forecast_hours is None:
            max_hours = self.MAX_FORECAST_HOURS[model]
            forecast_hours = list(range(0, min(max_hours + 1, 241), 3))

        if fields is None:
            fields = ["z500", "z850", "t850", "slp"]

        logger.info(
            f"Fetching {model.value} data for run {run_time}, "
            f"{len(forecast_hours)} hours, fields: {fields}"
        )

        # Create the forecast run object
        run = ForecastRun(
            model=model,
            init_time=run_time,
            forecast_hours=forecast_hours,
            valid_times=[run_time + timedelta(hours=h) for h in forecast_hours],
        )

        # Initialize model data
        model_data = ModelData(run=run)

        # Fetch each field for each forecast hour
        if self.use_synthetic_data:
            model_data = await self._generate_synthetic_data(model_data, fields)
        else:
            model_data = await self._fetch_real_data(model_data, fields)

        return model_data

    async def _generate_synthetic_data(
        self,
        model_data: ModelData,
        fields: list[str]
    ) -> ModelData:
        """
        Generate synthetic but meteorologically plausible data.

        This is used for testing and demonstration when real data
        is not available. The synthetic data maintains physical
        consistency and realistic patterns.
        """
        run = model_data.run

        # Define grid (Northern Hemisphere, Europe-Atlantic focus)
        lats = np.linspace(20, 70, 101)  # 0.5 degree resolution
        lons = np.linspace(-60, 40, 201)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Model-specific perturbation seed for reproducibility
        model_seeds = {
            NWPModel.ECMWF: 42,
            NWPModel.GFS: 123,
            NWPModel.UKMO: 456,
            NWPModel.GEM: 789,
            NWPModel.ICON: 101112,
        }
        np.random.seed(model_seeds.get(run.model, 0))

        for hour in run.forecast_hours:
            valid_time = run.init_time + timedelta(hours=hour)

            if "z500" in fields:
                z500_data = self._generate_z500_field(
                    lat_grid, lon_grid, valid_time, hour, run.model
                )
                model_data.z500[hour] = GeopotentialField(
                    level=PressureLevel.L500,
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=z500_data,
                    source_model=run.model,
                    run_time=run.init_time,
                )

            if "z850" in fields:
                z850_data = self._generate_z850_field(
                    lat_grid, lon_grid, valid_time, hour, run.model
                )
                model_data.z850[hour] = GeopotentialField(
                    level=PressureLevel.L850,
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=z850_data,
                    source_model=run.model,
                    run_time=run.init_time,
                )

            if "t850" in fields:
                t850_data = self._generate_t850_field(
                    lat_grid, lon_grid, valid_time, hour, run.model
                )
                model_data.t850[hour] = TemperatureField(
                    level=PressureLevel.L850,
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=t850_data,
                    source_model=run.model,
                    run_time=run.init_time,
                )

            if "slp" in fields:
                slp_data = self._generate_slp_field(
                    lat_grid, lon_grid, valid_time, hour, run.model
                )
                model_data.slp[hour] = PressureField(
                    valid_time=valid_time,
                    forecast_hour=hour,
                    lats=lats,
                    lons=lons,
                    data=slp_data,
                    source_model=run.model,
                    run_time=run.init_time,
                )

        return model_data

    def _generate_z500_field(
        self,
        lat_grid: NDArray[np.float64],
        lon_grid: NDArray[np.float64],
        valid_time: datetime,
        forecast_hour: int,
        model: NWPModel,
    ) -> NDArray[np.float64]:
        """
        Generate synthetic Z500 field with realistic Rossby wave patterns.

        The field includes:
        - Meridional temperature gradient effect
        - Rossby wave-like undulations
        - Model-specific perturbations
        - Temporal evolution
        """
        # Base height decreases with latitude (thermal wind)
        base_height = 560 - (lat_grid - 45) * 0.8

        # Add Rossby wave pattern (wavenumber 4-6 typical for mid-latitudes)
        time_phase = valid_time.hour / 24.0 + forecast_hour / 240.0
        wave_phase = np.deg2rad(lon_grid * 4 + time_phase * 90)  # Eastward propagation
        wave_amplitude = 8 + 2 * np.sin(np.deg2rad(lat_grid - 45) * 2)
        rossby_wave = wave_amplitude * np.sin(wave_phase)

        # Add cut-off low / blocking features based on time
        day_of_year = valid_time.timetuple().tm_yday
        block_lat, block_lon = 55, -10 + (forecast_hour / 24) * 5
        block_strength = 5 * np.sin(np.deg2rad(day_of_year))
        dist_to_block = np.sqrt((lat_grid - block_lat)**2 + (lon_grid - block_lon)**2)
        blocking_signal = block_strength * np.exp(-dist_to_block**2 / 200)

        # Model-specific perturbation
        model_perturbation = self._get_model_perturbation(lat_grid, lon_grid, model, 0.5)

        z500 = base_height + rossby_wave + blocking_signal + model_perturbation

        # Add realistic noise
        noise = np.random.normal(0, 0.5, z500.shape)
        z500 += noise

        return z500

    def _generate_z850_field(
        self,
        lat_grid: NDArray[np.float64],
        lon_grid: NDArray[np.float64],
        valid_time: datetime,
        forecast_hour: int,
        model: NWPModel,
    ) -> NDArray[np.float64]:
        """Generate synthetic Z850 field."""
        # 850 hPa is lower, follows similar patterns but with less amplitude
        base_height = 145 - (lat_grid - 45) * 0.4

        time_phase = valid_time.hour / 24.0 + forecast_hour / 240.0
        wave_phase = np.deg2rad(lon_grid * 4 + time_phase * 90)
        wave_amplitude = 4 + 1 * np.sin(np.deg2rad(lat_grid - 45) * 2)
        rossby_wave = wave_amplitude * np.sin(wave_phase)

        model_perturbation = self._get_model_perturbation(lat_grid, lon_grid, model, 0.3)

        z850 = base_height + rossby_wave + model_perturbation
        noise = np.random.normal(0, 0.3, z850.shape)

        return z850 + noise

    def _generate_t850_field(
        self,
        lat_grid: NDArray[np.float64],
        lon_grid: NDArray[np.float64],
        valid_time: datetime,
        forecast_hour: int,
        model: NWPModel,
    ) -> NDArray[np.float64]:
        """Generate synthetic T850 field in Kelvin."""
        # Base temperature decreases with latitude
        month = valid_time.month
        seasonal_offset = 10 * np.cos(np.deg2rad((month - 7) * 30))  # Warmest in July

        base_temp = 283 + seasonal_offset - (lat_grid - 45) * 0.6

        # Add thermal advection patterns
        time_phase = valid_time.hour / 24.0 + forecast_hour / 240.0
        wave_phase = np.deg2rad(lon_grid * 4 + time_phase * 90 - 45)  # Phase lag from Z500
        thermal_wave = 3 * np.sin(wave_phase)

        # Land-sea contrast (simplified)
        is_land = (lon_grid > -10) & (lon_grid < 30) & (lat_grid > 35) & (lat_grid < 60)
        land_effect = np.where(is_land, 2 * np.sign(seasonal_offset), 0)

        model_perturbation = self._get_model_perturbation(lat_grid, lon_grid, model, 1.0)

        t850 = base_temp + thermal_wave + land_effect + model_perturbation
        noise = np.random.normal(0, 0.5, t850.shape)

        return t850 + noise

    def _generate_slp_field(
        self,
        lat_grid: NDArray[np.float64],
        lon_grid: NDArray[np.float64],
        valid_time: datetime,
        forecast_hour: int,
        model: NWPModel,
    ) -> NDArray[np.float64]:
        """Generate synthetic SLP field in hPa."""
        # Base pressure with slight latitude dependence
        base_pressure = 1013 + (lat_grid - 45) * 0.1

        # Add synoptic-scale pressure systems
        time_phase = valid_time.hour / 24.0 + forecast_hour / 240.0

        # Icelandic Low
        iceland_lat, iceland_lon = 65, -20 + time_phase * 20
        dist_iceland = np.sqrt((lat_grid - iceland_lat)**2 + (lon_grid - iceland_lon)**2)
        iceland_low = -15 * np.exp(-dist_iceland**2 / 300)

        # Azores High
        azores_lat, azores_lon = 35, -30 + time_phase * 10
        dist_azores = np.sqrt((lat_grid - azores_lat)**2 + (lon_grid - azores_lon)**2)
        azores_high = 12 * np.exp(-dist_azores**2 / 400)

        # Mobile cyclone
        cyclone_lat = 50 + 5 * np.sin(time_phase * np.pi)
        cyclone_lon = -40 + forecast_hour * 0.5
        dist_cyclone = np.sqrt((lat_grid - cyclone_lat)**2 + (lon_grid - cyclone_lon)**2)
        mobile_cyclone = -10 * np.exp(-dist_cyclone**2 / 100)

        model_perturbation = self._get_model_perturbation(lat_grid, lon_grid, model, 2.0)

        slp = base_pressure + iceland_low + azores_high + mobile_cyclone + model_perturbation
        noise = np.random.normal(0, 1.0, slp.shape)

        return slp + noise

    def _get_model_perturbation(
        self,
        lat_grid: NDArray[np.float64],
        lon_grid: NDArray[np.float64],
        model: NWPModel,
        scale: float,
    ) -> NDArray[np.float64]:
        """
        Generate model-specific perturbations to simulate inter-model spread.

        Different models have different biases and characteristics.
        """
        # Model-specific bias patterns
        if model == NWPModel.ECMWF:
            # ECMWF tends to be most accurate, small perturbations
            perturbation = 0.3 * scale * np.sin(np.deg2rad(lat_grid * 3 + lon_grid * 2))
        elif model == NWPModel.GFS:
            # GFS can have phase speed errors
            perturbation = 0.5 * scale * np.sin(np.deg2rad(lat_grid * 2 + lon_grid * 3 + 30))
        elif model == NWPModel.UKMO:
            # UKMO good for European sector
            perturbation = 0.4 * scale * np.sin(np.deg2rad(lat_grid * 2.5 + lon_grid * 2.5 + 60))
        elif model == NWPModel.GEM:
            # GEM can differ in blocking scenarios
            perturbation = 0.6 * scale * np.sin(np.deg2rad(lat_grid * 2 + lon_grid * 4 + 90))
        elif model == NWPModel.ICON:
            # ICON different convective representation
            perturbation = 0.45 * scale * np.sin(np.deg2rad(lat_grid * 3 + lon_grid * 3 + 45))
        else:
            perturbation = np.zeros_like(lat_grid)

        return perturbation

    async def _fetch_real_data(
        self,
        model_data: ModelData,
        fields: list[str]
    ) -> ModelData:
        """
        Fetch real data from external sources.

        This method would implement actual HTTP requests to fetch
        data from Meteociel or equivalent sources.
        """
        # In production, this would parse Meteociel pages or use APIs
        # For now, fall back to synthetic data
        logger.warning("Real data fetching not implemented, using synthetic data")
        return await self._generate_synthetic_data(model_data, fields)


class MultiModelFetcher:
    """Orchestrates fetching data from multiple NWP models."""

    def __init__(
        self,
        models: Optional[list[NWPModel]] = None,
        cache_dir: Optional[Path] = None,
        use_synthetic_data: bool = True,
    ):
        self.models = models or list(NWPModel)
        self.fetcher = MeteocielFetcher(
            cache_dir=cache_dir,
            use_synthetic_data=use_synthetic_data
        )

    async def fetch_all_models(
        self,
        run_time: Optional[datetime] = None,
        forecast_hours: Optional[list[int]] = None,
        fields: Optional[list[str]] = None,
    ) -> dict[NWPModel, ModelData]:
        """
        Fetch data from all configured models.

        Args:
            run_time: Model run time (uses latest if None)
            forecast_hours: Hours to fetch
            fields: Fields to fetch

        Returns:
            Dict mapping model to ModelData
        """
        results: dict[NWPModel, ModelData] = {}

        # Create fetch tasks for all models
        async def fetch_one(model: NWPModel) -> tuple[NWPModel, ModelData]:
            model_run_time = run_time
            if model_run_time is None:
                model_run_time = await self.fetcher.get_latest_run_time(model)

            data = await self.fetcher.fetch_model_data(
                model=model,
                run_time=model_run_time,
                forecast_hours=forecast_hours,
                fields=fields,
            )
            return model, data

        # Fetch all models concurrently
        tasks = [fetch_one(model) for model in self.models]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for result in completed:
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch model data: {result}")
            else:
                model, data = result
                results[model] = data

        return results

    async def close(self) -> None:
        """Close the fetcher session."""
        await self.fetcher.close()
