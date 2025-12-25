"""Data fetching and processing modules for HAEM."""

from haem.data.fetcher import NWPDataFetcher, MeteocielFetcher
from haem.data.processor import DataProcessor, GridInterpolator
from haem.data.cache import DataCache
from haem.data.openmeteo import (
    OpenMeteoFetcher,
    MultiModelOpenMeteoFetcher,
    OpenMeteoConfig,
    fetch_current_data,
)
from haem.data.meteociel_fields import (
    MeteocielField,
    FieldSelection,
    FieldPresets,
    FieldCategory,
)

__all__ = [
    "NWPDataFetcher",
    "MeteocielFetcher",
    "DataProcessor",
    "GridInterpolator",
    "DataCache",
    "OpenMeteoFetcher",
    "MultiModelOpenMeteoFetcher",
    "OpenMeteoConfig",
    "fetch_current_data",
    "MeteocielField",
    "FieldSelection",
    "FieldPresets",
    "FieldCategory",
]
