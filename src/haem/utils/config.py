"""Configuration management for HAEM."""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from haem.models.meteorological import NWPModel
from haem.models.ensemble import EnsembleConfig
from haem.visualization.styles import PlotStyle, PlotDomain


class HAEMConfig(BaseModel):
    """Complete HAEM configuration."""

    # Data settings
    cache_dir: Path = Field(default=Path.home() / ".haem" / "cache")
    output_dir: Path = Field(default=Path("./output"))
    use_synthetic_data: bool = Field(default=True)

    # Models to use
    models: list[NWPModel] = Field(
        default_factory=lambda: [
            NWPModel.ECMWF,
            NWPModel.GFS,
            NWPModel.UKMO,
            NWPModel.GEM,
            NWPModel.ICON,
        ]
    )

    # Forecast settings
    forecast_hours: list[int] = Field(
        default_factory=lambda: list(range(0, 169, 6))
    )
    fields: list[str] = Field(
        default_factory=lambda: ["z500", "z850", "t850", "slp"]
    )

    # Ensemble configuration
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)

    # Plotting
    plot_domain: str = Field(default="europe")
    plot_dpi: int = Field(default=150)

    # Logging
    log_level: str = Field(default="INFO")
    log_file: Optional[Path] = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def get_plot_domain(self) -> PlotDomain:
        """Get the plot domain object."""
        domains = {
            "europe": PlotDomain.europe(),
            "north_atlantic": PlotDomain.north_atlantic(),
            "northern_hemisphere": PlotDomain.northern_hemisphere(),
        }
        return domains.get(self.plot_domain, PlotDomain.europe())

    def to_file(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            # Convert to dict and handle special types
            data = self.model_dump(mode="json")
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def from_file(cls, path: Path) -> "HAEMConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def load_config(path: Optional[Path] = None) -> HAEMConfig:
    """
    Load configuration from file or return defaults.

    Args:
        path: Path to config file (uses ~/.haem/config.json if None)

    Returns:
        HAEMConfig object
    """
    if path is None:
        path = Path.home() / ".haem" / "config.json"

    if path.exists():
        return HAEMConfig.from_file(path)

    return HAEMConfig()
