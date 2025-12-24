"""
Dashboard Configuration for HAEM.

Defines the structure and options for the web dashboard
that displays ensemble analysis results.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from haem.models.meteorological import NWPModel
from haem.data.meteociel_fields import MeteocielField, FieldSelection, FieldPresets
from haem.ai.providers import AIProvider, AIProviderConfig, AIPresets


class MapType(str, Enum):
    """Types of maps available in dashboard."""

    Z500 = "z500"                    # 500 hPa geopotential
    Z850 = "z850"                    # 850 hPa geopotential
    SLP = "slp"                      # Sea level pressure
    T850 = "t850"                    # 850 hPa temperature
    T2M = "t2m"                      # 2m temperature
    PRECIP = "precip"                # Precipitation
    WIND = "wind"                    # Wind
    JET_STREAM = "jet"               # Jet stream
    ENSEMBLE_SPREAD = "spread"       # Ensemble spread
    CONFIDENCE = "confidence"        # Confidence map
    ANOMALY = "anomaly"              # Anomaly map
    THETA_E = "theta_e"              # Theta-E
    CAPE = "cape"                    # CAPE/LI


class ChartType(str, Enum):
    """Types of charts available."""

    TIME_SERIES = "time_series"       # Variable over time
    MODEL_COMPARISON = "model_comp"   # Compare models
    CONFIDENCE_BARS = "conf_bars"     # Confidence by model
    ENSEMBLE_PLUME = "plume"          # Ensemble plume diagram
    SCENARIO_PROB = "scenario"        # Scenario probabilities


class DashboardPanel(BaseModel):
    """Configuration for a single dashboard panel."""

    panel_id: str
    title: str
    panel_type: str = Field(description="'map' or 'chart'")

    # For maps
    map_type: Optional[MapType] = None
    show_contours: bool = True
    show_fill: bool = True
    show_centers: bool = True

    # For charts
    chart_type: Optional[ChartType] = None

    # Position (grid layout)
    row: int = 0
    col: int = 0
    width: int = 1  # Grid units
    height: int = 1


class DashboardLayout(BaseModel):
    """Complete dashboard layout configuration."""

    name: str = "HAEM Dashboard"
    columns: int = Field(default=3, ge=1, le=6)
    panels: list[DashboardPanel] = Field(default_factory=list)

    @classmethod
    def default_synoptic(cls) -> "DashboardLayout":
        """Default layout for synoptic analysis."""
        return cls(
            name="Synoptic Analysis",
            columns=3,
            panels=[
                DashboardPanel(
                    panel_id="z500_map",
                    title="Z500 Geopotential",
                    panel_type="map",
                    map_type=MapType.Z500,
                    row=0, col=0, width=1, height=1,
                ),
                DashboardPanel(
                    panel_id="slp_map",
                    title="Sea Level Pressure",
                    panel_type="map",
                    map_type=MapType.SLP,
                    row=0, col=1, width=1, height=1,
                ),
                DashboardPanel(
                    panel_id="t850_map",
                    title="850 hPa Temperature",
                    panel_type="map",
                    map_type=MapType.T850,
                    row=0, col=2, width=1, height=1,
                ),
                DashboardPanel(
                    panel_id="spread_map",
                    title="Ensemble Spread",
                    panel_type="map",
                    map_type=MapType.ENSEMBLE_SPREAD,
                    row=1, col=0, width=1, height=1,
                ),
                DashboardPanel(
                    panel_id="confidence_map",
                    title="Forecast Confidence",
                    panel_type="map",
                    map_type=MapType.CONFIDENCE,
                    row=1, col=1, width=1, height=1,
                ),
                DashboardPanel(
                    panel_id="model_comparison",
                    title="Model Comparison",
                    panel_type="chart",
                    chart_type=ChartType.MODEL_COMPARISON,
                    row=1, col=2, width=1, height=1,
                ),
            ],
        )

    @classmethod
    def convective_analysis(cls) -> "DashboardLayout":
        """Layout for convective analysis."""
        return cls(
            name="Convective Analysis",
            columns=2,
            panels=[
                DashboardPanel(
                    panel_id="cape_map",
                    title="CAPE / Lifted Index",
                    panel_type="map",
                    map_type=MapType.CAPE,
                    row=0, col=0,
                ),
                DashboardPanel(
                    panel_id="theta_e_map",
                    title="Theta-E 850 hPa",
                    panel_type="map",
                    map_type=MapType.THETA_E,
                    row=0, col=1,
                ),
                DashboardPanel(
                    panel_id="precip_map",
                    title="Precipitation",
                    panel_type="map",
                    map_type=MapType.PRECIP,
                    row=1, col=0,
                ),
                DashboardPanel(
                    panel_id="wind_map",
                    title="Wind",
                    panel_type="map",
                    map_type=MapType.WIND,
                    row=1, col=1,
                ),
            ],
        )


class DashboardConfig(BaseModel):
    """Complete dashboard configuration."""

    # Layout
    layout: DashboardLayout = Field(default_factory=DashboardLayout.default_synoptic)

    # Data selection
    models: list[NWPModel] = Field(
        default_factory=lambda: list(NWPModel),
        description="NWP models to include"
    )
    fields: FieldSelection = Field(
        default_factory=FieldPresets.synoptic_analysis,
        description="Meteociel fields to fetch"
    )

    # AI configuration
    ai_providers: list[AIProviderConfig] = Field(
        default_factory=AIPresets.dual_analysis,
        description="AI providers for analysis"
    )

    # Time settings
    forecast_hours: list[int] = Field(
        default_factory=lambda: [0, 24, 48, 72, 96, 120, 144, 168],
        description="Forecast hours to analyze"
    )
    auto_refresh_minutes: int = Field(
        default=60,
        description="Auto-refresh interval (0 to disable)"
    )

    # Display settings
    default_hour: int = Field(default=48, description="Default forecast hour to show")
    show_ai_reasoning: bool = Field(default=True, description="Show AI analysis text")
    show_model_weights: bool = Field(default=True, description="Show ensemble weights")
    animation_enabled: bool = Field(default=True, description="Enable time animation")

    def get_api_selection_summary(self) -> str:
        """Get summary of selected APIs/providers."""
        lines = [
            "=== CONFIGURAZIONE API ===",
            "",
            "MODELLI NWP:",
        ]
        for model in self.models:
            lines.append(f"  ✓ {model.value.upper()} - {model.full_name}")

        lines.extend([
            "",
            "CAMPI METEOCIEL:",
        ])
        for cat, fields in self.fields.get_fields_by_category().items():
            lines.append(f"  {cat.value.upper()}:")
            for field in fields:
                lines.append(f"    ✓ {field.description}")

        lines.extend([
            "",
            "AI PROVIDERS:",
        ])
        for ai in self.ai_providers:
            status = "✓" if ai.enabled else "✗"
            lines.append(
                f"  {status} {ai.provider.full_name} "
                f"(model: {ai.effective_model}, weight: {ai.base_weight})"
            )

        return "\n".join(lines)


# Quick configuration presets
class DashboardPresets:
    """Preset dashboard configurations."""

    @staticmethod
    def quick_overview() -> DashboardConfig:
        """Quick overview with minimal resources."""
        return DashboardConfig(
            layout=DashboardLayout.default_synoptic(),
            models=[NWPModel.ECMWF, NWPModel.GFS],
            fields=FieldPresets.basic(),
            ai_providers=AIPresets.single_claude(),
            forecast_hours=[0, 24, 48, 72],
        )

    @staticmethod
    def full_analysis() -> DashboardConfig:
        """Comprehensive analysis with all resources."""
        return DashboardConfig(
            layout=DashboardLayout.default_synoptic(),
            models=list(NWPModel),
            fields=FieldPresets.full_analysis(),
            ai_providers=AIPresets.multi_ai_ensemble(),
            forecast_hours=list(range(0, 169, 6)),
        )

    @staticmethod
    def convective_focus() -> DashboardConfig:
        """Focus on convective weather."""
        return DashboardConfig(
            layout=DashboardLayout.convective_analysis(),
            models=[NWPModel.ECMWF, NWPModel.GFS, NWPModel.ICON],
            fields=FieldPresets.convective_analysis(),
            ai_providers=AIPresets.dual_analysis(),
            forecast_hours=[0, 6, 12, 18, 24, 36, 48],
        )
