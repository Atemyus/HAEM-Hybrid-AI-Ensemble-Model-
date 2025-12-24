"""
Meteorological Map Plotter.

This module provides the core plotting functionality for generating
professional meteorological visualizations using matplotlib and cartopy.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from haem.models.meteorological import (
    GeopotentialField,
    ModelData,
    NWPModel,
    PressureField,
    TemperatureField,
)
from haem.models.ensemble import EnsembleField, EnsembleResult
from haem.visualization.styles import (
    ContourLevels,
    MeteoColorMaps,
    PlotDomain,
    PlotStyle,
)

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
plt = None
ccrs = None
cfeature = None


def _ensure_plotting_imports():
    """Lazily import plotting libraries."""
    global plt, ccrs, cfeature
    if plt is None:
        import matplotlib.pyplot as plt_module
        import matplotlib.colors as mcolors

        plt = plt_module

        try:
            import cartopy.crs as ccrs_module
            import cartopy.feature as cfeature_module

            ccrs = ccrs_module
            cfeature = cfeature_module
        except ImportError:
            logger.warning("Cartopy not available, using basic plotting")
            ccrs = None
            cfeature = None


class MeteoPlotter:
    """
    Professional meteorological map generator.

    Creates publication-quality maps with:
    - Color-filled contours
    - Isoline overlays
    - Geographic features
    - Annotations and legends
    """

    def __init__(
        self,
        style: Optional[PlotStyle] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the plotter.

        Args:
            style: Plot styling configuration
            output_dir: Directory for saving plots
        """
        _ensure_plotting_imports()

        self.style = style or PlotStyle.meteociel_style()
        self.output_dir = output_dir or Path("./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.contour_levels = ContourLevels()

    def plot_z500(
        self,
        field: Union[GeopotentialField, EnsembleField],
        title: Optional[str] = None,
        show_spread: bool = False,
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Plot 500 hPa geopotential height.

        Args:
            field: Z500 field to plot
            title: Custom title (auto-generated if None)
            show_spread: Show ensemble spread if available
            save_path: Path to save figure

        Returns:
            Path to saved figure if save_path provided
        """
        if isinstance(field, EnsembleField):
            data = field.ensemble_mean
            lats = field.lats
            lons = field.lons
            spread = field.ensemble_spread if show_spread else None
            source = "HAEM Ensemble"
        else:
            data = field.data
            lats = field.lats
            lons = field.lons
            spread = None
            source = field.source_model.value.upper()

        # Create figure and projection
        fig, ax = self._create_figure_with_map()

        # Create meshgrid for plotting
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Color-filled contours
        cmap = self._create_colormap(MeteoColorMaps.z500_colormap())
        levels = self.contour_levels.z500_fill

        if ccrs is not None:
            transform = ccrs.PlateCarree()
            cf = ax.contourf(
                lon_grid, lat_grid, data,
                levels=levels,
                cmap=cmap,
                alpha=self.style.fill_alpha,
                transform=transform,
                extend="both",
            )

            # Contour lines
            cs = ax.contour(
                lon_grid, lat_grid, data,
                levels=self.contour_levels.z500,
                colors="black",
                linewidths=self.style.contour_linewidth,
                transform=transform,
            )
            ax.clabel(cs, inline=True, fontsize=self.style.contour_label_fontsize, fmt="%.0f")

            # Spread overlay if available
            if spread is not None:
                spread_levels = self.contour_levels.spread
                ax.contour(
                    lon_grid, lat_grid, spread,
                    levels=spread_levels,
                    colors="magenta",
                    linewidths=0.5,
                    linestyles="dashed",
                    transform=transform,
                )
        else:
            cf = ax.contourf(
                lon_grid, lat_grid, data,
                levels=levels,
                cmap=cmap,
                alpha=self.style.fill_alpha,
            )
            cs = ax.contour(
                lon_grid, lat_grid, data,
                levels=self.contour_levels.z500,
                colors="black",
                linewidths=self.style.contour_linewidth,
            )
            ax.clabel(cs, inline=True, fontsize=self.style.contour_label_fontsize, fmt="%.0f")

        # Colorbar
        cbar = plt.colorbar(
            cf, ax=ax,
            shrink=self.style.colorbar_shrink,
            pad=self.style.colorbar_pad,
        )
        cbar.set_label("Z500 (dam)", fontsize=self.style.colorbar_label_fontsize)

        # Title
        if title is None:
            if isinstance(field, EnsembleField):
                title = f"Z500 Geopotential Height - {source}"
                subtitle = (
                    f"Valid: {field.valid_time.strftime('%Y-%m-%d %H:%MZ')} "
                    f"(+{field.forecast_hour}h)"
                )
            else:
                title = f"Z500 Geopotential Height - {source}"
                subtitle = (
                    f"Valid: {field.valid_time.strftime('%Y-%m-%d %H:%MZ')} "
                    f"(+{field.forecast_hour}h)"
                )
            ax.set_title(title, fontsize=self.style.title_fontsize, fontweight="bold")
            ax.set_title(subtitle, fontsize=self.style.subtitle_fontsize, loc="right")

        else:
            ax.set_title(title, fontsize=self.style.title_fontsize, fontweight="bold")

        plt.tight_layout()

        # Save
        if save_path:
            plt.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
            plt.close(fig)
            return save_path
        else:
            return None

    def plot_slp(
        self,
        field: Union[PressureField, EnsembleField],
        mark_centers: bool = True,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Plot Mean Sea Level Pressure.

        Args:
            field: SLP field to plot
            mark_centers: Mark H/L pressure centers
            title: Custom title
            save_path: Path to save figure

        Returns:
            Path to saved figure if save_path provided
        """
        if isinstance(field, EnsembleField):
            data = field.ensemble_mean
            lats = field.lats
            lons = field.lons
            source = "HAEM Ensemble"
        else:
            data = field.data
            lats = field.lats
            lons = field.lons
            source = field.source_model.value.upper()

        # Create figure
        fig, ax = self._create_figure_with_map()
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Contour lines (no fill for SLP typically)
        if ccrs is not None:
            transform = ccrs.PlateCarree()
            cs = ax.contour(
                lon_grid, lat_grid, data,
                levels=self.contour_levels.slp,
                colors="black",
                linewidths=self.style.contour_linewidth,
                transform=transform,
            )
            # Highlight major isobars
            cs_major = ax.contour(
                lon_grid, lat_grid, data,
                levels=self.contour_levels.slp_highlight,
                colors="black",
                linewidths=self.style.contour_highlight_linewidth,
                transform=transform,
            )
        else:
            cs = ax.contour(
                lon_grid, lat_grid, data,
                levels=self.contour_levels.slp,
                colors="black",
                linewidths=self.style.contour_linewidth,
            )

        ax.clabel(cs, inline=True, fontsize=self.style.contour_label_fontsize, fmt="%.0f")

        # Mark pressure centers
        if mark_centers and isinstance(field, PressureField):
            centers = field.find_pressure_centers()
            for lat, lon, pressure in centers["lows"]:
                ax.plot(
                    lon, lat, "r*",
                    markersize=self.style.center_marker_size,
                    transform=ccrs.PlateCarree() if ccrs else None,
                )
                ax.text(
                    lon, lat - 1, "L",
                    color="red", fontsize=12, fontweight="bold",
                    ha="center", va="top",
                    transform=ccrs.PlateCarree() if ccrs else None,
                )
            for lat, lon, pressure in centers["highs"]:
                ax.plot(
                    lon, lat, "b*",
                    markersize=self.style.center_marker_size,
                    transform=ccrs.PlateCarree() if ccrs else None,
                )
                ax.text(
                    lon, lat - 1, "H",
                    color="blue", fontsize=12, fontweight="bold",
                    ha="center", va="top",
                    transform=ccrs.PlateCarree() if ccrs else None,
                )

        # Title
        if title is None:
            title = f"Mean Sea Level Pressure - {source}"
        ax.set_title(title, fontsize=self.style.title_fontsize, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
            plt.close(fig)
            return save_path

        return None

    def plot_t850(
        self,
        field: Union[TemperatureField, EnsembleField],
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Plot 850 hPa temperature.

        Args:
            field: T850 field to plot
            title: Custom title
            save_path: Path to save figure

        Returns:
            Path to saved figure if save_path provided
        """
        if isinstance(field, EnsembleField):
            data = field.ensemble_mean - 273.15  # Convert to Celsius
            lats = field.lats
            lons = field.lons
            source = "HAEM Ensemble"
        else:
            data = field.in_celsius
            lats = field.lats
            lons = field.lons
            source = field.source_model.value.upper()

        # Create figure
        fig, ax = self._create_figure_with_map()
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Color-filled contours
        cmap = self._create_colormap(MeteoColorMaps.temperature_colormap())
        levels = self.contour_levels.t850_celsius

        if ccrs is not None:
            transform = ccrs.PlateCarree()
            cf = ax.contourf(
                lon_grid, lat_grid, data,
                levels=levels,
                cmap=cmap,
                alpha=self.style.fill_alpha,
                transform=transform,
                extend="both",
            )
            cs = ax.contour(
                lon_grid, lat_grid, data,
                levels=levels[::2],
                colors="black",
                linewidths=self.style.contour_linewidth * 0.5,
                transform=transform,
            )
            # 0°C isotherm
            ax.contour(
                lon_grid, lat_grid, data,
                levels=[0],
                colors="red",
                linewidths=self.style.contour_highlight_linewidth,
                transform=transform,
            )
        else:
            cf = ax.contourf(
                lon_grid, lat_grid, data,
                levels=levels,
                cmap=cmap,
                alpha=self.style.fill_alpha,
            )

        # Colorbar
        cbar = plt.colorbar(
            cf, ax=ax,
            shrink=self.style.colorbar_shrink,
            pad=self.style.colorbar_pad,
        )
        cbar.set_label("T850 (°C)", fontsize=self.style.colorbar_label_fontsize)

        # Title
        if title is None:
            title = f"850 hPa Temperature - {source}"
        ax.set_title(title, fontsize=self.style.title_fontsize, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
            plt.close(fig)
            return save_path

        return None

    def plot_ensemble_spread(
        self,
        field: EnsembleField,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Plot ensemble spread field.

        Args:
            field: Ensemble field with spread
            title: Custom title
            save_path: Path to save figure

        Returns:
            Path to saved figure
        """
        data = field.ensemble_spread
        lats = field.lats
        lons = field.lons

        fig, ax = self._create_figure_with_map()
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        cmap = self._create_colormap(MeteoColorMaps.spread_colormap())
        levels = self.contour_levels.spread

        if ccrs is not None:
            transform = ccrs.PlateCarree()
            cf = ax.contourf(
                lon_grid, lat_grid, data,
                levels=levels,
                cmap=cmap,
                alpha=self.style.fill_alpha,
                transform=transform,
                extend="max",
            )
        else:
            cf = ax.contourf(
                lon_grid, lat_grid, data,
                levels=levels,
                cmap=cmap,
                alpha=self.style.fill_alpha,
                extend="max",
            )

        cbar = plt.colorbar(
            cf, ax=ax,
            shrink=self.style.colorbar_shrink,
            pad=self.style.colorbar_pad,
        )
        cbar.set_label(f"Spread ({field.units})", fontsize=self.style.colorbar_label_fontsize)

        if title is None:
            title = f"Ensemble Spread - {field.field_name.upper()}"
        ax.set_title(title, fontsize=self.style.title_fontsize, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
            plt.close(fig)
            return save_path

        return None

    def plot_confidence_map(
        self,
        field: EnsembleField,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Plot spatial confidence field.

        Args:
            field: Ensemble field to derive confidence from
            title: Custom title
            save_path: Path to save figure

        Returns:
            Path to saved figure
        """
        data = field.get_confidence_field()
        lats = field.lats
        lons = field.lons

        fig, ax = self._create_figure_with_map()
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        cmap = self._create_colormap(MeteoColorMaps.confidence_colormap())
        levels = self.contour_levels.confidence

        if ccrs is not None:
            transform = ccrs.PlateCarree()
            cf = ax.contourf(
                lon_grid, lat_grid, data,
                levels=levels,
                cmap=cmap,
                alpha=self.style.fill_alpha,
                transform=transform,
            )
        else:
            cf = ax.contourf(
                lon_grid, lat_grid, data,
                levels=levels,
                cmap=cmap,
                alpha=self.style.fill_alpha,
            )

        cbar = plt.colorbar(
            cf, ax=ax,
            shrink=self.style.colorbar_shrink,
            pad=self.style.colorbar_pad,
        )
        cbar.set_label("Confidence (%)", fontsize=self.style.colorbar_label_fontsize)

        if title is None:
            title = f"Forecast Confidence - {field.field_name.upper()}"
        ax.set_title(title, fontsize=self.style.title_fontsize, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
            plt.close(fig)
            return save_path

        return None

    def _create_figure_with_map(self):
        """Create figure with map projection and features."""
        domain = self.style.domain

        if ccrs is not None:
            # Create appropriate projection
            if domain.projection.value == "lambert":
                proj = ccrs.LambertConformal(
                    central_longitude=domain.central_longitude,
                    standard_parallels=domain.standard_parallels,
                )
            elif domain.projection.value == "polar":
                proj = ccrs.NorthPolarStereo(central_longitude=domain.central_longitude)
            else:
                proj = ccrs.PlateCarree()

            fig, ax = plt.subplots(
                figsize=(self.style.figure_width, self.style.figure_height),
                subplot_kw={"projection": proj},
            )

            # Set extent
            ax.set_extent(
                [domain.lon_min, domain.lon_max, domain.lat_min, domain.lat_max],
                crs=ccrs.PlateCarree(),
            )

            # Add features
            ax.coastlines(
                linewidth=self.style.coastline_linewidth,
                color=self.style.coastline_color,
            )
            ax.add_feature(
                cfeature.BORDERS,
                linewidth=self.style.border_linewidth,
                edgecolor=self.style.border_color,
            )

            if self.style.show_gridlines:
                gl = ax.gridlines(
                    draw_labels=True,
                    linewidth=self.style.gridline_linewidth,
                    color=self.style.gridline_color,
                    alpha=self.style.gridline_alpha,
                )
                gl.top_labels = False
                gl.right_labels = False
        else:
            fig, ax = plt.subplots(
                figsize=(self.style.figure_width, self.style.figure_height)
            )
            ax.set_xlim(domain.lon_min, domain.lon_max)
            ax.set_ylim(domain.lat_min, domain.lat_max)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        return fig, ax

    def _create_colormap(self, colors: list[tuple[float, float, float]]):
        """Create a matplotlib colormap from color list."""
        import matplotlib.colors as mcolors

        return mcolors.LinearSegmentedColormap.from_list(
            "custom", colors, N=self.style.n_colors
        )


# Convenience functions for quick plotting
def create_z500_map(
    field: Union[GeopotentialField, EnsembleField],
    save_path: Optional[Path] = None,
    **kwargs,
) -> Optional[Path]:
    """Quick function to create Z500 map."""
    plotter = MeteoPlotter()
    return plotter.plot_z500(field, save_path=save_path, **kwargs)


def create_slp_map(
    field: Union[PressureField, EnsembleField],
    save_path: Optional[Path] = None,
    **kwargs,
) -> Optional[Path]:
    """Quick function to create SLP map."""
    plotter = MeteoPlotter()
    return plotter.plot_slp(field, save_path=save_path, **kwargs)


def create_t850_map(
    field: Union[TemperatureField, EnsembleField],
    save_path: Optional[Path] = None,
    **kwargs,
) -> Optional[Path]:
    """Quick function to create T850 map."""
    plotter = MeteoPlotter()
    return plotter.plot_t850(field, save_path=save_path, **kwargs)


def create_ensemble_map(
    field: EnsembleField,
    field_type: str = "z500",
    save_path: Optional[Path] = None,
    **kwargs,
) -> Optional[Path]:
    """Quick function to create ensemble map."""
    plotter = MeteoPlotter()
    if field_type == "z500":
        return plotter.plot_z500(field, save_path=save_path, **kwargs)
    elif field_type == "slp":
        return plotter.plot_slp(field, save_path=save_path, **kwargs)
    elif field_type == "t850":
        return plotter.plot_t850(field, save_path=save_path, **kwargs)
    else:
        raise ValueError(f"Unknown field type: {field_type}")
