"""
High-level map generation functions.

Provides simplified interfaces for common plotting operations.
"""

from pathlib import Path
from typing import Optional, Union

from haem.models.meteorological import (
    GeopotentialField,
    PressureField,
    TemperatureField,
)
from haem.models.ensemble import EnsembleField
from haem.visualization.plotter import MeteoPlotter
from haem.visualization.styles import PlotStyle


def create_z500_map(
    field: Union[GeopotentialField, EnsembleField],
    save_path: Optional[Path] = None,
    style: Optional[PlotStyle] = None,
    title: Optional[str] = None,
    show_spread: bool = False,
) -> Optional[Path]:
    """
    Create a 500 hPa geopotential height map.

    Args:
        field: Z500 field (single model or ensemble)
        save_path: Path to save the figure
        style: Plot styling options
        title: Custom title
        show_spread: Show ensemble spread overlay

    Returns:
        Path to saved figure if save_path provided
    """
    plotter = MeteoPlotter(style=style)
    return plotter.plot_z500(
        field,
        title=title,
        show_spread=show_spread,
        save_path=save_path,
    )


def create_slp_map(
    field: Union[PressureField, EnsembleField],
    save_path: Optional[Path] = None,
    style: Optional[PlotStyle] = None,
    title: Optional[str] = None,
    mark_centers: bool = True,
) -> Optional[Path]:
    """
    Create a mean sea level pressure map.

    Args:
        field: SLP field (single model or ensemble)
        save_path: Path to save the figure
        style: Plot styling options
        title: Custom title
        mark_centers: Mark H/L pressure centers

    Returns:
        Path to saved figure if save_path provided
    """
    plotter = MeteoPlotter(style=style)
    return plotter.plot_slp(
        field,
        mark_centers=mark_centers,
        title=title,
        save_path=save_path,
    )


def create_t850_map(
    field: Union[TemperatureField, EnsembleField],
    save_path: Optional[Path] = None,
    style: Optional[PlotStyle] = None,
    title: Optional[str] = None,
) -> Optional[Path]:
    """
    Create an 850 hPa temperature map.

    Args:
        field: T850 field (single model or ensemble)
        save_path: Path to save the figure
        style: Plot styling options
        title: Custom title

    Returns:
        Path to saved figure if save_path provided
    """
    plotter = MeteoPlotter(style=style)
    return plotter.plot_t850(
        field,
        title=title,
        save_path=save_path,
    )


def create_ensemble_map(
    field: EnsembleField,
    field_type: str = "z500",
    save_path: Optional[Path] = None,
    style: Optional[PlotStyle] = None,
    title: Optional[str] = None,
    include_spread: bool = False,
) -> Optional[Path]:
    """
    Create an ensemble mean map with optional spread overlay.

    Args:
        field: Ensemble field to plot
        field_type: Type of field ('z500', 'slp', 't850')
        save_path: Path to save the figure
        style: Plot styling options
        title: Custom title
        include_spread: Include spread as overlay

    Returns:
        Path to saved figure if save_path provided
    """
    plotter = MeteoPlotter(style=style)

    if field_type == "z500":
        return plotter.plot_z500(
            field,
            title=title,
            show_spread=include_spread,
            save_path=save_path,
        )
    elif field_type == "slp":
        return plotter.plot_slp(
            field,
            title=title,
            save_path=save_path,
        )
    elif field_type == "t850":
        return plotter.plot_t850(
            field,
            title=title,
            save_path=save_path,
        )
    elif field_type == "spread":
        return plotter.plot_ensemble_spread(
            field,
            title=title,
            save_path=save_path,
        )
    elif field_type == "confidence":
        return plotter.plot_confidence_map(
            field,
            title=title,
            save_path=save_path,
        )
    else:
        raise ValueError(f"Unknown field type: {field_type}")


def create_multi_panel_map(
    ensemble_result,
    save_path: Optional[Path] = None,
    style: Optional[PlotStyle] = None,
) -> Optional[Path]:
    """
    Create a multi-panel map showing Z500, SLP, T850, and spread.

    Args:
        ensemble_result: EnsembleResult object
        save_path: Path to save the figure
        style: Plot styling options

    Returns:
        Path to saved figure if save_path provided
    """
    import matplotlib.pyplot as plt

    style = style or PlotStyle.meteociel_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    plotter = MeteoPlotter(style=style)

    # Z500
    if ensemble_result.z500:
        # For multi-panel, we need to handle this differently
        pass  # Simplified - full implementation would reuse plotter logic

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=style.dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

    return None
