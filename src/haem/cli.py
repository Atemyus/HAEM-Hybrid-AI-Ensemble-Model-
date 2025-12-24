"""
HAEM Command Line Interface.

Provides a Typer-based CLI for running the Weather Analysis Engine.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from haem import __version__
from haem.engine import WeatherAnalysisEngine
from haem.utils.config import HAEMConfig, load_config
from haem.models.meteorological import NWPModel

app = typer.Typer(
    name="haem",
    help="HAEM - Hybrid AI-NWP Ensemble Model",
    add_completion=False,
)
console = Console()


@app.command()
def analyze(
    hours: Optional[str] = typer.Option(
        None,
        "--hours", "-h",
        help="Forecast hours to analyze (comma-separated, e.g., '0,24,48,72')"
    ),
    models: Optional[str] = typer.Option(
        None,
        "--models", "-m",
        help="Models to include (comma-separated, e.g., 'ecmwf,gfs,ukmo')"
    ),
    output: Path = typer.Option(
        Path("./output"),
        "--output", "-o",
        help="Output directory for results"
    ),
    no_plots: bool = typer.Option(
        False,
        "--no-plots",
        help="Skip generating visualization plots"
    ),
    synthetic: bool = typer.Option(
        True,
        "--synthetic/--real",
        help="Use synthetic data (default) or attempt real data fetch"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output"
    ),
) -> None:
    """
    Run weather analysis and generate ensemble forecast.

    This command fetches NWP model data, performs analysis, generates
    an ensemble forecast, and produces visualizations.
    """
    console.print(Panel.fit(
        "[bold blue]HAEM Weather Analysis Engine[/bold blue]\n"
        f"Version {__version__}",
        border_style="blue"
    ))

    # Load or create configuration
    if config_file and config_file.exists():
        config = HAEMConfig.from_file(config_file)
    else:
        config = HAEMConfig()

    # Override with CLI options
    config.output_dir = output
    config.use_synthetic_data = synthetic
    config.log_level = "DEBUG" if verbose else "INFO"

    if hours:
        config.forecast_hours = [int(h.strip()) for h in hours.split(",")]

    if models:
        model_map = {m.value: m for m in NWPModel}
        config.models = [
            model_map[m.strip().lower()]
            for m in models.split(",")
            if m.strip().lower() in model_map
        ]

    # Display configuration
    config_table = Table(title="Configuration", show_header=False)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Models", ", ".join(m.value.upper() for m in config.models))
    config_table.add_row("Forecast Hours", str(config.forecast_hours))
    config_table.add_row("Output Directory", str(config.output_dir))
    config_table.add_row("Data Mode", "Synthetic" if config.use_synthetic_data else "Real")
    console.print(config_table)
    console.print()

    # Run analysis
    engine = WeatherAnalysisEngine(config)

    async def run():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running analysis...", total=None)
            try:
                results = await engine.run_analysis(
                    forecast_hours=config.forecast_hours,
                    generate_plots=not no_plots,
                )
                return results
            finally:
                await engine.close()

    results = asyncio.run(run())

    # Display summary
    console.print()
    console.print(Panel.fit(
        "[bold green]Analysis Complete![/bold green]",
        border_style="green"
    ))

    # Summary table
    summary_table = Table(title="Ensemble Summary")
    summary_table.add_column("Forecast Hour", style="cyan")
    summary_table.add_column("Confidence", style="yellow")
    summary_table.add_column("Dominant Pattern", style="magenta")

    for hour, result in sorted(results.items()):
        summary_table.add_row(
            f"+{hour:03d}h",
            f"{result.final_confidence_score:.0f}/100",
            result.physical_interpretation.dominant_pattern.value.replace("_", " ").title(),
        )

    console.print(summary_table)
    console.print()
    console.print(f"[dim]Output saved to: {config.output_dir}[/dim]")


@app.command()
def report(
    hour: int = typer.Argument(..., help="Forecast hour to report"),
    output: Path = typer.Option(
        Path("./output"),
        "--output", "-o",
        help="Output directory with analysis results"
    ),
) -> None:
    """
    Display detailed report for a specific forecast hour.
    """
    console.print(f"[bold]Ensemble Report for +{hour}h[/bold]")
    console.print("[dim]Note: Report requires running 'analyze' first[/dim]")


@app.command()
def config(
    show: bool = typer.Option(
        False,
        "--show",
        help="Show current configuration"
    ),
    generate: Optional[Path] = typer.Option(
        None,
        "--generate",
        help="Generate default configuration file"
    ),
) -> None:
    """
    Manage HAEM configuration.
    """
    if generate:
        config = HAEMConfig()
        config.to_file(generate)
        console.print(f"[green]Configuration saved to: {generate}[/green]")
    elif show:
        config = load_config()
        console.print(Panel.fit(str(config.model_dump()), title="Current Configuration"))
    else:
        console.print("Use --show to display config or --generate to create default config")


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"[bold]HAEM[/bold] version {__version__}")


@app.command()
def models() -> None:
    """List available NWP models."""
    table = Table(title="Available NWP Models")
    table.add_column("Code", style="cyan")
    table.add_column("Full Name", style="green")
    table.add_column("Resolution", style="yellow")

    for model in NWPModel:
        table.add_row(
            model.value.upper(),
            model.full_name,
            model.resolution,
        )

    console.print(table)


if __name__ == "__main__":
    app()
