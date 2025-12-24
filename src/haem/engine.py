"""
HAEM Weather Analysis Engine.

This is the main orchestration module that coordinates all HAEM components
to produce ensemble meteorological analysis and forecasts.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from haem.data.fetcher import MultiModelFetcher
from haem.data.processor import ModelDataValidator
from haem.analysis.model_analyzer import ModelAnalyzer
from haem.analysis.comparison import MultiModelComparator
from haem.ensemble.integrator import EnsembleIntegrator
from haem.visualization.plotter import MeteoPlotter
from haem.visualization.styles import PlotStyle
from haem.models.meteorological import ModelData, NWPModel
from haem.models.analysis import AnalysisResult, ModelComparison
from haem.models.ensemble import EnsembleConfig, EnsembleResult
from haem.utils.config import HAEMConfig
from haem.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class WeatherAnalysisEngine:
    """
    Main HAEM Weather Analysis Engine.

    Orchestrates the complete workflow:
    1. Fetch NWP model data
    2. Analyze individual models
    3. Compare multi-model solutions
    4. Generate ensemble integration
    5. Produce visualizations and reports

    Example:
        engine = WeatherAnalysisEngine()
        result = await engine.run_analysis(forecast_hours=[0, 24, 48, 72])
        engine.generate_report(result)
    """

    def __init__(
        self,
        config: Optional[HAEMConfig] = None,
    ):
        """
        Initialize the Weather Analysis Engine.

        Args:
            config: HAEM configuration (uses defaults if None)
        """
        self.config = config or HAEMConfig()

        # Setup logging
        setup_logging(
            level=self.config.log_level,
            log_file=self.config.log_file,
        )

        # Initialize components
        self.fetcher = MultiModelFetcher(
            models=self.config.models,
            cache_dir=self.config.cache_dir,
            use_synthetic_data=self.config.use_synthetic_data,
        )
        self.analyzer = ModelAnalyzer()
        self.comparator = MultiModelComparator()
        self.integrator = EnsembleIntegrator(self.config.ensemble)

        # Create output directory
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for results
        self.model_data: dict[NWPModel, ModelData] = {}
        self.analysis_results: dict[NWPModel, dict[int, AnalysisResult]] = {}
        self.comparisons: dict[int, ModelComparison] = {}
        self.ensemble_results: dict[int, EnsembleResult] = {}

    async def run_analysis(
        self,
        run_time: Optional[datetime] = None,
        forecast_hours: Optional[list[int]] = None,
        generate_plots: bool = True,
    ) -> dict[int, EnsembleResult]:
        """
        Run complete weather analysis workflow.

        Args:
            run_time: Model run time (uses latest if None)
            forecast_hours: Forecast hours to analyze
            generate_plots: Generate visualization plots

        Returns:
            Dict mapping forecast hour to EnsembleResult
        """
        if forecast_hours is None:
            forecast_hours = self.config.forecast_hours

        logger.info("=" * 70)
        logger.info("HAEM Weather Analysis Engine - Starting Analysis")
        logger.info("=" * 70)

        # Step 1: Fetch all model data
        logger.info("Step 1: Fetching NWP model data...")
        self.model_data = await self.fetcher.fetch_all_models(
            run_time=run_time,
            forecast_hours=forecast_hours,
            fields=self.config.fields,
        )
        logger.info(f"  Fetched data from {len(self.model_data)} models")

        # Step 2: Validate and clean data
        logger.info("Step 2: Validating model data...")
        for model, data in self.model_data.items():
            cleaned, quality = ModelDataValidator.validate(data)
            self.model_data[model] = cleaned
            logger.info(f"  {model.value.upper()}: Quality score = {data.data_quality_score:.2f}")

        # Step 3: Analyze each model
        logger.info("Step 3: Analyzing individual models...")
        for model, data in self.model_data.items():
            self.analysis_results[model] = {}
            for hour in forecast_hours:
                if hour in data.z500:
                    result = self.analyzer.analyze(data, hour)
                    self.analysis_results[model][hour] = result
            logger.info(f"  {model.value.upper()}: Analyzed {len(self.analysis_results[model])} times")

        # Step 4: Multi-model comparison
        logger.info("Step 4: Comparing models...")
        for hour in forecast_hours:
            # Check if all models have data for this hour
            available_models = {
                m: d for m, d in self.model_data.items()
                if hour in d.z500
            }
            if len(available_models) >= 2:
                comparison = self.comparator.compare(available_models, hour)
                self.comparisons[hour] = comparison
                logger.info(
                    f"  +{hour:03d}h: Agreement = {comparison.overall_agreement:.0f}%, "
                    f"Consensus = {comparison.pattern_consensus}"
                )

        # Step 5: Ensemble integration
        logger.info("Step 5: Generating ensemble...")
        for hour in forecast_hours:
            hour_analyses = {
                m: self.analysis_results[m][hour]
                for m in self.analysis_results
                if hour in self.analysis_results[m]
            }
            if hour_analyses:
                ensemble = self.integrator.integrate(
                    self.model_data,
                    hour_analyses,
                    hour,
                )
                self.ensemble_results[hour] = ensemble
                logger.info(
                    f"  +{hour:03d}h: Confidence = {ensemble.final_confidence_score:.0f}/100"
                )

        # Step 6: Generate visualizations
        if generate_plots:
            logger.info("Step 6: Generating visualizations...")
            await self._generate_all_plots(forecast_hours)

        logger.info("=" * 70)
        logger.info("Analysis Complete!")
        logger.info("=" * 70)

        return self.ensemble_results

    async def _generate_all_plots(
        self,
        forecast_hours: list[int],
    ) -> None:
        """Generate all visualization plots."""
        plotter = MeteoPlotter(
            style=PlotStyle.meteociel_style(),
            output_dir=self.output_dir,
        )

        for hour in forecast_hours:
            if hour not in self.ensemble_results:
                continue

            ensemble = self.ensemble_results[hour]

            # Z500 map
            if ensemble.z500:
                save_path = self.output_dir / f"z500_ensemble_{hour:03d}h.png"
                plotter.plot_z500(ensemble.z500, save_path=save_path, show_spread=True)
                logger.info(f"  Saved: {save_path}")

            # SLP map
            if ensemble.slp:
                save_path = self.output_dir / f"slp_ensemble_{hour:03d}h.png"
                plotter.plot_slp(ensemble.slp, save_path=save_path)
                logger.info(f"  Saved: {save_path}")

            # T850 map
            if ensemble.t850:
                save_path = self.output_dir / f"t850_ensemble_{hour:03d}h.png"
                plotter.plot_t850(ensemble.t850, save_path=save_path)
                logger.info(f"  Saved: {save_path}")

            # Spread map (Z500)
            if ensemble.z500:
                save_path = self.output_dir / f"spread_z500_{hour:03d}h.png"
                plotter.plot_ensemble_spread(ensemble.z500, save_path=save_path)
                logger.info(f"  Saved: {save_path}")

    def generate_report(
        self,
        forecast_hour: Optional[int] = None,
    ) -> str:
        """
        Generate a text report for the analysis.

        Args:
            forecast_hour: Specific hour to report (all hours if None)

        Returns:
            Formatted text report
        """
        lines = [
            "=" * 70,
            "HAEM ENSEMBLE WEATHER ANALYSIS REPORT",
            "=" * 70,
            "",
        ]

        hours = [forecast_hour] if forecast_hour else sorted(self.ensemble_results.keys())

        for hour in hours:
            if hour not in self.ensemble_results:
                continue

            ensemble = self.ensemble_results[hour]
            lines.append(ensemble.to_report())
            lines.append("")

        return "\n".join(lines)

    def get_ensemble_result(self, forecast_hour: int) -> Optional[EnsembleResult]:
        """Get ensemble result for a specific forecast hour."""
        return self.ensemble_results.get(forecast_hour)

    def get_model_analysis(
        self,
        model: NWPModel,
        forecast_hour: int,
    ) -> Optional[AnalysisResult]:
        """Get individual model analysis."""
        if model in self.analysis_results:
            return self.analysis_results[model].get(forecast_hour)
        return None

    def get_comparison(self, forecast_hour: int) -> Optional[ModelComparison]:
        """Get multi-model comparison for a specific hour."""
        return self.comparisons.get(forecast_hour)

    async def close(self) -> None:
        """Clean up resources."""
        await self.fetcher.close()


def run_analysis_sync(
    config: Optional[HAEMConfig] = None,
    forecast_hours: Optional[list[int]] = None,
    generate_plots: bool = True,
) -> dict[int, EnsembleResult]:
    """
    Synchronous wrapper for running analysis.

    Args:
        config: HAEM configuration
        forecast_hours: Forecast hours to analyze
        generate_plots: Generate visualization plots

    Returns:
        Dict mapping forecast hour to EnsembleResult
    """
    engine = WeatherAnalysisEngine(config)

    async def _run():
        try:
            return await engine.run_analysis(
                forecast_hours=forecast_hours,
                generate_plots=generate_plots,
            )
        finally:
            await engine.close()

    return asyncio.run(_run())
