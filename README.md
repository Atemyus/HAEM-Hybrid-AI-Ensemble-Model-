# HAEM - Hybrid AI-NWP Ensemble Model

A sophisticated meteorological analysis engine combining multiple Numerical Weather Prediction (NWP) models with AI-based validation and ensemble forecasting capabilities.

## Overview

HAEM produces physically consistent, explainable, and graphically plottable meteorological forecasts derived from:

- **Multi-model numerical data** (ECMWF, GFS, UKMO, GEM, ICON)
- **Independent AI-based meteorological reasoning**
- **Dynamic confidence scoring**
- **Ensemble mathematical aggregation**

The system generates continuous spatial fields equivalent to operational NWP products, with full support for visualization as professional meteorological maps.

## Features

### Supported NWP Models

| Model | Organization | Resolution |
|-------|--------------|------------|
| ECMWF | European Centre for Medium-Range Weather Forecasts | ~9 km |
| GFS | NCEP Global Forecast System | ~13 km |
| UKMO | UK Met Office | ~10 km |
| GEM | Canadian Global Environmental Multiscale | ~15 km |
| ICON | DWD Icosahedral Nonhydrostatic | ~13 km |

### Meteorological Fields

- **Z500**: 500 hPa Geopotential Height
- **Z850**: 850 hPa Geopotential Height
- **T850**: 850 hPa Temperature
- **SLP**: Mean Sea Level Pressure

### Analysis Capabilities

- Individual model analysis with synoptic pattern detection
- Multi-model comparison and synthesis
- Physical-dynamical interpretation with causal explanations
- Confidence scoring and reliability assessment
- Weighted ensemble integration

### Visualization

- Color-filled contour maps (Meteociel-style)
- Isoline overlays
- Ensemble spread visualization
- Confidence maps
- Multiple projection support (Lambert Conformal, Polar Stereographic)

## Installation

```bash
# Clone the repository
git clone https://github.com/example/haem.git
cd haem

# Install with pip
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Dependencies

- Python 3.10+
- NumPy, SciPy, Pandas
- xarray, netCDF4
- Matplotlib, Cartopy
- Pydantic, Typer, Rich

## Quick Start

### Command Line Interface

```bash
# Run analysis with default settings
haem analyze

# Specify forecast hours and models
haem analyze --hours 0,24,48,72,96 --models ecmwf,gfs,ukmo

# Generate output without plots
haem analyze --no-plots

# List available models
haem models
```

### Python API

```python
import asyncio
from haem import WeatherAnalysisEngine

async def run():
    engine = WeatherAnalysisEngine()

    # Run complete analysis
    results = await engine.run_analysis(
        forecast_hours=[0, 24, 48, 72, 96, 120],
        generate_plots=True,
    )

    # Get ensemble result for specific hour
    ensemble_72h = engine.get_ensemble_result(72)

    # Generate text report
    report = engine.generate_report()
    print(report)

    await engine.close()

asyncio.run(run())
```

## Architecture

```
haem/
├── models/           # Data models and schemas
│   ├── meteorological.py  # NWP field definitions
│   ├── analysis.py        # Analysis result structures
│   └── ensemble.py        # Ensemble integration models
├── data/             # Data fetching and processing
│   ├── fetcher.py         # Multi-model data fetcher
│   ├── processor.py       # Grid interpolation, QC
│   └── cache.py           # Data caching
├── analysis/         # Analysis algorithms
│   ├── model_analyzer.py  # Individual model analysis
│   ├── pattern_detector.py # Synoptic pattern detection
│   ├── interpreter.py     # Physical interpretation
│   ├── confidence.py      # Confidence scoring
│   └── comparison.py      # Multi-model comparison
├── ensemble/         # Ensemble integration
│   ├── integrator.py      # Weighted aggregation
│   └── weighting.py       # Weight calculation
├── visualization/    # Plotting and maps
│   ├── plotter.py         # Map generation
│   ├── styles.py          # Color maps, styling
│   └── maps.py            # Convenience functions
├── engine.py         # Main orchestration
└── cli.py            # Command line interface
```

## Ensemble Integration

The ensemble uses weighted aggregation:

```
Z_final(x,y,t) = Σ [ Z_model_i(x,y,t) × W_model_i × W_AI_i ]
```

Where:
- `Z_model_i` are the original NWP fields
- `W_model_i` are historical model skill weights
- `W_AI_i` are AI confidence and coherence weights

## Output Structure

Each analysis produces:

1. **Individual Model Analysis** - Per-model synoptic assessment
2. **Multi-Model Comparative Synthesis** - Agreement and divergence analysis
3. **Physical-Dynamical Explanation** - Causal interpretation
4. **Reliability Scoring** - Per-model reliability (0-100)
5. **Ensemble Scenario Description** - Identified forecast scenarios
6. **Probabilistic Confidence Assessment** - Spatial confidence fields
7. **Final Ensemble Confidence Score** - Overall score (0-100)

## Configuration

Create `~/.haem/config.json`:

```json
{
  "models": ["ecmwf", "gfs", "ukmo", "gem", "icon"],
  "forecast_hours": [0, 6, 12, 24, 48, 72, 96, 120, 144, 168],
  "output_dir": "./output",
  "log_level": "INFO"
}
```

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=haem --cov-report=html
```

## Core Principles

- **Scientific rigor** over narrative
- **Explainability** over black-box output
- **Physical causality** over pattern description
- **Transparency** over deterministic claims
- **Ensemble consistency** over single-model dominance

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.
