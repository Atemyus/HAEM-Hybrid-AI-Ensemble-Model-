"""
HAEM Streamlit Dashboard.

Interactive web dashboard for visualizing ensemble weather analysis
with multi-AI interpretation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import io

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Configure page
st.set_page_config(
    page_title="HAEM - Weather Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import HAEM modules
from haem.models.meteorological import NWPModel
from haem.data.meteociel_fields import FieldSelection, FieldPresets, MeteocielField
from haem.ai.providers import AIProvider, AIProviderConfig, AIPresets, MultiAIOrchestrator
from haem.config.settings import get_api_keys, get_app_config
from haem.data.openmeteo import MultiModelOpenMeteoFetcher, OpenMeteoConfig
from haem.data.cache import DashboardCache

logger = logging.getLogger(__name__)

# Initialize dashboard cache
dashboard_cache = DashboardCache()

# Load API keys
api_keys = get_api_keys()
app_config = get_app_config()


# ============================================================================
# DATA FETCHING AND ANALYSIS
# ============================================================================

def get_current_synoptic_run() -> str:
    """Get the current synoptic run identifier for cache key."""
    now = datetime.utcnow()
    # Data is available ~4 hours after run
    available_time = now - timedelta(hours=4)
    run_hour = (available_time.hour // 6) * 6
    run_date = available_time.date()
    return f"{run_date.isoformat()}_{run_hour:02d}Z"


@st.cache_data(ttl=3600, show_spinner=False)  # 1 hour TTL
def fetch_weather_data_cached(models_tuple: tuple, forecast_hours_tuple: tuple, synoptic_run: str):
    """
    Fetch weather data with Streamlit caching.

    The synoptic_run parameter ensures cache is invalidated when new model data is available.
    """
    models = list(models_tuple)
    forecast_hours = list(forecast_hours_tuple)

    config = OpenMeteoConfig(
        latitude_min=35.0,
        latitude_max=65.0,
        longitude_min=-15.0,
        longitude_max=30.0,
        grid_resolution=1.0,  # Faster with coarser grid
        forecast_days=7,
    )
    fetcher = MultiModelOpenMeteoFetcher(models=models, config=config)

    # Run async in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(fetcher.fetch_all_models(forecast_hours=forecast_hours))
        return result
    finally:
        loop.close()


async def fetch_weather_data(models: list[NWPModel], forecast_hours: list[int]):
    """Fetch weather data from Open-Meteo for all selected models."""
    config = OpenMeteoConfig(
        latitude_min=35.0,
        latitude_max=65.0,
        longitude_min=-15.0,
        longitude_max=30.0,
        grid_resolution=1.0,  # Faster with coarser grid
        forecast_days=7,
    )
    fetcher = MultiModelOpenMeteoFetcher(models=models, config=config)
    return await fetcher.fetch_all_models(forecast_hours=forecast_hours)


async def run_ai_analysis(model_data: dict, ai_configs: list[AIProviderConfig], forecast_hour: int):
    """Run AI analysis on the weather data."""
    if not ai_configs:
        return {}

    orchestrator = MultiAIOrchestrator(ai_configs)

    # Prepare data summary for AI
    data_summary = {}
    for model, data in model_data.items():
        if forecast_hour in data.z500:
            z500 = data.z500[forecast_hour]
            data_summary[model.value] = {
                "z500_mean": float(np.nanmean(z500.data)),
                "z500_min": float(np.nanmin(z500.data)),
                "z500_max": float(np.nanmax(z500.data)),
            }
        if forecast_hour in data.t850:
            t850 = data.t850[forecast_hour]
            data_summary[model.value]["t850_mean"] = float(np.nanmean(t850.data) - 273.15)
        if forecast_hour in data.slp:
            slp = data.slp[forecast_hour]
            data_summary[model.value]["slp_min"] = float(np.nanmin(slp.data))
            data_summary[model.value]["slp_max"] = float(np.nanmax(slp.data))

    results = await orchestrator.analyze_all(
        model_data=data_summary,
        field_selection=["Z500", "T850", "SLP"],
        forecast_hour=forecast_hour,
    )

    # Convert to dict format for display
    ai_analyses = {}
    for result in results:
        ai_analyses[result.provider.full_name] = {
            'confidence': result.confidence_score,
            'summary': result.synoptic_summary,
            'physics': result.physical_interpretation,
            'uncertainty': result.confidence_assessment,
            'patterns': result.pattern_identification,
            'findings': result.key_findings,
            'warnings': result.warnings,
            'inference_time_ms': result.inference_time_ms,
        }

    return ai_analyses


def compute_ensemble(model_data: dict, forecast_hour: int):
    """Compute ensemble mean and spread from model data."""
    from dataclasses import dataclass, field as dataclass_field

    @dataclass
    class EnsembleField:
        ensemble_mean: np.ndarray
        ensemble_spread: np.ndarray
        lats: np.ndarray
        lons: np.ndarray

    @dataclass
    class EnsembleResult:
        valid_time: datetime
        z500: Optional[EnsembleField] = None
        z850: Optional[EnsembleField] = None
        t850: Optional[EnsembleField] = None
        t500: Optional[EnsembleField] = None
        slp: Optional[EnsembleField] = None
        t2m: Optional[EnsembleField] = None
        precip: Optional[EnsembleField] = None
        snow: Optional[EnsembleField] = None
        wind_10m: Optional[EnsembleField] = None
        wind_300: Optional[EnsembleField] = None
        cape: Optional[EnsembleField] = None
        model_weights: list = dataclass_field(default_factory=list)
        final_confidence_score: float = 75.0
        synoptic_summary: str = ""
        warnings: list = dataclass_field(default_factory=list)

    lats = None
    lons = None
    valid_time = datetime.utcnow()

    # Helper function to collect and compute ensemble for a field
    def collect_field(field_name: str):
        nonlocal lats, lons, valid_time
        arrays = []
        for model, data in model_data.items():
            field_dict = getattr(data, field_name, {})
            if forecast_hour in field_dict:
                field = field_dict[forecast_hour]
                if field.data is not None and not np.all(np.isnan(field.data)):
                    arrays.append(field.data)
                    if lats is None:
                        lats = field.lats
                        lons = field.lons
                        valid_time = field.valid_time
        if len(arrays) >= 1 and lats is not None:
            stack = np.stack(arrays, axis=0)
            ensemble_mean = np.nanmean(stack, axis=0)
            # Only compute std if we have more than 1 model, otherwise set spread to 0
            if len(arrays) > 1:
                ensemble_spread = np.nanstd(stack, axis=0)
            else:
                ensemble_spread = np.zeros_like(ensemble_mean)
            return EnsembleField(
                ensemble_mean=ensemble_mean,
                ensemble_spread=ensemble_spread,
                lats=lats,
                lons=lons,
            )
        return None

    result = EnsembleResult(valid_time=valid_time)

    # Collect all fields
    result.z500 = collect_field('z500')
    result.z850 = collect_field('z850')
    result.t850 = collect_field('t850')
    result.t500 = collect_field('t500')
    result.slp = collect_field('slp')
    result.t2m = collect_field('t2m')
    result.precip = collect_field('precip')
    result.snow = collect_field('snow')
    result.wind_10m = collect_field('wind_10m')
    result.wind_300 = collect_field('wind_300')
    result.cape = collect_field('cape')

    return result


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'model_data' not in st.session_state:
        st.session_state.model_data = None
    if 'ensemble_results' not in st.session_state:
        st.session_state.ensemble_results = None
    if 'ai_analyses' not in st.session_state:
        st.session_state.ai_analyses = {}
    if 'current_hour' not in st.session_state:
        st.session_state.current_hour = 48
    if 'selected_field' not in st.session_state:
        st.session_state.selected_field = 'z500'
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False


# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

def render_sidebar():
    """Render the configuration sidebar."""

    st.sidebar.title("⚙️ Configurazione")

    # Model selection
    st.sidebar.subheader("🌐 Modelli NWP")
    models = {
        NWPModel.ECMWF: st.sidebar.checkbox("ECMWF (IFS)", value=True),
        NWPModel.GFS: st.sidebar.checkbox("GFS (NOAA)", value=True),
        NWPModel.ICON: st.sidebar.checkbox("ICON (DWD)", value=True),
        NWPModel.GEM: st.sidebar.checkbox("GEM (Canada)", value=True),
        NWPModel.UKMO: st.sidebar.checkbox("ARPEGE (Météo-France)", value=True),
    }
    selected_models = [m for m, selected in models.items() if selected]

    # Field selection
    st.sidebar.subheader("📊 Campi Meteorologici")
    field_preset = st.sidebar.selectbox(
        "Preset",
        ["Analisi Sinottica", "Analisi Convettiva", "Meteo Invernale", "Completo"],
        index=0,
    )

    preset_map = {
        "Analisi Sinottica": FieldPresets.synoptic_analysis(),
        "Analisi Convettiva": FieldPresets.convective_analysis(),
        "Meteo Invernale": FieldPresets.winter_weather(),
        "Completo": FieldPresets.full_analysis(),
    }
    field_selection = preset_map[field_preset]

    # Show selected fields
    with st.sidebar.expander("Campi selezionati"):
        for field in field_selection.get_selected_fields():
            st.write(f"✓ {field.description}")

    # AI selection
    st.sidebar.subheader("🤖 AI Providers")
    ai_configs = []

    # Show API key status
    with st.sidebar.expander("API Keys Status"):
        if api_keys.has_claude:
            st.success("Claude: Configurato")
        else:
            st.warning("Claude: Non configurato")
        if api_keys.has_gpt4:
            st.success("GPT-4: Configurato")
        else:
            st.warning("GPT-4: Non configurato")
        if api_keys.has_gemini:
            st.success("Gemini: Configurato")
        else:
            st.warning("Gemini: Non configurato")

        if not api_keys.available_providers:
            st.info("Modalita demo attiva")

    # Only show enabled providers
    if api_keys.has_claude:
        if st.sidebar.checkbox("Claude (Anthropic)", value=True):
            ai_configs.append(AIProviderConfig(provider=AIProvider.CLAUDE))
    if api_keys.has_gpt4:
        if st.sidebar.checkbox("GPT-4 (OpenAI)", value=True):
            ai_configs.append(AIProviderConfig(provider=AIProvider.GPT4))
    if api_keys.has_gemini:
        if st.sidebar.checkbox("Gemini (Google)", value=True):
            ai_configs.append(AIProviderConfig(provider=AIProvider.GEMINI))

    # Forecast hours
    st.sidebar.subheader("⏱️ Ore Previsione")
    max_hours = st.sidebar.slider("Massimo ore", 72, 168, 120, step=24)
    step_hours = st.sidebar.selectbox("Intervallo", [3, 6, 12], index=1)
    forecast_hours = list(range(0, max_hours + 1, step_hours))

    # Auto-refresh settings
    st.sidebar.subheader("🔄 Aggiornamento Automatico")
    auto_refresh = st.sidebar.checkbox("Auto-refresh attivo", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh

    # Synoptic update times explanation
    st.sidebar.caption("Aggiornamento sincronizzato con uscite sinottiche (00/06/12/18 UTC)")

    # Show last update time
    if st.session_state.last_update:
        st.sidebar.info(f"Ultimo update: {st.session_state.last_update.strftime('%H:%M:%S UTC')}")

    # Show next synoptic run
    now_utc = datetime.utcnow()
    synoptic_hours = [0, 6, 12, 18]
    # Models are available ~3-4h after run time
    availability_delay = 4  # hours

    # Find next available synoptic run
    current_hour = now_utc.hour
    for sh in synoptic_hours:
        available_hour = (sh + availability_delay) % 24
        if current_hour < available_hour:
            next_run = sh
            hours_until = available_hour - current_hour
            break
    else:
        next_run = synoptic_hours[0]
        hours_until = (24 - current_hour) + availability_delay

    st.sidebar.write(f"Prossima uscita completa: **{next_run:02d}Z** (disponibile tra ~{hours_until}h)")

    # Show model update frequencies
    with st.sidebar.expander("Uscite sinottiche modelli"):
        st.write("**Runs principali (00/06/12/18 UTC):**")
        st.write("- ECMWF IFS: ogni 6h")
        st.write("- GFS: ogni 6h")
        st.write("- ICON: ogni 6h")
        st.write("- GEM: ogni 6h")
        st.write("- ARPEGE: ogni 6h")
        st.write("")
        st.write("*Dati disponibili ~3-4h dopo il run*")

    # Run button
    st.sidebar.markdown("---")
    run_analysis = st.sidebar.button("🚀 ESEGUI ANALISI", use_container_width=True)

    return {
        'models': selected_models,
        'field_selection': field_selection,
        'ai_configs': ai_configs,
        'forecast_hours': forecast_hours,
        'run_analysis': run_analysis,
        'auto_refresh': auto_refresh,
        'next_synoptic_run': next_run,
        'hours_until_next': hours_until,
    }


# ============================================================================
# MAP VISUALIZATION
# ============================================================================

def create_map_figure(
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str,
    field_type: str = 'z500',
    show_contours: bool = True,
) -> plt.Figure:
    """Create a matplotlib figure for map display."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Choose colormap based on field type
    if field_type == 'z500':
        cmap = plt.cm.RdYlBu_r
        levels = np.arange(500, 596, 4)
        label = "Z500 (dam)"
    elif field_type == 'z850':
        cmap = plt.cm.RdYlBu_r
        levels = np.arange(120, 160, 2)
        label = "Z850 (dam)"
    elif field_type == 't850':
        cmap = plt.cm.RdYlBu_r
        levels = np.arange(-30, 25, 2)
        label = "T850 (°C)"
    elif field_type == 't500':
        cmap = plt.cm.RdYlBu_r
        levels = np.arange(-50, -10, 2)
        label = "T500 (°C)"
    elif field_type == 'slp':
        cmap = plt.cm.viridis
        levels = np.arange(980, 1040, 4)
        label = "SLP (hPa)"
    elif field_type == 't2m':
        cmap = plt.cm.RdYlBu_r
        levels = np.arange(-20, 40, 2)
        label = "T2m (°C)"
    elif field_type == 'precip':
        cmap = plt.cm.Blues
        levels = np.array([0, 0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100])
        label = "Precipitazioni (mm)"
    elif field_type == 'snow':
        cmap = plt.cm.cool
        levels = np.array([0, 1, 2, 5, 10, 20, 30, 50, 75, 100])
        label = "Neve (cm)"
    elif field_type == 'wind_10m':
        cmap = plt.cm.YlOrRd
        levels = np.arange(0, 30, 2)
        label = "Vento 10m (m/s)"
    elif field_type == 'wind_300':
        cmap = plt.cm.jet
        levels = np.arange(40, 200, 10)
        label = "Jet Stream 300hPa (kt)"
    elif field_type == 'cape':
        cmap = plt.cm.YlOrRd
        levels = np.array([0, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000])
        label = "CAPE (J/kg)"
    elif field_type == 'spread':
        cmap = plt.cm.YlOrRd
        levels = np.arange(0, 20, 2)
        label = "Spread (dam)"
    else:
        cmap = plt.cm.viridis
        levels = 20
        label = field_type

    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Filled contours
    cf = ax.contourf(
        lon_grid, lat_grid, data,
        levels=levels if isinstance(levels, np.ndarray) else 20,
        cmap=cmap,
        extend='both',
        alpha=0.85,
    )

    # Contour lines
    if show_contours:
        cs = ax.contour(
            lon_grid, lat_grid, data,
            levels=levels[::2] if isinstance(levels, np.ndarray) else 10,
            colors='black',
            linewidths=0.5,
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(label, fontsize=10)

    # Labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def render_map_panel(ensemble_data, hour: int, field_type: str):
    """Render the main map panel."""

    # Field display names
    field_names = {
        'z500': 'Z500 Geopotenziale',
        'z850': 'Z850 Geopotenziale',
        't850': 'T850 Temperatura',
        't500': 'T500 Temperatura',
        'slp': 'SLP Pressione',
        't2m': 'T2m Temperatura',
        'precip': 'Precipitazioni',
        'snow': 'Neve',
        'wind_10m': 'Vento 10m',
        'wind_300': 'Jet Stream',
        'cape': 'CAPE',
        'spread': 'Spread Ensemble',
    }
    display_name = field_names.get(field_type, field_type.upper())

    st.subheader(f"🗺️ {display_name} (+{hour}h)")

    if ensemble_data is None:
        # Show placeholder
        st.info("👆 Configura i parametri e clicca 'ESEGUI ANALISI' per vedere le mappe")

        # Show demo map
        demo_lats = np.linspace(30, 70, 81)
        demo_lons = np.linspace(-30, 40, 141)
        lon_grid, lat_grid = np.meshgrid(demo_lons, demo_lats)

        # Generate demo data (realistic-looking Z500)
        demo_data = 552 - (lat_grid - 50) * 0.5 + 8 * np.sin(np.deg2rad(lon_grid * 4))

        fig = create_map_figure(
            demo_data, demo_lats, demo_lons,
            f"Z500 Ensemble [DEMO] - +{hour}h",
            field_type='z500'
        )
        st.pyplot(fig)
        plt.close(fig)
        return

    # Real data rendering
    if hour in ensemble_data and ensemble_data[hour] is not None:
        result = ensemble_data[hour]
        data = None
        lats = None
        lons = None

        # Get the appropriate field data
        if field_type == 'z500' and result.z500 is not None:
            data = result.z500.ensemble_mean
            lats = result.z500.lats
            lons = result.z500.lons
        elif field_type == 'z850' and result.z850 is not None:
            data = result.z850.ensemble_mean
            lats = result.z850.lats
            lons = result.z850.lons
        elif field_type == 't850' and result.t850 is not None:
            data = result.t850.ensemble_mean - 273.15  # K to C
            lats = result.t850.lats
            lons = result.t850.lons
        elif field_type == 't500' and result.t500 is not None:
            data = result.t500.ensemble_mean - 273.15  # K to C
            lats = result.t500.lats
            lons = result.t500.lons
        elif field_type == 'slp' and result.slp is not None:
            data = result.slp.ensemble_mean
            lats = result.slp.lats
            lons = result.slp.lons
        elif field_type == 't2m' and result.t2m is not None:
            data = result.t2m.ensemble_mean
            lats = result.t2m.lats
            lons = result.t2m.lons
        elif field_type == 'precip' and result.precip is not None:
            data = result.precip.ensemble_mean
            lats = result.precip.lats
            lons = result.precip.lons
        elif field_type == 'snow' and result.snow is not None:
            data = result.snow.ensemble_mean
            lats = result.snow.lats
            lons = result.snow.lons
        elif field_type == 'wind_10m' and result.wind_10m is not None:
            data = result.wind_10m.ensemble_mean
            lats = result.wind_10m.lats
            lons = result.wind_10m.lons
        elif field_type == 'wind_300' and result.wind_300 is not None:
            data = result.wind_300.ensemble_mean
            lats = result.wind_300.lats
            lons = result.wind_300.lons
        elif field_type == 'cape' and result.cape is not None:
            data = result.cape.ensemble_mean
            lats = result.cape.lats
            lons = result.cape.lons
        elif field_type == 'spread' and result.z500 is not None:
            data = result.z500.ensemble_spread
            lats = result.z500.lats
            lons = result.z500.lons

        if data is None:
            st.warning(f"Dati non disponibili per {display_name}")
            return

        valid_time = result.valid_time.strftime("%Y-%m-%d %H:%MZ")
        title = f"{display_name} Ensemble - Valid: {valid_time}"

        fig = create_map_figure(data, lats, lons, title, field_type)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning(f"Nessun dato disponibile per +{hour}h")


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def render_model_comparison(ensemble_results, hour: int):
    """Render model comparison chart."""

    st.subheader("📊 Confronto Modelli")

    if ensemble_results is None or hour not in ensemble_results:
        # Demo data
        models = ['ECMWF', 'GFS', 'ICON', 'GEM', 'ARPEGE']
        weights = [0.89, 0.72, 0.81, 0.65, 0.78]
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f']
    else:
        result = ensemble_results[hour]
        models = [mw.model.value.upper() for mw in result.model_weights]
        weights = [mw.combined_weight for mw in result.model_weights]
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f'][:len(models)]

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 4))

    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, [w * 100 for w in weights], color=colors, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Peso Ensemble (%)')
    ax.set_xlim(0, 100)

    # Add value labels
    for bar, weight in zip(bars, weights):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{weight*100:.0f}%', va='center', fontsize=10)

    # Calculate agreement
    agreement = np.std(weights)
    agreement_level = "Alto" if agreement < 0.1 else "Moderato" if agreement < 0.2 else "Basso"

    ax.set_title(f'Accordo Modelli: {agreement_level}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ============================================================================
# AI ANALYSIS PANEL
# ============================================================================

def render_ai_analysis(ai_analyses: dict, hour: int):
    """Render AI analysis panel with reasoning."""

    st.subheader("🤖 Analisi AI - Motivazioni e Riflessioni")

    if not ai_analyses:
        # Demo analysis
        demo_analyses = {
            'Claude': {
                'confidence': 85,
                'summary': """Una saccatura atlantica si approfondisce verso l'Europa occidentale,
                con geopotenziali in calo di 8 dam nelle prossime 48h. Il pattern è guidato
                dalla propagazione di onde di Rossby con numero d'onda 4-5.""",
                'physics': """L'instabilità baroclina lungo il fronte polare sta convertendo
                energia potenziale disponibile in energia cinetica. Il jet stream si posiziona
                a 55°N con massimi di 120kt, favorendo ciclogenesi sul Nord Atlantico.
                L'avvezione di vorticità positiva in quota precede il sistema.""",
                'uncertainty': """GFS anticipa il passaggio di 6-12h rispetto a ECMWF.
                ICON mostra una soluzione più meridionale. La divergenza principale riguarda
                il timing del fronte e l'intensità del minimo al suolo.""",
            },
            'GPT-4': {
                'confidence': 78,
                'summary': """Configurazione tipica da maltempo atlantico con saccatura
                in approfondimento. Fronte freddo in transito nelle prossime 36-48h.""",
                'physics': """L'avvezione calda in quota (settore caldo) precede il fronte
                freddo. La convergenza nei bassi strati alimenta i moti verticali.
                Spessori 500/1000 in diminuzione indicano ingresso aria fredda.""",
                'uncertainty': """Noto divergenza tra i modelli sull'intensità del minimo
                al suolo (ECMWF: 995 hPa, GFS: 990 hPa). Tempistica incerta ±12h.""",
            },
        }
        ai_analyses = demo_analyses

    for ai_name, analysis in ai_analyses.items():
        if isinstance(analysis, dict):
            confidence = analysis.get('confidence', 75)
            summary = analysis.get('summary', 'Analisi non disponibile')
            physics = analysis.get('physics', '')
            uncertainty = analysis.get('uncertainty', '')
        else:
            confidence = getattr(analysis, 'confidence_score', 75)
            summary = getattr(analysis, 'synoptic_summary', 'Analisi non disponibile')
            physics = getattr(analysis, 'physical_interpretation', '')
            uncertainty = getattr(analysis, 'confidence_assessment', '')

        # Confidence color
        if confidence >= 80:
            conf_color = "🟢"
        elif confidence >= 60:
            conf_color = "🟡"
        else:
            conf_color = "🔴"

        with st.expander(f"**{ai_name}** {conf_color} Confidence: {confidence}/100", expanded=True):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**📋 Sintesi Sinottica:**")
                st.write(summary)

                st.markdown("**🔬 Interpretazione Fisica:**")
                st.write(physics)

            with col2:
                st.markdown("**⚠️ Incertezze e Divergenze:**")
                st.write(uncertainty)

                # Confidence gauge
                st.markdown("**📊 Livello Confidenza:**")
                st.progress(confidence / 100)


# ============================================================================
# ENSEMBLE VERDICT
# ============================================================================

def render_ensemble_verdict(ensemble_results, hour: int):
    """Render final ensemble verdict."""

    st.subheader("🎯 Verdetto Ensemble")

    if ensemble_results is None or hour not in ensemble_results:
        # Demo verdict
        confidence = 79
        verdict = """I modelli concordano sulla struttura generale ma divergono sul timing.
        ECMWF e ICON mostrano la soluzione più probabile (65% probabilità).
        Scenario alternativo (35%): GFS con passaggio anticipato di 6-12h e intensità maggiore."""
    else:
        result = ensemble_results[hour]
        confidence = result.final_confidence_score
        verdict = result.synoptic_summary

    # Confidence score display
    col1, col2 = st.columns([1, 3])

    with col1:
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.0f}/100",
            delta=None,
        )

    with col2:
        st.write(verdict)

        if ensemble_results and hour in ensemble_results:
            result = ensemble_results[hour]
            if result.warnings:
                st.warning("⚠️ " + " | ".join(result.warnings))


# ============================================================================
# TIME SLIDER
# ============================================================================

def render_time_slider(forecast_hours: list[int]) -> int:
    """Render time navigation slider."""

    st.subheader("⏱️ Navigazione Temporale")

    col1, col2, col3, col4 = st.columns([1, 3, 1, 1])

    with col1:
        if st.button("◀ -6h"):
            idx = forecast_hours.index(st.session_state.current_hour)
            if idx > 0:
                st.session_state.current_hour = forecast_hours[idx - 1]

    with col2:
        hour = st.select_slider(
            "Ora previsione",
            options=forecast_hours,
            value=st.session_state.current_hour,
            format_func=lambda x: f"+{x}h",
            label_visibility="collapsed",
        )
        st.session_state.current_hour = hour

    with col3:
        if st.button("+6h ▶"):
            idx = forecast_hours.index(st.session_state.current_hour)
            if idx < len(forecast_hours) - 1:
                st.session_state.current_hour = forecast_hours[idx + 1]

    with col4:
        st.write(f"**+{st.session_state.current_hour}h**")

    return st.session_state.current_hour


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit app."""

    init_session_state()

    # Header
    st.title("🌍 HAEM - Hybrid AI Ensemble Model")
    st.markdown("*Sistema di analisi meteorologica multi-modello con interpretazione AI*")
    st.markdown("---")

    # Sidebar
    config = render_sidebar()

    # Auto-start on first load (no need to click button)
    first_load = st.session_state.last_update is None and not st.session_state.analysis_complete

    # Check for auto-refresh based on synoptic schedule
    should_refresh = False
    if config['auto_refresh'] and st.session_state.last_update:
        from datetime import timedelta

        now_utc = datetime.utcnow()
        last_update = st.session_state.last_update

        # Synoptic runs: 00, 06, 12, 18 UTC - available ~4h after
        synoptic_availability = [4, 10, 16, 22]  # Hours when data is available

        # Check if we've passed a new synoptic availability window since last update
        for avail_hour in synoptic_availability:
            # Create datetime for today's availability window
            today_avail = now_utc.replace(hour=avail_hour, minute=0, second=0, microsecond=0)
            if now_utc.hour < avail_hour:
                # This window is in the future today
                continue

            # Check if this window is after our last update
            if last_update < today_avail <= now_utc:
                should_refresh = True
                synoptic_run = (avail_hour - 4) % 24
                st.info(f"🔄 Nuova uscita sinottica {synoptic_run:02d}Z disponibile! Aggiornamento in corso...")

    # Get current synoptic run identifier for cache key
    current_run = get_current_synoptic_run()

    # Determine if we should fetch data
    # - On first load, try to use Streamlit's cache (automatic)
    # - Manual button press forces a refresh
    # - Auto-refresh when new synoptic data is available
    force_refresh = config['run_analysis'] or should_refresh

    # Main content area - fetch data (uses Streamlit cache automatically)
    if first_load or force_refresh:
        try:
            # Convert to tuples for caching (lists are not hashable)
            models_tuple = tuple(config['models'])
            hours_tuple = tuple(config['forecast_hours'])

            # Use the synoptic run as cache key - this invalidates cache when new data is available
            cache_key = current_run if not force_refresh else f"{current_run}_{datetime.utcnow().timestamp()}"

            with st.spinner("🔄 Caricamento dati meteorologici..."):
                # This uses Streamlit's @st.cache_data - returns cached data if available
                model_data = fetch_weather_data_cached(models_tuple, hours_tuple, cache_key)
                st.session_state.model_data = model_data

                # Compute ensemble
                ensemble_results = {}
                for hour in config['forecast_hours']:
                    ensemble_results[hour] = compute_ensemble(model_data, hour)
                st.session_state.ensemble_results = ensemble_results

                # Also save to file cache as backup
                dashboard_cache.save(model_data, ensemble_results)

                # Run AI analysis if providers are configured
                if config['ai_configs']:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        ai_analyses = loop.run_until_complete(
                            run_ai_analysis(model_data, config['ai_configs'], st.session_state.current_hour)
                        )
                        st.session_state.ai_analyses = ai_analyses
                    finally:
                        loop.close()

                st.session_state.analysis_complete = True
                st.session_state.last_update = datetime.utcnow()

                if force_refresh:
                    st.success(f"✅ Analisi completata! Dati da {len(model_data)} modelli.")
                else:
                    st.success(f"✅ Dati caricati (run sinottica: {current_run})")

        except Exception as e:
            st.error(f"❌ Errore durante l'analisi: {str(e)}")
            logger.exception("Analysis failed")

    # Time slider
    current_hour = render_time_slider(config['forecast_hours'])

    # Field selector - organized in categories
    st.markdown("**Seleziona Campo:**")

    # Row 1: Synoptic fields (upper air)
    st.caption("Quota")
    cols_synop = st.columns(6)
    synop_fields = [
        ('z500', 'Z500'),
        ('z850', 'Z850'),
        ('t850', 'T850'),
        ('t500', 'T500'),
        ('wind_300', 'Jet'),
        ('spread', 'Spread'),
    ]
    for col, (field_id, label) in zip(cols_synop, synop_fields):
        with col:
            if st.button(label, use_container_width=True,
                         type="primary" if st.session_state.selected_field == field_id else "secondary",
                         key=f"btn_{field_id}"):
                st.session_state.selected_field = field_id

    # Row 2: Surface and precipitation
    st.caption("Superficie e Precipitazioni")
    cols_surf = st.columns(6)
    surf_fields = [
        ('t2m', 'T2m'),
        ('slp', 'SLP'),
        ('wind_10m', 'Vento'),
        ('precip', 'Pioggia'),
        ('snow', 'Neve'),
        ('cape', 'CAPE'),
    ]
    for col, (field_id, label) in zip(cols_surf, surf_fields):
        with col:
            if st.button(label, use_container_width=True,
                         type="primary" if st.session_state.selected_field == field_id else "secondary",
                         key=f"btn_{field_id}"):
                st.session_state.selected_field = field_id

    st.markdown("---")

    # Main layout: Map + Model comparison
    col_map, col_sidebar = st.columns([2, 1])

    with col_map:
        render_map_panel(
            st.session_state.ensemble_results,
            current_hour,
            st.session_state.selected_field
        )

    with col_sidebar:
        render_model_comparison(st.session_state.ensemble_results, current_hour)

    st.markdown("---")

    # AI Analysis
    render_ai_analysis(st.session_state.ai_analyses, current_hour)

    st.markdown("---")

    # Ensemble Verdict
    render_ensemble_verdict(st.session_state.ensemble_results, current_hour)

    # Footer
    st.markdown("---")
    last_update_str = st.session_state.last_update.strftime('%Y-%m-%d %H:%MZ') if st.session_state.last_update else "Mai"
    st.markdown(
        f"*HAEM v1.0 | Dati: Open-Meteo API (ECMWF, GFS, ICON, GEM, ARPEGE) | "
        f"Ultimo aggiornamento: {last_update_str}*"
    )

    # Auto-refresh mechanism synced with synoptic runs
    if config['auto_refresh']:
        import time as time_module
        from datetime import timedelta

        now_utc = datetime.utcnow()

        # Calculate time until next synoptic data availability (04, 10, 16, 22 UTC)
        synoptic_availability = [4, 10, 16, 22]
        current_hour = now_utc.hour
        current_minute = now_utc.minute

        # Find next availability window
        next_avail_hour = None
        for avail_hour in synoptic_availability:
            if current_hour < avail_hour:
                next_avail_hour = avail_hour
                break
        if next_avail_hour is None:
            next_avail_hour = synoptic_availability[0]  # Tomorrow's 04 UTC

        # Calculate seconds until next availability
        if next_avail_hour > current_hour:
            hours_until = next_avail_hour - current_hour
            seconds_until = (hours_until * 3600) - (current_minute * 60)
        else:
            hours_until = (24 - current_hour) + next_avail_hour
            seconds_until = (hours_until * 3600) - (current_minute * 60)

        synoptic_run = (next_avail_hour - 4) % 24
        st.sidebar.write(f"Prossimo refresh: run **{synoptic_run:02d}Z** tra {hours_until}h {60 - current_minute}m")

        # Check every 5 minutes for new data
        time_module.sleep(min(300, seconds_until))
        st.rerun()


if __name__ == "__main__":
    main()
