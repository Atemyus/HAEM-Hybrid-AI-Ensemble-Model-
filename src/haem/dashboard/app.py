"""
HAEM Streamlit Dashboard.

Interactive web dashboard for visualizing ensemble weather analysis
with multi-AI interpretation.
"""

import asyncio
import html
import logging
import re
import warnings
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import io

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Suppress numpy warnings for std with few data points (expected behavior)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# Configure page
st.set_page_config(
    page_title="HAEM - Weather Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def inject_custom_css():
    """Inject custom CSS for modern styling."""
    st.markdown("""
    <style>
    /* ===== GLOBAL BACKGROUND ===== */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }

    /* ===== HEADER/TOP BAR ===== */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        border-bottom: 1px solid rgba(102, 126, 234, 0.2);
    }

    /* Hide Streamlit branding/menu if needed */
    #MainMenu {
        background: transparent !important;
    }

    /* Top decoration bar */
    .stDeployButton, [data-testid="stToolbar"] {
        background: transparent !important;
    }

    /* Animated background particles effect */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image:
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(74, 144, 226, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(147, 112, 219, 0.1) 0%, transparent 40%);
        pointer-events: none;
        z-index: 0;
    }

    /* ===== GLOBAL TEXT COLOR FIX ===== */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label {
        color: #e0e0e0 !important;
    }

    /* All text elements */
    p, span, div, label, .stMarkdown, .stText {
        color: #e0e0e0 !important;
    }

    /* Streamlit specific text elements */
    .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #e0e0e0 !important;
    }

    /* Caption text */
    .stCaption, small, .caption {
        color: #a0a0a0 !important;
    }

    /* ===== MAIN CONTENT STYLING ===== */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ===== HEADER STYLING ===== */
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }

    h2, h3, h4, h5, h6 {
        color: #e0e0e0 !important;
    }

    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.3);
    }

    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0 !important;
    }

    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    /* Sidebar subheader */
    [data-testid="stSidebar"] .stSubheader {
        color: #ffffff !important;
    }

    /* ===== INPUT ELEMENTS ===== */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    /* Selectbox dropdown text */
    .stSelectbox label, .stMultiSelect label {
        color: #e0e0e0 !important;
    }

    /* ===== AI CARD STYLES ===== */
    .ai-card {
        background: linear-gradient(145deg, rgba(30, 30, 50, 0.9), rgba(20, 20, 40, 0.95));
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .ai-card:hover {
        transform: translateY(-4px);
        box-shadow:
            0 12px 40px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }

    /* Claude Card */
    .ai-card-claude {
        border-left: 4px solid #d97706;
        background: linear-gradient(145deg, rgba(217, 119, 6, 0.1), rgba(30, 30, 50, 0.95));
    }
    .ai-card-claude .ai-header { color: #fbbf24 !important; }

    /* GPT Card */
    .ai-card-gpt {
        border-left: 4px solid #10b981;
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(30, 30, 50, 0.95));
    }
    .ai-card-gpt .ai-header { color: #34d399 !important; }

    /* Gemini Card */
    .ai-card-gemini {
        border-left: 4px solid #3b82f6;
        background: linear-gradient(145deg, rgba(59, 130, 246, 0.1), rgba(30, 30, 50, 0.95));
    }
    .ai-card-gemini .ai-header { color: #60a5fa !important; }

    /* Qwen Card */
    .ai-card-qwen {
        border-left: 4px solid #8b5cf6;
        background: linear-gradient(145deg, rgba(139, 92, 246, 0.1), rgba(30, 30, 50, 0.95));
    }
    .ai-card-qwen .ai-header { color: #a78bfa !important; }

    /* Deepseek Card */
    .ai-card-deepseek {
        border-left: 4px solid #06b6d4;
        background: linear-gradient(145deg, rgba(6, 182, 212, 0.1), rgba(30, 30, 50, 0.95));
    }
    .ai-card-deepseek .ai-header { color: #22d3ee !important; }

    /* GLM Card */
    .ai-card-glm {
        border-left: 4px solid #f43f5e;
        background: linear-gradient(145deg, rgba(244, 63, 94, 0.1), rgba(30, 30, 50, 0.95));
    }
    .ai-card-glm .ai-header { color: #fb7185 !important; }

    /* Grok Card */
    .ai-card-grok {
        border-left: 4px solid #eab308;
        background: linear-gradient(145deg, rgba(234, 179, 8, 0.1), rgba(30, 30, 50, 0.95));
    }
    .ai-card-grok .ai-header { color: #facc15 !important; }

    /* AI Card Inner Elements */
    .ai-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #ffffff !important;
    }

    .ai-confidence {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .confidence-high {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1));
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .confidence-medium {
        background: linear-gradient(90deg, rgba(234, 179, 8, 0.2), rgba(234, 179, 8, 0.1));
        color: #fbbf24 !important;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    .confidence-low {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1));
        color: #f87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .ai-section {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.75rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .ai-section-title {
        color: #a0a0a0 !important;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }

    .ai-section-content {
        color: #e0e0e0 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* ===== VERDICT CARD ===== */
    .verdict-card {
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.1));
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 2px solid rgba(102, 126, 234, 0.3);
        box-shadow:
            0 10px 40px rgba(102, 126, 234, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    .verdict-title {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    /* ===== METRICS CARDS ===== */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #e0e0e0 !important;
    }

    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 0 0 8px 8px;
    }

    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    }

    /* ===== INFO/WARNING BOXES ===== */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border-left: 4px solid;
    }

    /* ===== SLIDER ===== */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    /* ===== SELECTBOX ===== */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }

    /* ===== CHECKBOX ===== */
    .stCheckbox > label {
        color: #e0e0e0 !important;
    }

    /* ===== DIVIDER ===== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);
        margin: 1.5rem 0;
    }

    /* ===== LOADING SPINNER ===== */
    .stSpinner > div {
        border-color: #667eea !important;
    }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2, #f093fb);
    }
    </style>
    """, unsafe_allow_html=True)


# AI Provider color themes
AI_THEMES = {
    'Claude': {'class': 'claude', 'icon': '🟠', 'color': '#d97706'},
    'Claude 4.5 Opus': {'class': 'claude', 'icon': '🟠', 'color': '#d97706'},
    'GPT-4': {'class': 'gpt', 'icon': '🟢', 'color': '#10b981'},
    'GPT-5 Pro': {'class': 'gpt', 'icon': '🟢', 'color': '#10b981'},
    'Gemini': {'class': 'gemini', 'icon': '🔵', 'color': '#3b82f6'},
    'Gemini 3 Pro': {'class': 'gemini', 'icon': '🔵', 'color': '#3b82f6'},
    'Qwen Max': {'class': 'qwen', 'icon': '🟣', 'color': '#8b5cf6'},
    'Deepseek V3.2': {'class': 'deepseek', 'icon': '🔷', 'color': '#06b6d4'},
    'GLM 4.7': {'class': 'glm', 'icon': '🔴', 'color': '#f43f5e'},
    'Grok 4.1 Fast': {'class': 'grok', 'icon': '🟡', 'color': '#eab308'},
}


def get_ai_theme(ai_name: str) -> dict:
    """Get the theme for an AI provider."""
    for key, theme in AI_THEMES.items():
        if key.lower() in ai_name.lower():
            return theme
    return {'class': 'claude', 'icon': '🤖', 'color': '#667eea'}


# Import HAEM modules
from haem.models.meteorological import NWPModel
from haem.data.meteociel_fields import FieldSelection, FieldPresets, MeteocielField
from haem.ai.providers import AIProvider, AIProviderConfig, AIPresets, MultiAIOrchestrator
from haem.config.settings import get_api_keys, get_app_config
from haem.data.openmeteo import MultiModelOpenMeteoFetcher, OpenMeteoConfig
from haem.data.cache import DashboardCache

logger = logging.getLogger(__name__)


# ============================================================================
# ENSEMBLE DATA CLASSES (module level for pickle compatibility)
# ============================================================================

@dataclass
class EnsembleField:
    """Represents ensemble statistics for a single meteorological field."""
    ensemble_mean: np.ndarray
    ensemble_spread: np.ndarray
    lats: np.ndarray
    lons: np.ndarray


@dataclass
class EnsembleResult:
    """Complete ensemble result for a forecast hour."""
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


async def run_ai_analysis(model_data: dict, ai_configs: list[AIProviderConfig], forecast_hour: int, selected_field: str = "z500", preset: str = "Analisi Sinottica"):
    """Run AI analysis on the weather data for a specific field, hour, and preset context."""
    if not ai_configs:
        return {}

    orchestrator = MultiAIOrchestrator(ai_configs)

    # Field descriptions for AI context
    field_descriptions = {
        'z500': 'Geopotenziale a 500 hPa - indica la posizione di saccature e promontori',
        'z850': 'Geopotenziale a 850 hPa - indica flussi nei bassi strati',
        't850': 'Temperatura a 850 hPa - indica masse d\'aria e fronti',
        't500': 'Temperatura a 500 hPa - indica instabilità e avvezione termica',
        'slp': 'Pressione al livello del mare - indica centri di alta e bassa pressione',
        't2m': 'Temperatura a 2 metri - indica condizioni al suolo',
        'precip': 'Precipitazioni - indica pioggia e temporali',
        'snow': 'Neve - indica accumuli nevosi',
        'wind_10m': 'Vento a 10 metri - indica condizioni ventose al suolo',
        'wind_300': 'Jet stream a 300 hPa - indica correnti a getto',
        'cape': 'CAPE - indica energia disponibile per convezione',
        'spread': 'Spread ensemble - indica incertezza tra i modelli',
    }

    # Preset descriptions for AI context
    preset_descriptions = {
        'Analisi Sinottica': 'Analisi sinottica: focus su pattern di larga scala, saccature, promontori, fronti e circolazione generale',
        'Analisi Convettiva': 'Analisi convettiva: focus su temporali, CAPE, wind shear, lifting mechanisms e severe weather',
        'Meteo Invernale': 'Meteo invernale: focus su neve, quota neve, isoterme, inversioni termiche e precipitazioni solide',
        'Completo': 'Analisi completa: valutazione integrata di tutti i parametri meteorologici',
    }
    preset_context = preset_descriptions.get(preset, preset)

    # Prepare comprehensive data summary for AI
    data_summary = {}
    for model, data in model_data.items():
        model_stats = {}

        # Always include basic synoptic fields
        if forecast_hour in data.z500:
            z500 = data.z500[forecast_hour]
            model_stats["z500_mean"] = float(np.nanmean(z500.data))
            model_stats["z500_min"] = float(np.nanmin(z500.data))
            model_stats["z500_max"] = float(np.nanmax(z500.data))

        if forecast_hour in data.t850:
            t850 = data.t850[forecast_hour]
            model_stats["t850_mean_C"] = float(np.nanmean(t850.data) - 273.15)
            model_stats["t850_min_C"] = float(np.nanmin(t850.data) - 273.15)
            model_stats["t850_max_C"] = float(np.nanmax(t850.data) - 273.15)

        if forecast_hour in data.slp:
            slp = data.slp[forecast_hour]
            model_stats["slp_min_hPa"] = float(np.nanmin(slp.data))
            model_stats["slp_max_hPa"] = float(np.nanmax(slp.data))

        # Add field-specific data based on selection
        if selected_field == 't2m' and hasattr(data, 't2m') and forecast_hour in data.t2m:
            t2m = data.t2m[forecast_hour]
            model_stats["t2m_mean_C"] = float(np.nanmean(t2m.data))
            model_stats["t2m_min_C"] = float(np.nanmin(t2m.data))
            model_stats["t2m_max_C"] = float(np.nanmax(t2m.data))

        if selected_field == 'precip' and hasattr(data, 'precip') and forecast_hour in data.precip:
            precip = data.precip[forecast_hour]
            model_stats["precip_max_mm"] = float(np.nanmax(precip.data))
            model_stats["precip_mean_mm"] = float(np.nanmean(precip.data))

        if selected_field == 'cape' and hasattr(data, 'cape') and forecast_hour in data.cape:
            cape = data.cape[forecast_hour]
            model_stats["cape_max_Jkg"] = float(np.nanmax(cape.data))
            model_stats["cape_mean_Jkg"] = float(np.nanmean(cape.data))

        if model_stats:
            data_summary[model.value] = model_stats

    # Build field selection list based on current field
    field_list = [selected_field.upper()]
    if selected_field not in ['z500', 't850', 'slp']:
        field_list.extend(['Z500', 'T850', 'SLP'])  # Always include basic synoptic

    results = await orchestrator.analyze_all(
        model_data=data_summary,
        field_selection=field_list,
        forecast_hour=forecast_hour,
        selected_field=selected_field,
        field_description=field_descriptions.get(selected_field, selected_field),
        preset_context=preset_context,
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
    # Uses module-level EnsembleField and EnsembleResult classes

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

# Cache version - increment this to invalidate old cached data
AI_CACHE_VERSION = 5


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
    # Cache for AI analyses keyed by (hour, field, preset)
    if 'ai_analyses_cache' not in st.session_state:
        st.session_state.ai_analyses_cache = {}
    # Cache version check - clear cache if version mismatch
    if 'ai_cache_version' not in st.session_state or st.session_state.ai_cache_version != AI_CACHE_VERSION:
        st.session_state.ai_analyses_cache = {}
        st.session_state.ai_analyses = {}
        st.session_state.ai_cache_version = AI_CACHE_VERSION
    if 'current_hour' not in st.session_state:
        st.session_state.current_hour = 48
    if 'selected_field' not in st.session_state:
        st.session_state.selected_field = 'z500'
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'last_ai_preset' not in st.session_state:
        st.session_state.last_ai_preset = None


def get_ai_cache_key(hour: int, field: str, preset: str) -> str:
    """Generate a cache key for AI analyses."""
    return f"{hour}_{field}_{preset}"


def get_cached_ai_analysis(hour: int, field: str, preset: str) -> dict:
    """Get cached AI analysis if available."""
    cache_key = get_ai_cache_key(hour, field, preset)
    return st.session_state.ai_analyses_cache.get(cache_key)


def cache_ai_analysis(hour: int, field: str, preset: str, analyses: dict):
    """Cache AI analysis for a specific hour/field/preset combination."""
    cache_key = get_ai_cache_key(hour, field, preset)
    st.session_state.ai_analyses_cache[cache_key] = analyses


def clear_ai_cache():
    """Clear all cached AI analyses (called when new data is fetched)."""
    st.session_state.ai_analyses_cache = {}


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
        if api_keys.has_aiml_api:
            st.success("🔑 AIML API: Configurato")
            st.info("Tutti i provider AI disponibili!")
        else:
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
                st.caption("💡 Configura AIML_API_KEY per accedere a tutti i modelli")

    # Show available providers based on AIML API or individual keys
    if api_keys.has_aiml_api:
        # All providers available via AIML API
        st.sidebar.caption("Seleziona i modelli AI da usare:")

        if st.sidebar.checkbox("Claude 4.5 Opus", value=True, key="ai_claude"):
            ai_configs.append(AIProviderConfig(provider=AIProvider.CLAUDE))
        if st.sidebar.checkbox("GPT-5 Pro", value=True, key="ai_gpt5"):
            ai_configs.append(AIProviderConfig(provider=AIProvider.GPT5))
        if st.sidebar.checkbox("Gemini 3 Pro", value=True, key="ai_gemini"):
            ai_configs.append(AIProviderConfig(provider=AIProvider.GEMINI_PRO))
        if st.sidebar.checkbox("Qwen Max", value=True, key="ai_qwen"):
            ai_configs.append(AIProviderConfig(provider=AIProvider.QWEN))
        if st.sidebar.checkbox("Deepseek V3.2", value=True, key="ai_deepseek"):
            ai_configs.append(AIProviderConfig(provider=AIProvider.DEEPSEEK))
        if st.sidebar.checkbox("GLM 4.7", value=False, key="ai_glm"):
            ai_configs.append(AIProviderConfig(provider=AIProvider.GLM))
        if st.sidebar.checkbox("Grok 4.1 Fast", value=False, key="ai_grok"):
            ai_configs.append(AIProviderConfig(provider=AIProvider.GROK))
    else:
        # Only show individually configured providers
        if api_keys.has_claude:
            if st.sidebar.checkbox("Claude (Anthropic)", value=True, key="ai_claude"):
                ai_configs.append(AIProviderConfig(provider=AIProvider.CLAUDE))
        if api_keys.has_gpt4:
            if st.sidebar.checkbox("GPT-4 (OpenAI)", value=True, key="ai_gpt4"):
                ai_configs.append(AIProviderConfig(provider=AIProvider.GPT4))
        if api_keys.has_gemini:
            if st.sidebar.checkbox("Gemini (Google)", value=True, key="ai_gemini"):
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
    run_analysis = st.sidebar.button("🔄 AGGIORNA DATI METEO", use_container_width=True)
    st.sidebar.caption("⚠️ Scarica nuovi dati e resetta la cache AI")
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Cambiare campo/ora/preset usa automaticamente la cache se disponibile")

    return {
        'models': selected_models,
        'field_selection': field_selection,
        'field_preset': field_preset,  # Added for AI re-analysis trigger
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

def render_ai_card(ai_name: str, analysis: dict):
    """Render a styled AI analysis card using Streamlit components."""
    theme = get_ai_theme(ai_name)

    if isinstance(analysis, dict):
        confidence = analysis.get('confidence', 75)
        summary = analysis.get('summary', 'Analisi non disponibile')
        physics = analysis.get('physics', '')
        uncertainty = analysis.get('uncertainty', '')
        inference_time = analysis.get('inference_time_ms', 0)
        patterns = analysis.get('patterns', [])
        findings = analysis.get('findings', [])
        alert_warnings = analysis.get('warnings', [])
    else:
        confidence = getattr(analysis, 'confidence_score', 75)
        summary = getattr(analysis, 'synoptic_summary', 'Analisi non disponibile')
        physics = getattr(analysis, 'physical_interpretation', '')
        uncertainty = getattr(analysis, 'confidence_assessment', '')
        inference_time = getattr(analysis, 'inference_time_ms', 0)
        patterns = getattr(analysis, 'pattern_identification', [])
        findings = getattr(analysis, 'key_findings', [])
        alert_warnings = getattr(analysis, 'warnings', [])

    # Check if this is an error response
    is_error = isinstance(summary, str) and ('Errore:' in summary or 'Error' in summary or 'credit balance' in summary.lower())

    # Confidence styling
    if confidence >= 80:
        conf_color = "#10b981"
        conf_emoji = "🟢"
    elif confidence >= 60:
        conf_color = "#eab308"
        conf_emoji = "🟡"
    else:
        conf_color = "#ef4444"
        conf_emoji = "🔴"

    # Use container for the card
    with st.container():
        # Header
        time_str = f" • {inference_time:.0f}ms" if inference_time > 0 else ""
        st.markdown(f"""
<div style="background: linear-gradient(145deg, rgba(30, 30, 50, 0.9), rgba(20, 20, 40, 0.95)); border-left: 4px solid {theme['color']}; border-radius: 16px; padding: 1.5rem; margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
<div style="display: flex; align-items: center; gap: 0.5rem;">
<span style="font-size: 1.5rem;">{theme['icon']}</span>
<span style="font-size: 1.3rem; font-weight: 700; color: {theme['color']};">{ai_name}</span>
</div>
<div style="display: flex; gap: 0.5rem; align-items: center;">
<span style="background: {conf_color}22; color: {conf_color}; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 600; border: 1px solid {conf_color}44;">{conf_emoji} {confidence:.0f}/100</span>
<span style="color: #808080; font-size: 0.8rem;">{time_str}</span>
</div>
</div>
""", unsafe_allow_html=True)

        if is_error:
            # Show error in a styled box
            st.markdown(f"""
<div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;">
<div style="color: #f87171; font-weight: 600; margin-bottom: 0.5rem;">⚠️ Errore API</div>
<div style="color: #fca5a5; font-size: 0.9rem;">Crediti API insufficienti o errore di connessione. Verifica il tuo account.</div>
</div>
""", unsafe_allow_html=True)
        else:
            # Synoptic Summary
            safe_summary = html.escape(str(summary)) if summary else "Non disponibile"
            st.markdown(f"""
<div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem; border: 1px solid rgba(255,255,255,0.05);">
<div style="color: #a0a0a0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">📋 SINTESI SINOTTICA</div>
<div style="color: #e0e0e0; line-height: 1.6;">{safe_summary}</div>
</div>
""", unsafe_allow_html=True)

            # Physics and Uncertainty side by side
            safe_physics = html.escape(str(physics)) if physics else "<span style='color:#606060;'>Non disponibile</span>"
            safe_uncertainty = html.escape(str(uncertainty)) if uncertainty else "<span style='color:#606060;'>Non disponibile</span>"

            st.markdown(f"""
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
<div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 1rem; border: 1px solid rgba(255,255,255,0.05);">
<div style="color: #a0a0a0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">🔬 INTERPRETAZIONE FISICA</div>
<div style="color: #e0e0e0; line-height: 1.6;">{safe_physics}</div>
</div>
<div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 1rem; border: 1px solid rgba(255,255,255,0.05);">
<div style="color: #a0a0a0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">⚠️ INCERTEZZE E DIVERGENZE</div>
<div style="color: #e0e0e0; line-height: 1.6;">{safe_uncertainty}</div>
</div>
</div>
""", unsafe_allow_html=True)

            # Patterns and Findings
            if (patterns and len(patterns) > 0) or (findings and len(findings) > 0):
                patterns_items = ""
                if patterns and len(patterns) > 0:
                    for p in patterns[:4]:
                        patterns_items += f"<li style='color: #c0c0c0; margin: 0.3rem 0;'>{html.escape(str(p))}</li>"
                    patterns_list = f"<ul style='margin: 0; padding-left: 1.2rem;'>{patterns_items}</ul>"
                else:
                    patterns_list = "<span style='color:#606060;'>Nessun pattern</span>"

                findings_items = ""
                if findings and len(findings) > 0:
                    for f in findings[:4]:
                        findings_items += f"<li style='color: #c0c0c0; margin: 0.3rem 0;'>{html.escape(str(f))}</li>"
                    findings_list = f"<ul style='margin: 0; padding-left: 1.2rem;'>{findings_items}</ul>"
                else:
                    findings_list = "<span style='color:#606060;'>In elaborazione...</span>"

                st.markdown(f"""
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.75rem;">
<div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 1rem; border: 1px solid rgba(255,255,255,0.05);">
<div style="color: #a0a0a0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">🎯 PATTERN IDENTIFICATI</div>
<div style="color: #e0e0e0;">{patterns_list}</div>
</div>
<div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 1rem; border: 1px solid rgba(255,255,255,0.05);">
<div style="color: #a0a0a0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">📌 CONCLUSIONI CHIAVE</div>
<div style="color: #e0e0e0;">{findings_list}</div>
</div>
</div>
""", unsafe_allow_html=True)

        # Warnings
        if alert_warnings and len(alert_warnings) > 0:
            warnings_items = ""
            for w in alert_warnings[:3]:
                warnings_items += f"<div style='color: #fca5a5; font-size: 0.9rem;'>• {html.escape(str(w))}</div>"
            st.markdown(f"""
<div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 0.75rem; margin-top: 0.75rem;">
<div style="color: #f87171; font-weight: 600; margin-bottom: 0.5rem;">⚠️ Avvisi</div>
{warnings_items}
</div>
""", unsafe_allow_html=True)

        # Confidence bar and close card
        st.markdown(f"""
<div style="margin-top: 1rem;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
<span style="color: #808080; font-size: 0.85rem;">Livello di Confidenza</span>
<span style="color: {theme['color']}; font-weight: 600;">{confidence:.0f}%</span>
</div>
<div style="background: rgba(255,255,255,0.1); border-radius: 10px; height: 8px; overflow: hidden;">
<div style="background: linear-gradient(90deg, {theme['color']}, {theme['color']}88); height: 100%; width: {confidence}%; border-radius: 10px;"></div>
</div>
</div>
</div>
""", unsafe_allow_html=True)


def render_ai_analysis(ai_analyses: dict, hour: int):
    """Render AI analysis panel with styled cards."""

    st.markdown("""
    <h2 style="color: #e0e0e0; display: flex; align-items: center; gap: 0.5rem;">
        <span style="font-size: 1.5rem;">🤖</span>
        Analisi AI - Ragionamenti Multi-Modello
    </h2>
    """, unsafe_allow_html=True)

    if not ai_analyses:
        # Demo analysis with more data
        demo_analyses = {
            'Claude 4.5 Opus': {
                'confidence': 85,
                'summary': """Una saccatura atlantica si approfondisce verso l'Europa occidentale,
                con geopotenziali in calo di 8 dam nelle prossime 48h. Il pattern è guidato
                dalla propagazione di onde di Rossby con numero d'onda 4-5.""",
                'physics': """L'instabilità baroclina lungo il fronte polare sta convertendo
                energia potenziale disponibile in energia cinetica. Il jet stream si posiziona
                a 55°N con massimi di 120kt, favorendo ciclogenesi sul Nord Atlantico.""",
                'uncertainty': """GFS anticipa il passaggio di 6-12h rispetto a ECMWF.
                ICON mostra una soluzione più meridionale.""",
                'patterns': ['Saccatura atlantica in approfondimento', 'Jet stream a 55°N', 'Ciclogenesi attiva'],
                'findings': ['Peggioramento in arrivo', 'Timing incerto ±12h'],
                'warnings': [],
                'inference_time_ms': 1250,
            },
            'GPT-5 Pro': {
                'confidence': 78,
                'summary': """Configurazione tipica da maltempo atlantico con saccatura
                in approfondimento. Fronte freddo in transito nelle prossime 36-48h.""",
                'physics': """L'avvezione calda in quota precede il fronte freddo.
                La convergenza nei bassi strati alimenta i moti verticali.""",
                'uncertainty': """Divergenza tra i modelli sull'intensità del minimo
                al suolo (ECMWF: 995 hPa, GFS: 990 hPa).""",
                'patterns': ['Avvezione calda prefrontale', 'Convergenza nei bassi strati'],
                'findings': ['Precipitazioni moderate attese', 'Rinforzo ventilazione'],
                'warnings': ['Possibili fenomeni intensi sul Tirreno'],
                'inference_time_ms': 980,
            },
        }
        ai_analyses = demo_analyses

    # Render each AI card
    for ai_name, analysis in ai_analyses.items():
        render_ai_card(ai_name, analysis)


# ============================================================================
# ENSEMBLE VERDICT
# ============================================================================

def render_ensemble_verdict(ensemble_results, hour: int, ai_analyses: dict = None):
    """Render final ensemble verdict based on highest confidence AI."""

    # Find the best AI based on confidence score
    best_ai = None
    best_confidence = 0
    best_summary = ""
    best_uncertainty = ""
    best_physics = ""
    best_theme = None

    if ai_analyses:
        for ai_name, analysis in ai_analyses.items():
            if isinstance(analysis, dict):
                conf = analysis.get('confidence', 0)
                # Skip error responses
                summary = analysis.get('summary', '')
                if 'Errore:' in str(summary) or 'Error' in str(summary):
                    continue
                if conf > best_confidence:
                    best_confidence = conf
                    best_ai = ai_name
                    best_summary = summary
                    best_uncertainty = analysis.get('uncertainty', '')
                    best_physics = analysis.get('physics', '')
                    best_theme = get_ai_theme(ai_name)

    if best_ai:
        confidence = best_confidence
        verdict = best_summary
        uncertainty = best_uncertainty
        physics = best_physics
        source = f"Analisi piu accreditata: {best_ai} ({confidence:.0f}% confidenza)"
    elif ensemble_results is not None and hour in ensemble_results:
        result = ensemble_results[hour]
        confidence = result.final_confidence_score
        verdict = result.synoptic_summary if result.synoptic_summary else "Analisi ensemble dei modelli NWP completata."
        uncertainty = ""
        physics = ""
        source = "Basato sui modelli NWP"
        best_theme = {'icon': '🌐', 'color': '#667eea'}
    else:
        confidence = 79
        verdict = "I modelli concordano sulla struttura generale ma divergono sul timing. ECMWF e ICON mostrano la soluzione più probabile (65% probabilità). Scenario alternativo (35%): GFS con passaggio anticipato di 6-12h e intensità maggiore."
        uncertainty = ""
        physics = ""
        source = "Modalità demo"
        best_theme = {'icon': '🎯', 'color': '#667eea'}

    # Safe escape content
    verdict_content = html.escape(str(verdict)) if verdict else "Non disponibile"
    uncertainty_content = html.escape(str(uncertainty)) if uncertainty else ""
    physics_content = html.escape(str(physics)) if physics else ""

    # Confidence styling
    if confidence >= 80:
        conf_color = "#10b981"
        conf_label = "ALTA"
        conf_emoji = "🟢"
        gradient = "linear-gradient(90deg, #10b981, #059669)"
    elif confidence >= 60:
        conf_color = "#eab308"
        conf_label = "MEDIA"
        conf_emoji = "🟡"
        gradient = "linear-gradient(90deg, #eab308, #ca8a04)"
    else:
        conf_color = "#ef4444"
        conf_label = "BASSA"
        conf_emoji = "🔴"
        gradient = "linear-gradient(90deg, #ef4444, #dc2626)"

    # Warnings HTML
    warnings_html = ""
    if ensemble_results and hour in ensemble_results:
        result = ensemble_results[hour]
        if result.warnings:
            warnings_items = ""
            for w in result.warnings:
                warnings_items += f"<div style='color: #fca5a5;'>• {html.escape(str(w))}</div>"
            warnings_html = f"""
<div style="margin-top: 1rem; padding: 1rem; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px;">
<div style="color: #f87171; font-weight: 600; margin-bottom: 0.5rem;">⚠️ Avvertenze</div>
{warnings_items}
</div>"""

    # Build physics section if available
    physics_section = ""
    if physics_content:
        physics_section = f"""
<div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 1rem; margin-top: 1rem; border: 1px solid rgba(255,255,255,0.05);">
<div style="color: #a0a0a0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">🔬 INTERPRETAZIONE FISICA</div>
<div style="color: #e0e0e0; line-height: 1.6;">{physics_content}</div>
</div>"""

    # Build uncertainty section if available
    uncertainty_section = ""
    if uncertainty_content:
        uncertainty_section = f"""
<div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 1rem; margin-top: 1rem; border: 1px solid rgba(255,255,255,0.05);">
<div style="color: #a0a0a0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">⚠️ INCERTEZZE RESIDUE</div>
<div style="color: #e0e0e0; line-height: 1.6;">{uncertainty_content}</div>
</div>"""

    # Render verdict with pure inline styles
    st.markdown(f"""
<div style="background: linear-gradient(145deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.1)); border-radius: 20px; padding: 2rem; margin: 1.5rem 0; border: 2px solid rgba(102, 126, 234, 0.3); box-shadow: 0 10px 40px rgba(102, 126, 234, 0.2);">
<div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
<div style="font-size: 2.5rem;">{best_theme['icon'] if best_theme else '🎯'}</div>
<div>
<div style="font-size: 1.5rem; font-weight: 700; background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">Verdetto Finale Ensemble</div>
<div style="color: #a0a0a0; font-size: 0.9rem;">{html.escape(source)}</div>
</div>
</div>
<div style="display: grid; grid-template-columns: auto 1fr; gap: 2rem; align-items: start;">
<div style="text-align: center;">
<div style="width: 120px; height: 120px; border-radius: 50%; background: {gradient}; display: flex; flex-direction: column; align-items: center; justify-content: center; box-shadow: 0 0 30px {best_theme['color'] if best_theme else '#667eea'}44;">
<div style="font-size: 2rem; font-weight: 800; color: white;">{confidence:.0f}</div>
<div style="font-size: 0.7rem; color: rgba(255,255,255,0.8);">/ 100</div>
</div>
<div style="margin-top: 0.75rem;">
<span style="background: {conf_color}22; color: {conf_color}; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 600; border: 1px solid {conf_color}44;">{conf_emoji} {conf_label}</span>
</div>
</div>
<div>
<div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 1rem; border: 1px solid rgba(255,255,255,0.05);">
<div style="color: #a0a0a0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">📋 SINTESI FINALE</div>
<div style="color: #e0e0e0; line-height: 1.6; font-size: 1rem;">{verdict_content}</div>
</div>
{physics_section}
{uncertainty_section}
{warnings_html}
</div>
</div>
</div>
""", unsafe_allow_html=True)


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

    # Inject custom CSS styling
    inject_custom_css()

    # Header with styled gradient
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">
            🌍 HAEM - Hybrid AI Ensemble Model
        </h1>
        <p style="color: #a0a0a0; font-size: 1.1rem; margin: 0;">
            Sistema di analisi meteorologica multi-modello con interpretazione AI
        </p>
    </div>
    """, unsafe_allow_html=True)
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

                # Clear AI analysis cache when new data is loaded
                clear_ai_cache()

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
                            run_ai_analysis(
                                model_data,
                                config['ai_configs'],
                                st.session_state.current_hour,
                                selected_field=st.session_state.selected_field,
                                preset=config['field_preset']
                            )
                        )
                        # Cache the initial analysis
                        cache_ai_analysis(
                            st.session_state.current_hour,
                            st.session_state.selected_field,
                            config['field_preset'],
                            ai_analyses
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

    # Check if we need to update AI analysis display (hour, field, or preset changed)
    current_preset = config['field_preset']
    current_field = st.session_state.selected_field

    # Check if we have cached analysis for this combination
    cached_analysis = get_cached_ai_analysis(current_hour, current_field, current_preset)
    cache_count = len(st.session_state.ai_analyses_cache)

    if cached_analysis:
        # Use cached analysis - no API call needed
        st.session_state.ai_analyses = cached_analysis
        # Show cache indicator
        st.markdown(
            f'<div style="background: linear-gradient(135deg, #1a472a, #2d5a3d); border-radius: 8px; '
            f'padding: 8px 16px; margin-bottom: 10px; display: inline-block;">'
            f'<span style="color: #4ade80;">💾 Cache utilizzata</span> '
            f'<span style="color: #a0a0a0;">({cache_count} analisi in memoria)</span></div>',
            unsafe_allow_html=True
        )
    elif st.session_state.model_data and config['ai_configs']:
        # No cache found - need to run AI analysis
        with st.spinner(f"🤖 Generazione analisi AI per {current_field.upper()} +{current_hour}h ({current_preset})..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ai_analyses = loop.run_until_complete(
                    run_ai_analysis(
                        st.session_state.model_data,
                        config['ai_configs'],
                        current_hour,
                        selected_field=current_field,
                        preset=current_preset
                    )
                )
                # Cache the results
                cache_ai_analysis(current_hour, current_field, current_preset, ai_analyses)
                st.session_state.ai_analyses = ai_analyses
                # Show new API call indicator
                new_cache_count = len(st.session_state.ai_analyses_cache)
                st.markdown(
                    f'<div style="background: linear-gradient(135deg, #4a3a1a, #5a4a2d); border-radius: 8px; '
                    f'padding: 8px 16px; margin-bottom: 10px; display: inline-block;">'
                    f'<span style="color: #fbbf24;">🌐 Nuova analisi AI</span> '
                    f'<span style="color: #a0a0a0;">(salvata in cache - {new_cache_count} totali)</span></div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Errore nell'analisi AI: {e}")
            finally:
                loop.close()

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

    # Ensemble Verdict (now uses AI analyses for best verdict)
    render_ensemble_verdict(st.session_state.ensemble_results, current_hour, st.session_state.ai_analyses)

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
