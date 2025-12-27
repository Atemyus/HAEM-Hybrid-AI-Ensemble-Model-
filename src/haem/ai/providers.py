"""
AI Provider Configuration for HAEM.

Supports multiple AI providers for meteorological analysis via AIML API:
- Claude (Anthropic)
- GPT-4/5 (OpenAI)
- Gemini (Google)
- Qwen (Alibaba)
- Deepseek
- GLM (Zhipu)
- Grok (xAI)

Each AI performs independent analysis, then results are weighted
and combined in the ensemble.
"""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# AIML API Configuration
AIML_API_BASE_URL = "https://api.aimlapi.com/v1"


class AIProvider(str, Enum):
    """Supported AI providers."""

    CLAUDE = "claude"           # Anthropic Claude
    GPT4 = "gpt4"               # OpenAI GPT-4
    GPT4O = "gpt4o"             # OpenAI GPT-4o
    GPT5 = "gpt5"               # OpenAI GPT-5 Pro
    GEMINI = "gemini"           # Google Gemini
    GEMINI_PRO = "gemini_pro"   # Google Gemini Pro
    QWEN = "qwen"               # Alibaba Qwen Max
    DEEPSEEK = "deepseek"       # Deepseek
    GLM = "glm"                 # Zhipu GLM
    GROK = "grok"               # xAI Grok
    MISTRAL = "mistral"         # Mistral AI
    LLAMA = "llama"             # Meta Llama
    LOCAL = "local"             # Generic local LLM

    @property
    def full_name(self) -> str:
        """Full name of the provider."""
        names = {
            "claude": "Claude 4.5 Opus",
            "gpt4": "OpenAI GPT-4",
            "gpt4o": "OpenAI GPT-4o",
            "gpt5": "GPT-5 Pro",
            "gemini": "Google Gemini",
            "gemini_pro": "Gemini 3 Pro",
            "qwen": "Qwen Max",
            "deepseek": "Deepseek V3.2",
            "glm": "GLM 4.7",
            "grok": "Grok 4.1 Fast",
            "mistral": "Mistral AI",
            "llama": "Meta Llama",
            "local": "Local LLM",
        }
        return names.get(self.value, self.value)

    @property
    def default_model(self) -> str:
        """Default model ID for this provider (AIML API model names)."""
        models = {
            "claude": "anthropic/claude-4-opus-20250514",
            "gpt4": "openai/gpt-4-turbo",
            "gpt4o": "openai/gpt-4o",
            "gpt5": "openai/o3",
            "gemini": "google/gemini-2.0-flash",
            "gemini_pro": "google/gemini-2.5-pro-preview-06-05",
            "qwen": "Qwen/Qwen3-235B-A22B",
            "deepseek": "deepseek/DeepSeek-R1",
            "glm": "THUDM/GLM-4-32B-0414",
            "grok": "x-ai/grok-3-fast",
            "mistral": "mistralai/mistral-large-latest",
            "llama": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "local": "llama3.1:8b",
        }
        return models.get(self.value, "")

    @property
    def strengths(self) -> list[str]:
        """Known strengths for meteorological analysis."""
        strengths = {
            "claude": [
                "Excellent reasoning and explanation",
                "Strong physical interpretation",
                "Detailed causal analysis",
                "Good uncertainty quantification",
            ],
            "gpt4": [
                "Broad knowledge base",
                "Good pattern recognition",
                "Strong spatial reasoning",
            ],
            "gpt4o": [
                "Fast inference",
                "Multimodal capabilities",
                "Good for map interpretation",
            ],
            "gpt5": [
                "State-of-the-art reasoning",
                "Advanced scientific analysis",
                "Superior pattern recognition",
            ],
            "gemini": [
                "Fast processing",
                "Good for large contexts",
                "Multimodal analysis",
            ],
            "gemini_pro": [
                "Advanced reasoning",
                "Long context window",
                "Strong scientific knowledge",
            ],
            "qwen": [
                "Strong multilingual",
                "Excellent reasoning",
                "Good scientific knowledge",
            ],
            "deepseek": [
                "Advanced thinking/reasoning",
                "Strong analytical capabilities",
                "Good for complex problems",
            ],
            "glm": [
                "Fast inference",
                "Good general knowledge",
                "Cost-effective",
            ],
            "grok": [
                "Fast reasoning",
                "Real-time knowledge",
                "Strong analytical skills",
            ],
            "mistral": [
                "Efficient processing",
                "Good European focus",
                "Cost-effective",
            ],
            "llama": [
                "Open source",
                "Customizable",
                "Good general purpose",
            ],
        }
        return strengths.get(self.value, ["General purpose"])


class AIProviderConfig(BaseModel):
    """Configuration for an AI provider."""

    provider: AIProvider
    model_id: Optional[str] = Field(default=None, description="Specific model ID")
    api_key: Optional[str] = Field(default=None, description="API key (if required)")
    base_url: Optional[str] = Field(default=None, description="Custom API endpoint")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=100, le=100000)
    enabled: bool = Field(default=True)

    # Weight in ensemble
    base_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Base weight for this AI in ensemble"
    )

    # Historical performance (updated over time)
    historical_accuracy: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Historical accuracy score"
    )

    @property
    def effective_model(self) -> str:
        """Get the effective model ID."""
        return self.model_id or self.provider.default_model


@dataclass
class AIAnalysisResult:
    """Result from a single AI's meteorological analysis."""

    provider: AIProvider
    model_id: str
    timestamp: datetime

    # Analysis outputs
    synoptic_summary: str
    pattern_identification: list[str]
    physical_interpretation: str
    confidence_assessment: str
    key_findings: list[str]
    warnings: list[str]

    # Scores
    confidence_score: float  # 0-100
    reasoning_quality: float  # 0-1, self-assessed

    # Raw response for debugging
    raw_response: Optional[str] = None

    # Timing
    inference_time_ms: float = 0.0


class BaseAIAnalyzer(ABC):
    """Abstract base class for AI analyzers."""

    def __init__(self, config: AIProviderConfig):
        self.config = config
        self.provider = config.provider

    @abstractmethod
    async def analyze(
        self,
        model_data: dict,
        field_selection: list,
        forecast_hour: int,
    ) -> AIAnalysisResult:
        """Perform meteorological analysis."""
        pass

    def _build_analysis_prompt(
        self,
        model_data: dict,
        field_selection: list,
        forecast_hour: int,
    ) -> str:
        """Build the analysis prompt for the AI."""
        prompt = f"""You are an expert meteorological analyst. Analyze the following NWP model data
and provide a comprehensive meteorological assessment.

## Data Summary
- Forecast Hour: +{forecast_hour}h
- Models Available: {', '.join(model_data.keys()) if isinstance(model_data, dict) else 'Multiple'}
- Fields Analyzed: {', '.join(str(f) for f in field_selection)}

## Required Analysis

1. **Synoptic Summary**: Brief overview of the current/forecast synoptic situation

2. **Pattern Identification**: List the main synoptic patterns identified:
   - Troughs, ridges, cut-off lows, blocking
   - Pressure centers (highs/lows)
   - Frontal zones, jet stream position

3. **Physical Interpretation**: Explain WHY these patterns exist using:
   - Rossby wave dynamics
   - Baroclinic/barotropic processes
   - Thermal advection
   - Jet stream interactions

4. **Confidence Assessment**: Rate your confidence (0-100) and explain:
   - Model agreement level
   - Pattern predictability
   - Uncertainty sources

5. **Key Findings**: Bullet points of most important takeaways

6. **Warnings**: Any significant concerns or caveats

Provide scientifically rigorous analysis with physical causality.
"""
        return prompt


class ClaudeAnalyzer(BaseAIAnalyzer):
    """Anthropic Claude analyzer."""

    async def analyze(
        self,
        model_data: dict,
        field_selection: list,
        forecast_hour: int,
    ) -> AIAnalysisResult:
        """Analyze using Claude."""
        import time
        import os

        start_time = time.time()
        prompt = self._build_analysis_prompt(model_data, field_selection, forecast_hour)

        try:
            import anthropic

            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured")

            client = anthropic.Anthropic(api_key=api_key)

            message = client.messages.create(
                model=self.config.effective_model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text
            inference_time = (time.time() - start_time) * 1000

            # Parse response
            parsed = self._parse_response(response_text)

            return AIAnalysisResult(
                provider=self.provider,
                model_id=self.config.effective_model,
                timestamp=datetime.utcnow(),
                synoptic_summary=parsed.get("synoptic_summary", response_text[:500]),
                pattern_identification=parsed.get("patterns", ["Analisi pattern completata"]),
                physical_interpretation=parsed.get("physics", "Interpretazione fisica disponibile"),
                confidence_assessment=parsed.get("confidence_text", "Confidenza valutata"),
                key_findings=parsed.get("findings", ["Analisi completata"]),
                warnings=parsed.get("warnings", []),
                confidence_score=parsed.get("confidence_score", 80.0),
                reasoning_quality=0.90,
                raw_response=response_text,
                inference_time_ms=inference_time,
            )

        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            inference_time = (time.time() - start_time) * 1000
            return AIAnalysisResult(
                provider=self.provider,
                model_id=self.config.effective_model,
                timestamp=datetime.utcnow(),
                synoptic_summary=f"Errore nell'analisi: {str(e)}",
                pattern_identification=[],
                physical_interpretation="",
                confidence_assessment="Analisi non completata",
                key_findings=[],
                warnings=[f"Errore: {str(e)}"],
                confidence_score=0.0,
                reasoning_quality=0.0,
                inference_time_ms=inference_time,
            )

    def _parse_response(self, text: str) -> dict:
        """Parse Claude's response into structured data."""
        result = {
            "synoptic_summary": "",
            "patterns": [],
            "physics": "",
            "confidence_text": "",
            "confidence_score": 75.0,
            "findings": [],
            "warnings": [],
        }

        # Extract sections from response
        sections = text.split("##")
        for section in sections:
            lower = section.lower()
            if "synoptic" in lower or "sinottic" in lower:
                result["synoptic_summary"] = section.strip()[:800]
            elif "pattern" in lower:
                lines = [l.strip() for l in section.split("\n") if l.strip().startswith("-")]
                result["patterns"] = [l.lstrip("- ") for l in lines[:5]]
            elif "physic" in lower or "fisic" in lower:
                result["physics"] = section.strip()[:600]
            elif "confidence" in lower or "confidenza" in lower:
                result["confidence_text"] = section.strip()[:400]
                # Try to extract score
                import re
                match = re.search(r'(\d{1,3})\s*[%/]?\s*(?:su\s*100|out of 100)?', section)
                if match:
                    result["confidence_score"] = min(100, float(match.group(1)))
            elif "finding" in lower or "key" in lower:
                lines = [l.strip() for l in section.split("\n") if l.strip().startswith("-")]
                result["findings"] = [l.lstrip("- ") for l in lines[:5]]
            elif "warning" in lower or "avvert" in lower:
                lines = [l.strip() for l in section.split("\n") if l.strip().startswith("-")]
                result["warnings"] = [l.lstrip("- ") for l in lines[:3]]

        if not result["synoptic_summary"]:
            result["synoptic_summary"] = text[:800]

        return result


class GPT4Analyzer(BaseAIAnalyzer):
    """OpenAI GPT-4 analyzer."""

    async def analyze(
        self,
        model_data: dict,
        field_selection: list,
        forecast_hour: int,
    ) -> AIAnalysisResult:
        """Analyze using GPT-4."""
        import time
        import os

        start_time = time.time()
        prompt = self._build_analysis_prompt(model_data, field_selection, forecast_hour)

        try:
            from openai import OpenAI

            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not configured")

            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=self.config.effective_model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content
            inference_time = (time.time() - start_time) * 1000

            # Parse response
            parsed = self._parse_response(response_text)

            return AIAnalysisResult(
                provider=self.provider,
                model_id=self.config.effective_model,
                timestamp=datetime.utcnow(),
                synoptic_summary=parsed.get("synoptic_summary", response_text[:500]),
                pattern_identification=parsed.get("patterns", ["Analisi completata"]),
                physical_interpretation=parsed.get("physics", ""),
                confidence_assessment=parsed.get("confidence_text", ""),
                key_findings=parsed.get("findings", []),
                warnings=parsed.get("warnings", []),
                confidence_score=parsed.get("confidence_score", 75.0),
                reasoning_quality=0.85,
                raw_response=response_text,
                inference_time_ms=inference_time,
            )

        except Exception as e:
            logger.error(f"GPT-4 analysis failed: {e}")
            inference_time = (time.time() - start_time) * 1000
            return AIAnalysisResult(
                provider=self.provider,
                model_id=self.config.effective_model,
                timestamp=datetime.utcnow(),
                synoptic_summary=f"Errore: {str(e)}",
                pattern_identification=[],
                physical_interpretation="",
                confidence_assessment="",
                key_findings=[],
                warnings=[f"Errore: {str(e)}"],
                confidence_score=0.0,
                reasoning_quality=0.0,
                inference_time_ms=inference_time,
            )

    def _parse_response(self, text: str) -> dict:
        """Parse GPT-4's response."""
        result = {"synoptic_summary": text[:800], "patterns": [], "physics": "",
                  "confidence_text": "", "confidence_score": 75.0, "findings": [], "warnings": []}
        sections = text.split("##")
        for section in sections:
            lower = section.lower()
            if "synoptic" in lower:
                result["synoptic_summary"] = section.strip()[:800]
            elif "pattern" in lower:
                lines = [l.strip() for l in section.split("\n") if l.strip().startswith("-")]
                result["patterns"] = [l.lstrip("- ") for l in lines[:5]]
            elif "physical" in lower:
                result["physics"] = section.strip()[:600]
            elif "confidence" in lower:
                result["confidence_text"] = section.strip()[:400]
            elif "finding" in lower:
                lines = [l.strip() for l in section.split("\n") if l.strip().startswith("-")]
                result["findings"] = [l.lstrip("- ") for l in lines[:5]]
        return result


class GeminiAnalyzer(BaseAIAnalyzer):
    """Google Gemini analyzer."""

    async def analyze(
        self,
        model_data: dict,
        field_selection: list,
        forecast_hour: int,
    ) -> AIAnalysisResult:
        """Analyze using Gemini."""
        import time
        import os

        start_time = time.time()
        prompt = self._build_analysis_prompt(model_data, field_selection, forecast_hour)

        try:
            import google.generativeai as genai

            api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not configured")

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.config.effective_model)

            response = model.generate_content(prompt)
            response_text = response.text
            inference_time = (time.time() - start_time) * 1000

            # Parse response
            parsed = self._parse_response(response_text)

            return AIAnalysisResult(
                provider=self.provider,
                model_id=self.config.effective_model,
                timestamp=datetime.utcnow(),
                synoptic_summary=parsed.get("synoptic_summary", response_text[:500]),
                pattern_identification=parsed.get("patterns", ["Analisi completata"]),
                physical_interpretation=parsed.get("physics", ""),
                confidence_assessment=parsed.get("confidence_text", ""),
                key_findings=parsed.get("findings", []),
                warnings=parsed.get("warnings", []),
                confidence_score=parsed.get("confidence_score", 75.0),
                reasoning_quality=0.82,
                raw_response=response_text,
                inference_time_ms=inference_time,
            )

        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            inference_time = (time.time() - start_time) * 1000
            return AIAnalysisResult(
                provider=self.provider,
                model_id=self.config.effective_model,
                timestamp=datetime.utcnow(),
                synoptic_summary=f"Errore: {str(e)}",
                pattern_identification=[],
                physical_interpretation="",
                confidence_assessment="",
                key_findings=[],
                warnings=[f"Errore: {str(e)}"],
                confidence_score=0.0,
                reasoning_quality=0.0,
                inference_time_ms=inference_time,
            )

    def _parse_response(self, text: str) -> dict:
        """Parse Gemini's response."""
        result = {"synoptic_summary": text[:800], "patterns": [], "physics": "",
                  "confidence_text": "", "confidence_score": 75.0, "findings": [], "warnings": []}
        sections = text.split("##")
        for section in sections:
            lower = section.lower()
            if "synoptic" in lower:
                result["synoptic_summary"] = section.strip()[:800]
            elif "pattern" in lower:
                lines = [l.strip() for l in section.split("\n") if l.strip().startswith("-")]
                result["patterns"] = [l.lstrip("- ") for l in lines[:5]]
            elif "physical" in lower:
                result["physics"] = section.strip()[:600]
            elif "confidence" in lower:
                result["confidence_text"] = section.strip()[:400]
            elif "finding" in lower:
                lines = [l.strip() for l in section.split("\n") if l.strip().startswith("-")]
                result["findings"] = [l.lstrip("- ") for l in lines[:5]]
        return result


class AIMLAPIAnalyzer(BaseAIAnalyzer):
    """
    Unified analyzer using AIML API for all providers.

    AIML API provides access to multiple AI models through a single
    OpenAI-compatible endpoint.
    """

    async def analyze(
        self,
        model_data: dict[str, Any],
        field_selection: list[str],
        forecast_hour: int,
    ) -> AIAnalysisResult:
        """Analyze weather data using AIML API."""
        start_time = time.time()
        prompt = self._build_analysis_prompt(model_data, field_selection, forecast_hour)

        try:
            from openai import OpenAI

            api_key = self.config.api_key or os.getenv("AIML_API_KEY")
            if not api_key:
                raise ValueError("AIML_API_KEY not configured")

            client = OpenAI(
                api_key=api_key,
                base_url=AIML_API_BASE_URL,
            )

            response = client.chat.completions.create(
                model=self.config.effective_model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": "Sei un meteorologo esperto. Analizza i dati in italiano."},
                    {"role": "user", "content": prompt},
                ],
            )

            response_text = response.choices[0].message.content
            inference_time = (time.time() - start_time) * 1000

            # Parse response
            parsed = self._parse_response(response_text)

            return AIAnalysisResult(
                provider=self.provider,
                model_id=self.config.effective_model,
                timestamp=datetime.utcnow(),
                synoptic_summary=parsed.get("synoptic_summary", response_text[:500]),
                pattern_identification=parsed.get("patterns", ["Analisi completata"]),
                physical_interpretation=parsed.get("physics", ""),
                confidence_assessment=parsed.get("confidence_text", ""),
                key_findings=parsed.get("findings", []),
                warnings=parsed.get("warnings", []),
                confidence_score=parsed.get("confidence_score", 75.0),
                reasoning_quality=0.85,
                raw_response=response_text,
                inference_time_ms=inference_time,
            )

        except Exception as e:
            logger.error(f"{self.provider.full_name} analysis failed: {e}")
            inference_time = (time.time() - start_time) * 1000
            return AIAnalysisResult(
                provider=self.provider,
                model_id=self.config.effective_model,
                timestamp=datetime.utcnow(),
                synoptic_summary=f"Errore: {str(e)}",
                pattern_identification=[],
                physical_interpretation="",
                confidence_assessment="",
                key_findings=[],
                warnings=[f"Errore: {str(e)}"],
                confidence_score=0.0,
                reasoning_quality=0.0,
                inference_time_ms=inference_time,
            )

    def _parse_response(self, text: str) -> dict:
        """Parse the AI response."""
        result = {
            "synoptic_summary": text[:800],
            "patterns": [],
            "physics": "",
            "confidence_text": "",
            "confidence_score": 75.0,
            "findings": [],
            "warnings": []
        }

        sections = text.split("##")
        for section in sections:
            lower = section.lower()
            if "sinott" in lower or "synoptic" in lower:
                result["synoptic_summary"] = section.strip()[:800]
            elif "pattern" in lower:
                lines = [l.strip() for l in section.split("\n") if l.strip().startswith("-")]
                result["patterns"] = [l.lstrip("- ") for l in lines[:5]]
            elif "fisic" in lower or "physical" in lower:
                result["physics"] = section.strip()[:600]
            elif "confiden" in lower or "incertezz" in lower:
                result["confidence_text"] = section.strip()[:400]
            elif "finding" in lower or "conclus" in lower:
                lines = [l.strip() for l in section.split("\n") if l.strip().startswith("-")]
                result["findings"] = [l.lstrip("- ") for l in lines[:5]]

        return result


class MultiAIOrchestrator:
    """
    Orchestrates analysis across multiple AI providers.

    Runs analyses in parallel and combines results with weighting.
    Uses AIML API as unified backend when AIML_API_KEY is configured.
    """

    def __init__(self, configs: list[AIProviderConfig]):
        self.configs = [c for c in configs if c.enabled]
        self.use_aiml_api = bool(os.getenv("AIML_API_KEY"))
        self.analyzers = self._create_analyzers()

    def _create_analyzers(self) -> dict[AIProvider, BaseAIAnalyzer]:
        """Create analyzer instances for each configured provider."""
        # If AIML API key is set, use it for all providers
        if self.use_aiml_api:
            logger.info("Using AIML API for all AI providers")
            analyzers = {}
            for config in self.configs:
                analyzers[config.provider] = AIMLAPIAnalyzer(config)
            return analyzers

        # Otherwise, use native APIs
        analyzer_classes = {
            AIProvider.CLAUDE: ClaudeAnalyzer,
            AIProvider.GPT4: GPT4Analyzer,
            AIProvider.GPT4O: GPT4Analyzer,
            AIProvider.GPT5: GPT4Analyzer,  # Uses OpenAI API
            AIProvider.GEMINI: GeminiAnalyzer,
            AIProvider.GEMINI_PRO: GeminiAnalyzer,
            AIProvider.QWEN: AIMLAPIAnalyzer,
            AIProvider.DEEPSEEK: AIMLAPIAnalyzer,
            AIProvider.GLM: AIMLAPIAnalyzer,
            AIProvider.GROK: AIMLAPIAnalyzer,
            AIProvider.MISTRAL: AIMLAPIAnalyzer,
            AIProvider.LLAMA: AIMLAPIAnalyzer,
        }

        analyzers = {}
        for config in self.configs:
            analyzer_class = analyzer_classes.get(config.provider)
            if analyzer_class:
                analyzers[config.provider] = analyzer_class(config)

        return analyzers

    async def analyze_all(
        self,
        model_data: dict,
        field_selection: list,
        forecast_hour: int,
    ) -> list[AIAnalysisResult]:
        """Run analysis on all configured AI providers in parallel."""
        tasks = []

        for provider, analyzer in self.analyzers.items():
            task = analyzer.analyze(model_data, field_selection, forecast_hour)
            tasks.append(task)

        if not tasks:
            logger.warning("No AI analyzers configured")
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, AIAnalysisResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"AI analysis failed: {result}")

        return valid_results

    def compute_ensemble_weights(
        self,
        results: list[AIAnalysisResult]
    ) -> dict[AIProvider, float]:
        """Compute ensemble weights for AI results."""
        weights = {}

        for result in results:
            # Find config for this provider
            config = next(
                (c for c in self.configs if c.provider == result.provider),
                None
            )
            if config is None:
                continue

            # Combine base weight, historical accuracy, and self-reported quality
            weight = (
                config.base_weight *
                config.historical_accuracy *
                result.reasoning_quality *
                (result.confidence_score / 100.0)
            )
            weights[result.provider] = weight

        # Normalize weights
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}


# Default configurations
class AIPresets:
    """Preset AI configurations."""

    @staticmethod
    def single_claude() -> list[AIProviderConfig]:
        """Use only Claude."""
        return [AIProviderConfig(provider=AIProvider.CLAUDE)]

    @staticmethod
    def dual_analysis() -> list[AIProviderConfig]:
        """Claude + GPT-4 for cross-validation."""
        return [
            AIProviderConfig(provider=AIProvider.CLAUDE, base_weight=1.0),
            AIProviderConfig(provider=AIProvider.GPT4, base_weight=0.9),
        ]

    @staticmethod
    def multi_ai_ensemble() -> list[AIProviderConfig]:
        """Full multi-AI ensemble."""
        return [
            AIProviderConfig(provider=AIProvider.CLAUDE, base_weight=1.0),
            AIProviderConfig(provider=AIProvider.GPT4O, base_weight=0.95),
            AIProviderConfig(provider=AIProvider.GEMINI_PRO, base_weight=0.9),
            AIProviderConfig(provider=AIProvider.MISTRAL, base_weight=0.85),
        ]

    @staticmethod
    def local_only() -> list[AIProviderConfig]:
        """Local LLMs only (no API costs)."""
        return [
            AIProviderConfig(
                provider=AIProvider.LLAMA,
                base_url="http://localhost:11434",
                base_weight=1.0,
            ),
        ]
