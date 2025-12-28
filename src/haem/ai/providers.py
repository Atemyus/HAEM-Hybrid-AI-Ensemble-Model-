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
            "claude": "claude-opus-4-5",
            "gpt4": "gpt-4-turbo",
            "gpt4o": "gpt-4o",
            "gpt5": "gpt-5.2-chat-latest",
            "gemini": "gemini-2.5-flash",
            "gemini_pro": "gemini-2.5-pro",
            "qwen": "qwen-max",
            "deepseek": "deepseek/deepseek-thinking-v3.2-exp",
            "glm": "glm-4.7",
            "grok": "grok-4-1-fast-reasoning",
            "mistral": "mistralai/mistral-nemo",
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
        selected_field: str = "z500",
        field_description: str = "",
        preset_context: str = "",
    ) -> AIAnalysisResult:
        """Perform meteorological analysis."""
        pass

    def _build_analysis_prompt(
        self,
        model_data: dict,
        field_selection: list,
        forecast_hour: int,
        selected_field: str = "z500",
        field_description: str = "",
        preset_context: str = "",
    ) -> str:
        """Build the analysis prompt for the AI with field-specific and preset context."""

        # Format model data for display
        data_str = ""
        if isinstance(model_data, dict):
            for model, stats in model_data.items():
                data_str += f"\n### {model.upper()}:\n"
                for key, val in stats.items():
                    data_str += f"  - {key}: {val:.1f}\n"

        # Build preset-specific instructions
        preset_instructions = ""
        if preset_context:
            preset_instructions = f"\n**CONTESTO ANALISI**: {preset_context}\n"

        prompt = f"""Sei un meteorologo sinottico esperto con profonda conoscenza della fisica atmosferica.
Analizza i seguenti dati dei modelli NWP e fornisci una valutazione meteorologica DETTAGLIATA e COMPLETA in ITALIANO.

{preset_instructions}
## Dati Analizzati
- **Campo Selezionato**: {selected_field.upper()} - {field_description}
- **Ora di Previsione**: +{forecast_hour}h
- **Modelli Disponibili**: {', '.join(model_data.keys()) if isinstance(model_data, dict) else 'Multipli'}
- **Campi**: {', '.join(str(f) for f in field_selection)}

## Statistiche dai Modelli NWP:
{data_str}

## ISTRUZIONI IMPORTANTI:
Devi compilare TUTTI i seguenti campi con analisi DETTAGLIATE (minimo 3-4 frasi per campo).
NON lasciare campi vuoti o con risposte brevi.

### 1. SINTESI SINOTTICA
Descrivi in modo DETTAGLIATO la situazione sinottica prevista a +{forecast_hour}h:
- Posizione e intensità dei centri d'azione principali (anticicloni, cicloni)
- Configurazione del campo di {selected_field.upper()} ({field_description})
- Flusso prevalente e sue caratteristiche
- Evoluzione attesa nelle ore successive
(Scrivi almeno 4-5 frasi complete e tecnicamente accurate)

### 2. IDENTIFICAZIONE PATTERN
Elenca E DESCRIVI i pattern meteorologici identificati:
- Saccature: posizione, ampiezza, inclinazione dell'asse
- Promontori: estensione, intensità
- Cut-off o gocce fredde: presenza e caratteristiche
- Configurazioni di blocco: tipo e stabilità
- Posizione del jet stream: latitudine, intensità (kt), ondulazioni
(Per ogni pattern fornisci dettagli specifici, non solo elenchi)

### 3. INTERPRETAZIONE FISICA
Spiega i PROCESSI FISICI che determinano questa configurazione:
- Dinamica delle onde di Rossby: numero d'onda, propagazione, gruppo vs fase
- Processi baroclini: conversione di energia, sviluppo di cicloni
- Avvezione termica: calda/fredda, intensità, effetti sulla struttura
- Interazione jet-superficie: divergenza/convergenza, forzanti dinamiche
- Vorticità: avvezione, stretching, tilting
(Fornisci una spiegazione fisica causale completa, non superficiale)

### 4. VALUTAZIONE CONFIDENZA
**PUNTEGGIO: [inserisci un numero da 0 a 100]/100**

Valuta la confidenza della previsione considerando:
- Accordo tra modelli: quanto sono concordi ECMWF, GFS, ICON, ecc.?
- Prevedibilità intrinseca: quanto è predicibile questo tipo di pattern?
- Range temporale: +{forecast_hour}h è entro i limiti di buona prevedibilità?
- Stabilità delle corse: le ultime run hanno mostrato continuità?
(Giustifica il punteggio che hai dato con argomentazioni specifiche)

### 5. INCERTEZZE E DIVERGENZE TRA I MODELLI
Descrivi in DETTAGLIO:
- Quali modelli divergono e su quali aspetti (timing, posizione, intensità)
- Scenari alternativi possibili con probabilità stimate
- Elementi della previsione più incerti
- Soglie critiche da monitorare
(Non limitarti a dire "i modelli concordano", specifica le differenze)

### 6. CONCLUSIONI CHIAVE E IMPLICAZIONI METEO
- Sintesi dei punti più importanti dell'analisi
- Possibili impatti meteo al suolo (precipitazioni, vento, temperature)
- Eventuali criticità o allerte da considerare
- Raccomandazioni per il monitoraggio
(Rendi l'analisi utile per chi deve prendere decisioni operative)

IMPORTANTE:
- Scrivi SEMPRE il punteggio nel formato "PUNTEGGIO: XX/100" nella sezione 4
- Compila TUTTI i campi in modo dettagliato
- Usa terminologia tecnica ma comprensibile
- Basa le conclusioni sui dati forniti
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
                confidence_assessment=parsed.get("uncertainty", parsed.get("confidence_text", "")),
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
                confidence_assessment=parsed.get("uncertainty", parsed.get("confidence_text", "")),
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
        selected_field: str = "z500",
        field_description: str = "",
        preset_context: str = "",
    ) -> AIAnalysisResult:
        """Analyze weather data using AIML API."""
        start_time = time.time()
        prompt = self._build_analysis_prompt(
            model_data, field_selection, forecast_hour,
            selected_field=selected_field, field_description=field_description,
            preset_context=preset_context
        )

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
                confidence_assessment=parsed.get("uncertainty", parsed.get("confidence_text", "")),
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
        """Parse the AI response and extract all fields properly."""
        import re

        result = {
            "synoptic_summary": "",
            "patterns": [],
            "physics": "",
            "confidence_text": "",
            "confidence_score": 75.0,
            "uncertainty": "",
            "findings": [],
            "warnings": []
        }

        # Extract confidence score using multiple patterns
        confidence_patterns = [
            r'PUNTEGGIO[:\s]*(\d{1,3})\s*/\s*100',
            r'punteggio[:\s]*(\d{1,3})\s*/\s*100',
            r'\*\*PUNTEGGIO[:\s]*(\d{1,3})',
            r'confidenza[:\s]*(\d{1,3})\s*[%/]',
            r'(\d{1,3})\s*/\s*100',
        ]

        for pattern in confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    if 0 <= score <= 100:
                        result["confidence_score"] = score
                        break
                except (ValueError, IndexError):
                    continue

        # Split by ### or ## headers
        section_pattern = r'(?:^|\n)(?:###?\s*\d*\.?\s*)(.*?)(?=\n###?\s*\d*\.?\s*|\Z)'

        # Alternative: split by numbered sections or ## markers
        sections = re.split(r'\n(?:#{2,3}\s*\d*\.?\s*)', text)

        # Also try splitting by numbered patterns like "1." "2." etc at start of lines
        if len(sections) < 3:
            sections = re.split(r'\n(?=\d+\.\s+[A-Z])', text)

        for section in sections:
            if not section.strip():
                continue

            # Get first line as header, rest as content
            lines = section.strip().split('\n')
            header = lines[0].lower() if lines else ""
            content_lines = lines[1:] if len(lines) > 1 else []
            full_content = '\n'.join(content_lines).strip()

            # Also check full section for keywords if header is short
            section_lower = section.lower()

            # 1. SINTESI SINOTTICA
            if ('sintesi' in header and 'sinott' in header) or \
               ('sinott' in header) or \
               (not result["synoptic_summary"] and '1.' in header and 'sintesi' in section_lower):
                result["synoptic_summary"] = full_content[:1500] if full_content else section.strip()[:1500]

            # 2. PATTERN IDENTIFICATI (must check BEFORE interpretazione since both have similar words)
            elif ('pattern' in header and 'identific' in header) or \
                 ('2.' in header and 'pattern' in section_lower) or \
                 (header.startswith('2') and 'pattern' in section_lower):
                # Extract list items
                items = []
                for line in section.split('\n'):
                    line = line.strip()
                    if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                        items.append(line.lstrip('-•* ').strip())
                if items:
                    result["patterns"] = items[:8]
                elif full_content:
                    # If no list, use paragraphs
                    result["patterns"] = [full_content[:500]]

            # 3. INTERPRETAZIONE FISICA
            elif ('interpretazione' in header and 'fisic' in header) or \
                 ('fisic' in header and 'interpretazione' in section_lower) or \
                 ('3.' in header and 'fisic' in section_lower) or \
                 ('interpretazione' in header):
                result["physics"] = full_content[:1200] if full_content else ""

            # 4. VALUTAZIONE CONFIDENZA
            elif ('valutazione' in header and 'confiden' in header) or \
                 ('confidenza' in header) or \
                 ('4.' in header and 'confiden' in section_lower):
                result["confidence_text"] = full_content[:800] if full_content else ""

            # 5. INCERTEZZE E DIVERGENZE
            elif ('incertezz' in header and 'divergenz' in header) or \
                 ('incertezz' in header) or \
                 ('divergenz' in header) or \
                 ('5.' in header and ('incertezz' in section_lower or 'divergenz' in section_lower)):
                result["uncertainty"] = full_content[:1000] if full_content else ""

            # 6. CONCLUSIONI CHIAVE
            elif ('conclus' in header and 'chiave' in header) or \
                 ('conclus' in header) or \
                 ('implicazioni' in header) or \
                 ('6.' in header and 'conclus' in section_lower):
                # Extract list items OR paragraphs
                items = []
                for line in section.split('\n'):
                    line = line.strip()
                    if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                        items.append(line.lstrip('-•* ').strip())
                if items:
                    result["findings"] = items[:8]
                elif full_content:
                    # Split by sentences if no list
                    sentences = [s.strip() for s in full_content.split('.') if len(s.strip()) > 20]
                    result["findings"] = sentences[:6] if sentences else [full_content[:400]]

                # Check for warnings/alerts
                if 'allert' in section_lower or 'avvis' in section_lower or 'critic' in section_lower:
                    for line in section.split('\n'):
                        if 'allert' in line.lower() or 'avvis' in line.lower() or 'critic' in line.lower():
                            result["warnings"].append(line.strip().lstrip('-•* '))

        # Fallback parsing if main sections not found
        if not result["synoptic_summary"]:
            # Try to find any substantial text
            result["synoptic_summary"] = text[:1000]

        if not result["physics"]:
            # Look for physics keywords in full text
            physics_keywords = ['rossby', 'baroclino', 'avvezione', 'vorticità', 'jet stream', 'dinamica']
            for keyword in physics_keywords:
                if keyword in text.lower():
                    # Find the paragraph containing this keyword
                    paragraphs = text.split('\n\n')
                    for para in paragraphs:
                        if keyword in para.lower() and len(para) > 100:
                            result["physics"] = para[:1000]
                            break
                    if result["physics"]:
                        break

        if not result["uncertainty"]:
            # Look for uncertainty keywords
            uncertainty_keywords = ['incertezza', 'divergenza', 'modelli', 'differenz']
            for keyword in uncertainty_keywords:
                if keyword in text.lower():
                    paragraphs = text.split('\n\n')
                    for para in paragraphs:
                        if keyword in para.lower() and len(para) > 50:
                            result["uncertainty"] = para[:800]
                            break
                    if result["uncertainty"]:
                        break

        if not result["findings"]:
            # Look for conclusions keywords
            conclusion_keywords = ['conclus', 'sintesi', 'riassumendo', 'in definitiva']
            for keyword in conclusion_keywords:
                if keyword in text.lower():
                    paragraphs = text.split('\n\n')
                    for para in paragraphs:
                        if keyword in para.lower():
                            sentences = [s.strip() for s in para.split('.') if len(s.strip()) > 15]
                            result["findings"] = sentences[:5]
                            break
                    if result["findings"]:
                        break

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
        selected_field: str = "z500",
        field_description: str = "",
        preset_context: str = "",
    ) -> list[AIAnalysisResult]:
        """Run analysis on all configured AI providers in parallel."""
        tasks = []

        for provider, analyzer in self.analyzers.items():
            task = analyzer.analyze(
                model_data,
                field_selection,
                forecast_hour,
                selected_field=selected_field,
                field_description=field_description,
                preset_context=preset_context,
            )
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
