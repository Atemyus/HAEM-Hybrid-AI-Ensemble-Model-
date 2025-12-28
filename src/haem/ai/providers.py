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
    max_tokens: int = Field(default=8192, ge=100, le=100000)  # Increased for complete responses
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

        prompt = f"""Sei un meteorologo sinottico esperto che scrive per un pubblico misto: sia appassionati meteo alle prime armi sia professionisti del settore. Il tuo compito è fornire un'analisi COMPLETA, ACCURATA e COMPRENSIBILE.

{preset_instructions}
## Dati Analizzati
- **Campo Selezionato**: {selected_field.upper()} - {field_description}
- **Ora di Previsione**: +{forecast_hour}h
- **Modelli Disponibili**: {', '.join(model_data.keys()) if isinstance(model_data, dict) else 'ECMWF, GFS, ICON, GEM, ARPEGE'}
- **Campi di analisi**: {', '.join(str(f) for f in field_selection)}

## Statistiche dai Modelli NWP:
{data_str}

---

## REGOLE FONDAMENTALI (OBBLIGATORIE):

1. **OGNI SEZIONE DEVE CONTENERE ALMENO 4-6 FRASI COMPLETE** - Mai risposte brevi o incomplete
2. **NON SCRIVERE MAI "Non disponibile" o "Dati insufficienti"** - Anche se i dati sono limitati, fornisci comunque un'analisi ragionata basandoti sulla tua conoscenza meteorologica
3. **USA UN LINGUAGGIO ACCESSIBILE** - Spiega i termini tecnici tra parentesi quando li usi per la prima volta
4. **SPIEGA IL "PERCHÉ"** - Non limitarti a descrivere cosa succede, spiega sempre i meccanismi causali

---

### SEZIONE 1: SINTESI SINOTTICA (minimo 5-6 frasi)

Descrivi la situazione meteorologica a +{forecast_hour}h in modo chiaro e completo:

- **Centri d'azione**: Dove si trovano gli anticicloni (alta pressione) e i cicloni (bassa pressione)? Quanto sono intensi?
- **Campo di {selected_field.upper()}**: Come appare la distribuzione del campo? Ci sono gradienti (variazioni) significativi?
- **Flusso atmosferico**: Da dove soffia il vento in quota? È un flusso zonale (ovest-est) o meridiano (nord-sud)?
- **Evoluzione**: Come cambierà la situazione nelle ore successive?

Scrivi in modo che anche un lettore non esperto possa capire la situazione generale, mentre un esperto trovi informazioni tecniche utili.

---

### SEZIONE 2: INTERPRETAZIONE FISICA (minimo 5-6 frasi)

Spiega i PROCESSI FISICI che stanno causando questa configurazione atmosferica:

- **Dinamica delle onde**: Le ondulazioni del flusso (onde di Rossby) stanno amplificandosi o smorzandosi? Perché?
- **Processi energetici**: C'è conversione di energia potenziale in cinetica (ciclogenesi)? Dove e perché?
- **Avvezione termica**: Aria calda o fredda si sta muovendo verso la nostra area? Quali effetti produce?
- **Interazioni verticali**: Come interagisce il flusso in quota con quello al suolo? Ci sono forzanti dinamiche?

IMPORTANTE: Anche se il campo selezionato ({selected_field.upper()}) è "semplice", esistono SEMPRE processi fisici in atto. Descrivili. Se la situazione è stabile (es. anticiclone), spiega perché è stabile e cosa mantiene tale stabilità.

---

### SEZIONE 3: INCERTEZZE E DIVERGENZE TRA I MODELLI (minimo 5-6 frasi)

Analizza criticamente l'accordo tra i modelli meteorologici:

- **Grado di accordo**: I modelli (ECMWF, GFS, ICON, GEM, ARPEGE) concordano sulla situazione generale? Dove e quanto?
- **Divergenze specifiche**: Su quali aspetti i modelli differiscono? (timing, posizione, intensità dei fenomeni)
- **Scenari alternativi**: Se i modelli divergono, quali sono i possibili scenari? Stima le probabilità.
- **Affidabilità a +{forecast_hour}h**: A questa scadenza temporale, quanto è affidabile la previsione?

IMPORTANTE: Non scrivere mai solo "I modelli concordano" senza dettagli. Anche quando concordano, specifica SU COSA concordano e se ci sono piccole differenze. Se mancano dati espliciti sui singoli modelli, ragiona su come tipicamente si comportano i modelli in situazioni simili.

---

### SEZIONE 4: PATTERN IDENTIFICATI (minimo 4-5 elementi)

Elenca e DESCRIVI i pattern meteorologici presenti, spiegando cosa significano:

Formato richiesto (usa esattamente questo formato con il trattino):
- [Nome pattern]: [Descrizione dettagliata di posizione, intensità, e implicazioni meteo]

Esempi di pattern da cercare:
- Saccature atlantiche (avvallamenti del flusso)
- Promontori anticiclonici (espansioni di alta pressione)
- Cut-off/gocce fredde (vortici isolati)
- Configurazioni di blocco (pattern che bloccano il flusso zonale)
- Posizione del jet stream
- Fronti (caldi, freddi, occlusi)

IMPORTANTE: Identifica SEMPRE almeno 3-4 pattern. Anche in situazioni "tranquille" ci sono pattern: un anticiclone stazionario È un pattern. Un flusso zonale indisturbato È un pattern. Descrivilo.

---

### SEZIONE 5: CONCLUSIONI CHIAVE (minimo 4-5 punti)

Riassumi le conclusioni operative più importanti:

Formato richiesto:
- [Conclusione 1]: [Spiegazione pratica]
- [Conclusione 2]: [Spiegazione pratica]
...

Includi sempre:
- Cosa aspettarsi nelle prossime ore
- Livello di certezza della previsione (alta/media/bassa)
- Eventuali criticità o fenomeni da monitorare
- Implicazioni pratiche (precipitazioni, vento, temperature attese)

---

### SEZIONE 6: VALUTAZIONE CONFIDENZA

**PUNTEGGIO: [numero da 0 a 100]/100**

Giustifica il punteggio considerando:
- Accordo tra modelli (più concordano, più alta la confidenza)
- Prevedibilità del pattern (pattern stabili = alta confidenza)
- Scadenza temporale (+{forecast_hour}h è breve/media/lunga?)
- Continuità con le corse precedenti

---

## PROMEMORIA FINALE:
- OGNI sezione DEVE avere contenuto sostanziale (4-6 frasi minimo)
- MAI scrivere "Non disponibile", "Dati insufficienti", "In elaborazione"
- Se non hai dati specifici, usa la tua conoscenza meteorologica per fornire un'analisi ragionata
- Scrivi il PUNTEGGIO nel formato esatto: "PUNTEGGIO: XX/100"
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

        # === FALLBACK FINALE: Mai lasciare campi vuoti ===
        # Se dopo tutto il parsing un campo è vuoto, fornisci un contenuto di default contestuale

        if not result["synoptic_summary"] or len(result["synoptic_summary"]) < 50:
            result["synoptic_summary"] = ("L'analisi sinottica indica una configurazione atmosferica "
                "che richiede ulteriore monitoraggio. I modelli numerici mostrano una situazione "
                "in evoluzione che necessita di aggiornamenti continui per definire con precisione "
                "lo scenario meteorologico previsto. Si consiglia di verificare le prossime emissioni "
                "dei modelli per un quadro più dettagliato.")

        if not result["physics"] or len(result["physics"]) < 50:
            result["physics"] = ("I processi fisici in atto sono tipici della dinamica atmosferica "
                "a medie latitudini, dove l'interazione tra masse d'aria di diversa natura termica "
                "genera i pattern meteorologici osservati. La circolazione è governata dall'equilibrio "
                "geostrofico e dalle forzanti termiche e dinamiche che modulano il flusso zonale. "
                "L'evoluzione segue i principi della dinamica delle onde di Rossby.")

        if not result["uncertainty"] or len(result["uncertainty"]) < 50:
            result["uncertainty"] = ("L'accordo tra i modelli (ECMWF, GFS, ICON, GEM, ARPEGE) è da "
                "valutare in base alla scadenza temporale. Per le prime 48-72 ore generalmente si "
                "osserva buona convergenza sullo scenario principale, mentre per scadenze successive "
                "le incertezze aumentano progressivamente. Le differenze tra i modelli riguardano "
                "principalmente tempistiche e intensità dei fenomeni, tipiche della previsione numerica.")

        if not result["patterns"] or len(result["patterns"]) == 0:
            result["patterns"] = [
                "Flusso atmosferico principale: configurazione da analizzare in dettaglio",
                "Centri d'azione: posizione e intensità in evoluzione",
                "Gradienti termici: distribuzione tipica della stagione",
                "Jet stream: posizione e intensità da monitorare"
            ]

        if not result["findings"] or len(result["findings"]) == 0:
            result["findings"] = [
                "Monitorare l'evoluzione dei modelli nelle prossime emissioni",
                "Verificare la convergenza tra i diversi modelli numerici",
                "Prestare attenzione alle eventuali variazioni rispetto allo scenario base",
                "Considerare le incertezze tipiche della scadenza temporale analizzata"
            ]

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
