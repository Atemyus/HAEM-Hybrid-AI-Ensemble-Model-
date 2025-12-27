"""
Environment and API Keys Configuration for HAEM.

Loads configuration from environment variables for secure deployment.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load from .env file if it exists
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, use system env vars only


class APIKeys(BaseModel):
    """API Keys configuration loaded from environment."""

    # AIML API (unified access to all models)
    aiml_api_key: Optional[str] = Field(
        default=None,
        description="AIML API key for unified access to all AI models"
    )

    # Individual AI Providers (fallback if no AIML API)
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for GPT-4"
    )
    google_api_key: Optional[str] = Field(
        default=None,
        description="Google API key for Gemini"
    )

    @classmethod
    def from_env(cls) -> "APIKeys":
        """Load API keys from environment variables."""
        return cls(
            aiml_api_key=os.getenv("AIML_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    @property
    def has_aiml_api(self) -> bool:
        """Check if AIML API key is available (enables all providers)."""
        return bool(self.aiml_api_key)

    @property
    def has_claude(self) -> bool:
        """Check if Claude is available."""
        return self.has_aiml_api or bool(self.anthropic_api_key)

    @property
    def has_gpt4(self) -> bool:
        """Check if GPT-4 is available."""
        return self.has_aiml_api or bool(self.openai_api_key)

    @property
    def has_gpt5(self) -> bool:
        """Check if GPT-5 is available (via AIML API)."""
        return self.has_aiml_api

    @property
    def has_gemini(self) -> bool:
        """Check if Gemini is available."""
        return self.has_aiml_api or bool(self.google_api_key)

    @property
    def has_qwen(self) -> bool:
        """Check if Qwen is available (via AIML API)."""
        return self.has_aiml_api

    @property
    def has_deepseek(self) -> bool:
        """Check if Deepseek is available (via AIML API)."""
        return self.has_aiml_api

    @property
    def has_glm(self) -> bool:
        """Check if GLM is available (via AIML API)."""
        return self.has_aiml_api

    @property
    def has_grok(self) -> bool:
        """Check if Grok is available (via AIML API)."""
        return self.has_aiml_api

    @property
    def available_providers(self) -> list[str]:
        """List of available AI providers."""
        if self.has_aiml_api:
            return ["Claude", "GPT-5", "Gemini", "Qwen", "Deepseek", "GLM", "Grok"]

        providers = []
        if self.has_claude:
            providers.append("Claude")
        if self.has_gpt4:
            providers.append("GPT-4")
        if self.has_gemini:
            providers.append("Gemini")
        return providers

    def get_status_summary(self) -> str:
        """Get a summary of API key status."""
        lines = ["API Keys Status:", ""]

        if self.has_aiml_api:
            lines.append("  AIML API: ✅ Configured (all providers available)")
            lines.append("")
            lines.append("  Available via AIML API:")
            lines.append("    - Claude 4.5 Opus")
            lines.append("    - GPT-5 Pro")
            lines.append("    - Gemini 3 Pro")
            lines.append("    - Qwen Max")
            lines.append("    - Deepseek V3.2")
            lines.append("    - GLM 4.7")
            lines.append("    - Grok 4.1 Fast")
        else:
            # Claude
            status = "✅ Configured" if bool(self.anthropic_api_key) else "❌ Not set"
            lines.append(f"  Anthropic (Claude): {status}")

            # OpenAI
            status = "✅ Configured" if bool(self.openai_api_key) else "❌ Not set"
            lines.append(f"  OpenAI (GPT-4):     {status}")

            # Google
            status = "✅ Configured" if bool(self.google_api_key) else "❌ Not set"
            lines.append(f"  Google (Gemini):    {status}")

            lines.append("")
            if self.available_providers:
                lines.append(f"  Available: {', '.join(self.available_providers)}")
            else:
                lines.append("  ⚠️ No AI providers configured - using demo mode")
                lines.append("  💡 Tip: Set AIML_API_KEY for access to all models")

        return "\n".join(lines)


class AppConfig(BaseModel):
    """Application configuration."""

    # Server settings
    port: int = Field(default=8501, description="Server port")
    host: str = Field(default="0.0.0.0", description="Server host")

    # Cache settings
    cache_dir: str = Field(default="/tmp/haem_cache", description="Cache directory")

    # Update settings
    auto_refresh_minutes: int = Field(default=60, description="Auto-refresh interval")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load config from environment."""
        return cls(
            port=int(os.getenv("PORT", "8501")),
            host=os.getenv("HOST", "0.0.0.0"),
            cache_dir=os.getenv("HAEM_CACHE_DIR", "/tmp/haem_cache"),
            auto_refresh_minutes=int(os.getenv("HAEM_REFRESH_MINUTES", "60")),
            log_level=os.getenv("HAEM_LOG_LEVEL", "INFO"),
        )


# Global instances
api_keys = APIKeys.from_env()
app_config = AppConfig.from_env()


def get_api_keys() -> APIKeys:
    """Get the global API keys instance."""
    return api_keys


def get_app_config() -> AppConfig:
    """Get the global app config instance."""
    return app_config
