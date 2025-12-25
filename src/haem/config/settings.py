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

    # AI Providers
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
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    @property
    def has_claude(self) -> bool:
        """Check if Claude API key is available."""
        return bool(self.anthropic_api_key)

    @property
    def has_gpt4(self) -> bool:
        """Check if GPT-4 API key is available."""
        return bool(self.openai_api_key)

    @property
    def has_gemini(self) -> bool:
        """Check if Gemini API key is available."""
        return bool(self.google_api_key)

    @property
    def available_providers(self) -> list[str]:
        """List of available AI providers."""
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

        # Claude
        status = "✅ Configured" if self.has_claude else "❌ Not set"
        lines.append(f"  Anthropic (Claude): {status}")

        # OpenAI
        status = "✅ Configured" if self.has_gpt4 else "❌ Not set"
        lines.append(f"  OpenAI (GPT-4):     {status}")

        # Google
        status = "✅ Configured" if self.has_gemini else "❌ Not set"
        lines.append(f"  Google (Gemini):    {status}")

        lines.append("")
        if self.available_providers:
            lines.append(f"  Available: {', '.join(self.available_providers)}")
        else:
            lines.append("  ⚠️ No AI providers configured - using demo mode")

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
