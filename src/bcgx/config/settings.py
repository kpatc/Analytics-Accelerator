"""Application settings using Pydantic v2 BaseSettings.

All configuration is read from environment variables (and optionally a .env
file).  Nested models group related settings for clarity.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataPathsConfig(BaseSettings):
    """Paths for the data layer."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DATA_",
        case_sensitive=False,
        extra="ignore",
    )

    raw_path: str = Field(default="data/raw", alias="DATA_RAW_PATH")
    processed_path: str = Field(default="data/processed", alias="DATA_PROCESSED_PATH")
    features_path: str = Field(default="data/features", alias="DATA_FEATURES_PATH")
    outputs_path: str = Field(default="data/outputs", alias="DATA_OUTPUTS_PATH")


class MLflowConfig(BaseSettings):
    """MLflow experiment tracking configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MLFLOW_",
        case_sensitive=False,
        extra="ignore",
    )

    tracking_uri: str = Field(
        default="http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )
    experiment_name: str = Field(
        default="bcgx-novamart", alias="MLFLOW_EXPERIMENT_NAME"
    )


class APIConfig(BaseSettings):
    """FastAPI backend configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="API_",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, alias="API_PORT")
    workers: int = Field(default=4, alias="API_WORKERS")


class AnthropicConfig(BaseSettings):
    """Anthropic Copilot configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ANTHROPIC_",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    model: str = Field(default="claude-sonnet-4-6", alias="ANTHROPIC_MODEL")


class OpenRouterConfig(BaseSettings):
    """OpenRouter configuration (OpenAI-compatible, for testing alternative models)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str = Field(default="", alias="OPENROUTER_KEY")
    model: str = Field(default="anthropic/claude-sonnet-4.5", alias="OPENROUTER_MODEL")


class DashboardConfig(BaseSettings):
    """Streamlit dashboard configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DASHBOARD_",
        case_sensitive=False,
        extra="ignore",
    )

    api_url: str = Field(default="http://localhost:8000", alias="DASHBOARD_API_URL")


class Settings(BaseSettings):
    """Root application settings.

    Composes all nested config groups and reads top-level env vars.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: str = Field(default="development")
    app_debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # Random seed for reproducibility
    random_seed: int = Field(default=42)

    # Nested config groups — populated from their own env prefixes
    data: DataPathsConfig = Field(default_factory=DataPathsConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)

    @property
    def is_production(self) -> bool:
        """Return True when running in the production environment."""
        return self.app_env.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Return True when running in the development environment."""
        return self.app_env.lower() == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton.

    Using ``@lru_cache`` ensures that the .env file is only read once per
    process lifetime, which is important for performance and consistency.
    """
    return Settings()
