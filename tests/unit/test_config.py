"""Unit tests for the settings / configuration module.

These tests do NOT require a .env file — they rely on default values baked into
the Settings model, which ensures the test suite is fully hermetic.
"""

from __future__ import annotations

from src.bcgx.config.settings import Settings, get_settings


class TestSettingsDefaults:
    """Verify that Settings can be instantiated with sensible defaults."""

    def test_settings_loads_with_defaults(self) -> None:
        """Settings must be instantiable without any environment variables."""
        s = Settings()
        assert s is not None
        assert isinstance(s, Settings)

    def test_random_seed_defaults_to_42(self) -> None:
        """RANDOM_SEED should default to 42 when not set in the environment."""
        s = Settings()
        assert s.random_seed == 42

    def test_app_env_defaults_to_development(self) -> None:
        """APP_ENV should default to 'development'."""
        s = Settings()
        assert s.app_env == "development"

    def test_log_level_defaults_to_info(self) -> None:
        """LOG_LEVEL should default to 'INFO'."""
        s = Settings()
        assert s.log_level == "INFO"

    def test_api_port_defaults_to_8000(self) -> None:
        """API_PORT should default to 8000."""
        s = Settings()
        assert s.api.port == 8000

    def test_mlflow_experiment_name_default(self) -> None:
        """MLFLOW_EXPERIMENT_NAME should default to 'bcgx-novamart'."""
        s = Settings()
        assert s.mlflow.experiment_name == "bcgx-novamart"

    def test_is_development_property(self) -> None:
        """is_development property should return True for default env."""
        s = Settings()
        assert s.is_development is True
        assert s.is_production is False

    def test_is_production_property(self) -> None:
        """is_production should return True when app_env='production'."""
        s = Settings(app_env="production")
        assert s.is_production is True
        assert s.is_development is False


class TestSettingsCache:
    """Verify that get_settings() caching behaviour is correct."""

    def test_get_settings_returns_settings_instance(self) -> None:
        """get_settings() must return a Settings object."""
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_is_cached(self) -> None:
        """get_settings() must return the exact same object on repeated calls."""
        s1 = get_settings()
        s2 = get_settings()
        # lru_cache guarantees identity (not just equality)
        assert s1 is s2

    def test_get_settings_random_seed(self) -> None:
        """The cached settings must expose the expected random seed."""
        s = get_settings()
        assert s.random_seed == 42


class TestNestedConfigs:
    """Verify that nested config models are properly composed."""

    def test_data_paths_are_accessible(self) -> None:
        """DataPathsConfig sub-model must be accessible via settings.data."""
        s = Settings()
        assert hasattr(s, "data")
        assert s.data.raw_path == "data/raw"
        assert s.data.processed_path == "data/processed"
        assert s.data.features_path == "data/features"
        assert s.data.outputs_path == "data/outputs"

    def test_anthropic_config_is_accessible(self) -> None:
        """AnthropicConfig sub-model must be accessible via settings.anthropic."""
        s = Settings()
        assert hasattr(s, "anthropic")
        assert isinstance(s.anthropic.model, str)

    def test_dashboard_config_is_accessible(self) -> None:
        """DashboardConfig sub-model must be accessible."""
        s = Settings()
        assert hasattr(s, "dashboard")
        assert "localhost" in s.dashboard.api_url or "http" in s.dashboard.api_url
