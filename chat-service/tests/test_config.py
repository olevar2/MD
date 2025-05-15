"""
Tests for configuration settings (app.config.settings)
"""
import pytest
import os
from pydantic import ValidationError

from app.config.settings import Settings, get_settings

# Mark all tests in this file as asyncio if any async fixtures/tests are added
# pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="function")
def mock_env_vars(monkeypatch):
    """Fixture to temporarily set environment variables for a test."""
    # Backup original env vars that we might change
    original_vars = {}
    def set_env(key, value):
        if key in os.environ:
            original_vars[key] = os.environ[key]
        monkeypatch.setenv(key, value)

    yield set_env

    # Restore original env vars
    for key, value in original_vars.items():
        monkeypatch.setenv(key, value)
    # Unset any vars that were newly added
    # This part is tricky with monkeypatch; usually, it handles unsetting.
    # If a var was set that wasn't there before, monkeypatch should remove it.

def test_settings_load_from_env(mock_env_vars):
    """Test that Settings loads values from environment variables."""
    mock_env_vars("SECRET_KEY", "env_secret_key")
    mock_env_vars("DATABASE_URL", "env_db_url")
    mock_env_vars("REDIS_URL", "redis://env_redis_host:1234/1")
    mock_env_vars("ANALYSIS_SERVICE_URL", "http://env.analysis.service")
    mock_env_vars("ML_SERVICE_URL", "http://env.ml.service")
    mock_env_vars("LOG_LEVEL", "DEBUG")
    mock_env_vars("KAFKA_BOOTSTRAP_SERVERS", "env_kafka:9092")
    mock_env_vars("ALLOWED_ORIGINS", "http://example.com,http://localhost:3000")

    # Clear lru_cache for get_settings to force re-evaluation
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.SECRET_KEY == "env_secret_key"
    assert settings.DATABASE_URL == "env_db_url"
    assert settings.REDIS_URL == "redis://env_redis_host:1234/1"
    assert settings.ANALYSIS_SERVICE_URL == "http://env.analysis.service"
    assert settings.ML_SERVICE_URL == "http://env.ml.service"
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.KAFKA_BOOTSTRAP_SERVERS == "env_kafka:9092"
    assert settings.ALLOWED_ORIGINS == ["http://example.com", "http://localhost:3000"]
    assert settings.cors_origins == ["http://example.com", "http://localhost:3000"]

def test_settings_default_values(mock_env_vars):
    """Test that Settings uses default values when env vars are not set (for non-required fields)."""
    # Ensure required fields are set to avoid validation error, others should take defaults
    mock_env_vars("SECRET_KEY", "default_test_secret")
    mock_env_vars("DATABASE_URL", "default_test_db_url")
    mock_env_vars("REDIS_URL", "redis://default_redis_url:6379/0") # Needs a default structure if not fully mocked
    mock_env_vars("ANALYSIS_SERVICE_URL", "http://default.analysis.service")
    mock_env_vars("ML_SERVICE_URL", "http://default.ml.service")

    # Unset some optional env vars to test defaults
    if "APP_HOST" in os.environ: os.environ.pop("APP_HOST")
    if "LOG_LEVEL" in os.environ: os.environ.pop("LOG_LEVEL")
    if "ALLOWED_ORIGINS" in os.environ: os.environ.pop("ALLOWED_ORIGINS")

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.APP_NAME == "chat-service"
    assert settings.APP_HOST == "0.0.0.0"
    assert settings.APP_PORT == 8000
    assert settings.API_PREFIX == "/api/v1"
    assert settings.API_DEBUG is False # Default for bool
    assert settings.API_KEY_NAME == "X-API-Key"
    assert settings.DATABASE_POOL_SIZE == 5
    assert settings.EVENT_BUS_TYPE == "kafka" # Default
    assert settings.LOG_LEVEL == "INFO" # Default
    assert settings.ALLOWED_ORIGINS == ["*"] # Default
    assert settings.cors_origins == ["*"]

def test_settings_missing_required_env_vars(mock_env_vars):
    """Test that Settings raises ValidationError if required env vars are missing."""
    # Unset a required environment variable (e.g., SECRET_KEY)
    # Ensure other required ones are present to isolate the error
    mock_env_vars("DATABASE_URL", "some_db_url")
    mock_env_vars("REDIS_URL", "some_redis_url")
    mock_env_vars("ANALYSIS_SERVICE_URL", "http://some.analysis.service")
    mock_env_vars("ML_SERVICE_URL", "http://some.ml.service")
    
    # Use monkeypatch to remove SECRET_KEY from os.environ for this test
    # This is safer than os.environ.pop directly in a test if not using monkeypatch for it.
    original_secret_key = os.environ.pop("SECRET_KEY", None)

    get_settings.cache_clear()
    with pytest.raises(ValidationError) as exc_info:
        Settings() # Instantiate directly to bypass lru_cache if needed for this specific test
    
    assert "SECRET_KEY" in str(exc_info.value).lower() # Check that the error message mentions SECRET_KEY
    assert any(err['loc'][0] == 'SECRET_KEY' for err in exc_info.value.errors() if 'loc' in err)

    # Restore SECRET_KEY if it was originally set
    if original_secret_key is not None:
        os.environ["SECRET_KEY"] = original_secret_key

def test_redis_url_parsing(mock_env_vars):
    """Test the REDIS_URL validator and parsing logic."""
    # Case 1: REDIS_URL is provided
    mock_env_vars("SECRET_KEY", "test_secret")
    mock_env_vars("DATABASE_URL", "test_db")
    mock_env_vars("ANALYSIS_SERVICE_URL", "http://analysis.service")
    mock_env_vars("ML_SERVICE_URL", "http://ml.service")
    mock_env_vars("REDIS_URL", "redis://mycustomhost:1111/2")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.REDIS_URL == "redis://mycustomhost:1111/2"

    # Case 2: REDIS_URL is not provided, should use defaults from REDIS_HOST, REDIS_PORT, REDIS_DB
    # For this, we need to ensure REDIS_URL is NOT in env, and other REDIS_* vars are at defaults
    # Pydantic's @validator(pre=True) runs before other field assignments if REDIS_URL is missing.
    # The current validator logic: if REDIS_URL is not set (None or empty string from env), it constructs one.
    # So, we need to simulate REDIS_URL being absent or empty.
    if "REDIS_URL" in os.environ: os.environ.pop("REDIS_URL")
    # Ensure other REDIS_ fields are at their defaults for the Settings class
    # Settings class defaults: REDIS_HOST="localhost", REDIS_PORT=6379, REDIS_DB=0
    get_settings.cache_clear()
    settings_default_redis = Settings(
        SECRET_KEY="test_secret", 
        DATABASE_URL="test_db", 
        ANALYSIS_SERVICE_URL="http://a.s", 
        ML_SERVICE_URL="http://m.s"
        # REDIS_URL is deliberately omitted to trigger the validator's default construction
    )
    assert settings_default_redis.REDIS_URL == "redis://localhost:6379/0"

def test_cors_origins_parsing(mock_env_vars):
    """Test the cors_origins property for different ALLOWED_ORIGINS inputs."""
    mock_env_vars("SECRET_KEY", "test_secret")
    mock_env_vars("DATABASE_URL", "test_db")
    mock_env_vars("REDIS_URL", "redis://test:6379/0")
    mock_env_vars("ANALYSIS_SERVICE_URL", "http://analysis.service")
    mock_env_vars("ML_SERVICE_URL", "http://ml.service")

    # Case 1: Default (not set in env, should be ["*"])
    if "ALLOWED_ORIGINS" in os.environ: os.environ.pop("ALLOWED_ORIGINS")
    get_settings.cache_clear()
    settings1 = get_settings()
    assert settings1.ALLOWED_ORIGINS == ["*"]
    assert settings1.cors_origins == ["*"]

    # Case 2: Single string value in env
    mock_env_vars("ALLOWED_ORIGINS", "http://single.example.com")
    get_settings.cache_clear()
    settings2 = get_settings()
    # Pydantic converts single string to list of one string for List[str] type
    assert settings2.ALLOWED_ORIGINS == ["http://single.example.com"]
    assert settings2.cors_origins == ["http://single.example.com"]

    # Case 3: Comma-separated string in env
    mock_env_vars("ALLOWED_ORIGINS", "http://one.com, http://two.com,http://three.com")
    get_settings.cache_clear()
    settings3 = get_settings()
    # The property `cors_origins` does the splitting if ALLOWED_ORIGINS is a string.
    # However, Pydantic BaseSettings might automatically convert comma-separated to list for List[str]
    # Let's check the type of settings3.ALLOWED_ORIGINS first.
    # If Pydantic already converted it to a list of one string "http://one.com, http://two.com,http://three.com"
    # then cors_origins will split that. If Pydantic made it a list of strings, cors_origins just returns it.
    # Based on Pydantic behavior, it should parse it into a list of strings directly if the env var is a comma-separated string.
    # Let's assume Pydantic parses it into a list of strings: ["http://one.com", " http://two.com", "http://three.com"]
    # The current cors_origins property handles if self.ALLOWED_ORIGINS is a string, not if it's already a list.
    # If ALLOWED_ORIGINS is already a list (e.g. from a .env file with JSON array format or Pydantic's auto-conversion),
    # the property should just return it. The validator for ALLOWED_ORIGINS might be needed if complex parsing is required.
    # The current Settings class has `ALLOWED_ORIGINS: List[str] = Field(default=["*"])`
    # Pydantic v1.10+ can parse comma-separated strings from env into List[str]
    expected_origins = ["http://one.com", "http://two.com", "http://three.com"]
    assert settings3.ALLOWED_ORIGINS == expected_origins
    assert settings3.cors_origins == expected_origins

    # Case 4: Empty string in env (should default to ["*"] via cors_origins logic)
    mock_env_vars("ALLOWED_ORIGINS", "")
    get_settings.cache_clear()
    settings4 = get_settings()
    assert settings4.ALLOWED_ORIGINS == [] # Pydantic might parse empty string to empty list for List[str]
    assert settings4.cors_origins == ["*"] # Property handles empty list

# Ensure get_settings is cached
def test_get_settings_is_cached(mock_env_vars):
    """Test that get_settings uses lru_cache."""
    mock_env_vars("SECRET_KEY", "cache_test_secret")
    mock_env_vars("DATABASE_URL", "cache_test_db")
    mock_env_vars("REDIS_URL", "redis://cache_test_redis:6379/0")
    mock_env_vars("ANALYSIS_SERVICE_URL", "http://cache.analysis.service")
    mock_env_vars("ML_SERVICE_URL", "http://cache.ml.service")

    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2 # Should be the same object due to caching

    # Change an env var and check if cache returns old or new (it should return old until cleared)
    mock_env_vars("LOG_LEVEL", "DEBUG_FOR_CACHE_TEST")
    s3 = get_settings()
    assert s3.LOG_LEVEL != "DEBUG_FOR_CACHE_TEST" # Still INFO from first call
    assert s3 is s1 # Still the same cached object

    get_settings.cache_clear()
    s4 = get_settings()
    assert s4.LOG_LEVEL == "DEBUG_FOR_CACHE_TEST" # New object with updated value
    assert s4 is not s1 # Different object after cache clear