import os
import json

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load .env file ---
_env_file = os.path.join(_BASE_DIR, '.env')
if os.path.exists(_env_file):
    with open(_env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key not in os.environ:  # don't override existing env vars
                os.environ[key] = val

# --- Helper: load settings.json ---
def _load_settings():
    settings_file = os.path.join(_BASE_DIR, 'settings.json')
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _get_key(settings_name, env_name, default=""):
    """Priority: env var > settings.json > default"""
    env_val = os.getenv(env_name, "")
    if env_val:
        return env_val
    settings = _load_settings()
    return settings.get(settings_name, default)


# --- Database ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///gold_ai.db")

# --- API Keys ---
MOONSHOT_API_KEY = _get_key("kimi_api_key", "MOONSHOT_API_KEY", "")
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")

TAVILY_API_KEY = _get_key("tavily_api_key", "TAVILY_API_KEY", "")
SERPAPI_KEY = _get_key("serpapi_key", "SERPAPI_KEY", "")
DEEPSEEK_API_KEY = _get_key("deepseek_api_key", "DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
FRED_API_KEY = _get_key("fred_api_key", "FRED_API_KEY", "")

# --- Proxy ---
HTTP_PROXY = os.getenv("HTTP_PROXY", "")
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "")

# --- Flask ---
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")

# --- Cache TTLs (seconds) ---
MARKET_CACHE_TTL = int(os.getenv("MARKET_CACHE_TTL", "300"))
NEWS_CACHE_TTL = int(os.getenv("NEWS_CACHE_TTL", "300"))
SNAPSHOT_CACHE_TTL = int(os.getenv("SNAPSHOT_CACHE_TTL", "60"))


def reload_settings():
    """Force reload from settings.json (useful after user saves via web UI)."""
    import importlib
    importlib.reload(__import__(__name__))
