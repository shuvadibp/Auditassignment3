"""Central configuration. Works locally and on Kaggle.

API keys are looked up in this order:
  1. environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY)
  2. Kaggle Secrets (Add-ons -> Secrets, same names)
  3. a .env file in the project root
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ON_KAGGLE = Path("/kaggle").exists()
# On Kaggle the project folder may be read-only (if added as a dataset) -> cache in /kaggle/working
CACHE_DIR = Path(os.environ.get("SMARTAUDIT_CACHE",
                 "/kaggle/working/.smartaudit_cache" if ON_KAGGLE else ROOT / ".cache"))
MODELS_JSON = ROOT / "models.json"
PROMPT_VERSION = "v1"

try:  # optional
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass


def get_secret(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    if ON_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            val = UserSecretsClient().get_secret(name)
            if val:
                os.environ[name] = val      # cache for SDKs that read the env
                return val
        except Exception:
            pass
    return default


def load_secrets_into_env() -> dict:
    """Call once at start (notebook / app). Returns which keys were found."""
    return {k: bool(get_secret(k)) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
