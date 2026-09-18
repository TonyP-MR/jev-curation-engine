import os
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Keys this rig owns. The .env file is the single source of truth for all of them:
# the harness session exports staging defaults (ENVIRONMENT, MYSQL_HOST, AUDIT_BLOB_*)
# into every spawned process, which would otherwise silently override .env.
_MANAGED_KEYS = [
    "ENVIRONMENT",
    "MYSQL_HOST", "MYSQL_PORT", "MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD",
    "MYSQL_SSL_CA",
    "AUDIT_BLOB_ACCOUNT_URL", "AUDIT_BLOB_CONTAINER", "AUDIT_BLOB_SUFFIX",
    "AZURE_STORAGE_ACCOUNT", "AZURE_STORAGE_KEY",
    "TYPESAFE_API_KEY", "TYPESAFE_MODEL", "TYPESAFE_API_BASE",
    "LLM_INPUT_COST_PER_MTOK", "LLM_OUTPUT_COST_PER_MTOK",
    "LLM_FLAT_COST_PER_ARTICLE_USD",
]


def _parse_env_file(path: str) -> dict:
    result: dict = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip().strip("'\"")
    except FileNotFoundError:
        pass
    return result


def _apply_env_file_authority() -> None:
    """Force .env values (or absence) over the ambient process environment."""
    file_values = _parse_env_file(_ENV_FILE)
    for key in _MANAGED_KEYS:
        if key in file_values and file_values[key] != "":
            os.environ[key] = file_values[key]
        else:
            os.environ.pop(key, None)


_apply_env_file_authority()
_ENV_NAME = os.environ.get("ENVIRONMENT", "staging").lower()

# Per-environment infrastructure defaults. Credentials come from .env.
_ENV_DEFAULTS = {
    "staging": {
        "MYSQL_HOST": "mysql-curation-staging.mysql.database.azure.com",
        "AUDIT_BLOB_ACCOUNT_URL": "https://stcurationauditstg.blob.core.windows.net",
        "AUDIT_BLOB_SUFFIX": "_staging",
        "AZURE_STORAGE_ACCOUNT": "stcurationauditstg",
    },
    "production": {
        "MYSQL_HOST": "mysql-curation-production.mysql.database.azure.com",
        "AUDIT_BLOB_ACCOUNT_URL": "https://stcurationauditprod.blob.core.windows.net",
        "AUDIT_BLOB_SUFFIX": "_production",
        "AZURE_STORAGE_ACCOUNT": "stcurationauditprod",
    },
}
_defaults = _ENV_DEFAULTS.get(_ENV_NAME, _ENV_DEFAULTS["staging"])


class Settings(BaseSettings):
    ENVIRONMENT: str = _ENV_NAME
    MYSQL_HOST: str = _defaults["MYSQL_HOST"]
    MYSQL_PORT: int = 3306
    MYSQL_DATABASE: str = "curation_engine"
    MYSQL_USER: str = "caboradmin"
    MYSQL_PASSWORD: str = ""
    MYSQL_SSL_CA: str = "DigiCertGlobalRootG2.crt.pem"

    AUDIT_BLOB_ACCOUNT_URL: str = _defaults["AUDIT_BLOB_ACCOUNT_URL"]
    AUDIT_BLOB_CONTAINER: str = "pipeline-audit"
    AUDIT_BLOB_SUFFIX: str = _defaults["AUDIT_BLOB_SUFFIX"]
    AZURE_STORAGE_ACCOUNT: str = _defaults["AZURE_STORAGE_ACCOUNT"]
    AZURE_STORAGE_KEY: str = ""

    TYPESAFE_API_KEY: str = ""
    TYPESAFE_API_BASE: str = "https://api.typesafe.ai/v1"
    TYPESAFE_MODEL: str = "jev-latest"
    # --- Jev pricing ---
    # Charged per input token only: $42/Btok = $0.042/Mtok. Output tokens are free.
    TYPESAFE_COST_PER_MILLION_INPUT_TOKENS: float = 0.042

    # --- Baseline LLM pricing (used when the blob has no recorded cost) ---
    # Audit blobs write llm_cost_usd = null, so the rig estimates from llm_tokens.
    # Google Gemini 2.5 Flash list pricing:
    LLM_INPUT_COST_PER_MTOK: float = 0.30
    LLM_OUTPUT_COST_PER_MTOK: float = 2.50
    # Flat per-article override. When > 0 this wins over the token estimate; it is the
    # honest default because blob token counts omit billed thinking tokens.
    # Ground truth observed on staging: ~$0.007 per article for gemini-2.5-flash.
    LLM_FLAT_COST_PER_ARTICLE_USD: float = 0.007

    # Caches are separated per environment so staging and production never mix.
    CACHE_DIR: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".cache", _ENV_NAME)
    )
    RUNS_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs"))

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")


settings = Settings()
