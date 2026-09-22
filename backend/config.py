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
    "JEV_PROVIDER", "OPENROUTER_API_KEY", "OPENROUTER_JEV_API_KEY",
    "OPENROUTER_BASE_URL", "OPENROUTER_JEV_MODEL", "OPENROUTER_APP_TITLE",
    "OPENROUTER_USER_AGENT", "OPENROUTER_HTTP_REFERER",
    "GEMINI_API_KEY", "GEMINI_OPTIMIZER_MODEL", "PROMPT_OPTIMIZATION_ENABLED",
    "LLM_INPUT_COST_PER_MTOK", "LLM_OUTPUT_COST_PER_MTOK",
    "LLM_FLAT_COST_PER_ARTICLE_USD",
    "LAYA_MODEL", "LAYA_DEVICE", "LAYA_REMOTE_URL", "LAYA_API_KEY",
    "DATABRICKS_HOST", "DATABRICKS_TOKEN", "LAYA_DATABRICKS_ENDPOINT",
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
    JEV_PROVIDER: str = "typesafe"
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_JEV_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai"
    OPENROUTER_JEV_MODEL: str = "typesafe/jev-1.13"
    OPENROUTER_APP_TITLE: str = "jev-curation-engine"
    OPENROUTER_USER_AGENT: str = "jev-curation-engine-benchmark/1.0"
    OPENROUTER_HTTP_REFERER: str = "https://muckrack.com/curation-engine"
    GEMINI_API_KEY: str = ""
    GEMINI_OPTIMIZER_MODEL: str = "gemini-2.5-flash"
    PROMPT_OPTIMIZATION_ENABLED: bool = False
    TYPESAFE_COST_PER_MILLION_INPUT_TOKENS: float = 0.042
    LAYA_MODEL: str = "laya:azure:t4"
    LAYA_DEVICE: str = "mps"
    LAYA_REMOTE_URL: str = "http://20.90.113.57:8000/api/alpha/decisions"
    LAYA_API_KEY: str = "laya_sec_8bc85b1d61947e71783c7ad23a65f14534558bf4127af230"
    # --- Databricks Model Serving (System 1 engine trained on Databricks GPU) ---
    # DATABRICKS_TOKEN is a PAT or service-principal token. The endpoint name is
    # the serving endpoint, not the Unity Catalog model name.
    DATABRICKS_HOST: str = ""
    DATABRICKS_TOKEN: str = ""
    LAYA_DATABRICKS_ENDPOINT: str = "laya-curation-engine"
    BENCHMARK_CONCURRENCY: int = 5
    LAYA_BENCHMARK_CONCURRENCY: int = 8
    VALIDATION_THRESHOLD: float = 0.45
    # Applied when no match term for the subject appears in the article text. The
    # presence check is a heuristic over a curator-authored label with no alias list
    # behind it, so absence raises the bar for the model instead of overruling it.
    ABSENT_ENTITY_VALIDATION_THRESHOLD: float = 0.65
    TAG_THRESHOLD: float = 0.40
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
