import time
import logging
from typing import Any, Dict, Optional, Tuple
import httpx
from config import settings

logger = logging.getLogger(__name__)

class TypeSafeRunner:
    def __init__(self):
        self.provider = settings.JEV_PROVIDER.lower()
        self.api_key = settings.TYPESAFE_API_KEY
        self.api_base = settings.TYPESAFE_API_BASE
        self.model = settings.TYPESAFE_MODEL
        self.cost_per_m_input = settings.TYPESAFE_COST_PER_MILLION_INPUT_TOKENS
        if self.provider == "openrouter":
            self.api_key = settings.OPENROUTER_JEV_API_KEY or settings.OPENROUTER_API_KEY
            self.api_base = settings.OPENROUTER_BASE_URL
            self.model = settings.OPENROUTER_JEV_MODEL

    async def evaluate_article(
        self,
        state: str,
        questions: Dict[str, Dict[str, Any]],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        target_model = model or self.model
        if self.provider == "openrouter":
            root = self.api_base.rstrip("/")
            if root.endswith("/api/v1"):
                root = root[:-len("/api/v1")]
            endpoint = f"{root}/api/alpha/decisions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": settings.OPENROUTER_APP_TITLE,
            }
        else:
            endpoint = f"{self.api_base.rstrip('/')}/systemone"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        payload = {
            "state": state,
            "model": target_model,
            "questions": questions
        }

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            if resp.is_error:
                raise RuntimeError(f"Jev {self.provider} API {resp.status_code}: {resp.text[:1000]}")
            data = resp.json()

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # Jev cost: $0.042 / M input tokens, output tokens free
        cost_usd = (input_tokens / 1_000_000.0) * self.cost_per_m_input

        return {
            "provider": self.provider,
            "model": data.get("model", target_model),
            "answers": data.get("answers", {}),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            },
            "duration_ms": round(duration_ms, 2),
            "cost_usd": cost_usd,
            # Observability: the exact wire payload and full raw response.
            "request_payload": payload,
            "raw_response": data,
        }

typesafe_runner = TypeSafeRunner()
