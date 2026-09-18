import time
import logging
from typing import Any, Dict, Optional, Tuple
import httpx
from config import settings

logger = logging.getLogger(__name__)

class TypeSafeRunner:
    def __init__(self):
        self.api_key = settings.TYPESAFE_API_KEY
        self.api_base = settings.TYPESAFE_API_BASE
        self.model = settings.TYPESAFE_MODEL
        self.cost_per_m_input = settings.TYPESAFE_COST_PER_MILLION_INPUT_TOKENS

    async def evaluate_article(
        self,
        state: str,
        questions: Dict[str, Dict[str, Any]],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes a single POST /v1/systemone request against TypeSafe Jev model.
        Returns answers, usage, timing, and calculated cost.
        """
        target_model = model or self.model
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "state": state,
            "model": target_model,
            "questions": questions
        }

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.api_base}/systemone",
                json=payload,
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # Jev cost: $0.042 / M input tokens, output tokens free
        cost_usd = (input_tokens / 1_000_000.0) * self.cost_per_m_input

        return {
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
