from typing import Any, Dict, List, Optional
import time
import logging
from typing import Any, Dict, Optional
import httpx
from config import settings
from laya_runner import laya_runner
logger = logging.getLogger(__name__)

class TypeSafeRunner:
    def __init__(self):
        self.provider = settings.JEV_PROVIDER.lower()
        self.api_key = settings.TYPESAFE_API_KEY
        self.api_base = settings.TYPESAFE_API_BASE
        self.model = settings.TYPESAFE_MODEL
        self.cost_per_m_input = settings.TYPESAFE_COST_PER_MILLION_INPUT_TOKENS
        self._client: Optional[httpx.AsyncClient] = None
        if self.provider == "openrouter":
            self.api_key = settings.OPENROUTER_JEV_API_KEY or settings.OPENROUTER_API_KEY
            self.api_base = settings.OPENROUTER_BASE_URL
            self.model = settings.OPENROUTER_JEV_MODEL

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
            self._client = httpx.AsyncClient(limits=limits, timeout=120.0)
        return self._client
    async def evaluate_article(
        self,
        state: str,
        questions: Dict[str, Dict[str, Any]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        effective_provider = (provider or self.provider).lower()
        target_model = model or (settings.LAYA_MODEL if effective_provider.startswith("laya") else self.model)

        if (
            effective_provider in ("laya_azure", "laya")
            or (model and str(model).lower().startswith("laya:azure"))
            or (target_model and str(target_model).lower().startswith("laya:azure"))
        ):
            t0 = time.perf_counter()
            payload = {
                "state": state,
                "model": "laya:azure:t4",
                "questions": questions,
            }
            headers = {
                "Authorization": f"Bearer {settings.LAYA_API_KEY}",
                "Content-Type": "application/json",
            }
            client = self._get_client()
            resp = await client.post(settings.LAYA_REMOTE_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            duration_ms = (time.perf_counter() - t0) * 1000.0
            usage = data.get("usage", {})
            return {
                "provider": "laya_azure",
                "model": data.get("model", "laya:azure:t4:finetuned-10k"),
                "answers": data.get("answers", {}),
                "usage": {
                    "input_tokens": int(usage.get("input_tokens", 0)),
                    "output_tokens": 0,
                },
                "duration_ms": round(duration_ms, 2),
                "cost_usd": 0.0,
                "request_payload": payload,
                "raw_response": data,
            }
    async def evaluate_article_batch(
        self,
        batch_items: List[Dict[str, Any]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluates multiple articles concurrently using the Azure GPU batch endpoint."""
        effective_provider = (provider or self.provider).lower()
        if (
            effective_provider in ("laya_azure", "laya")
            or (model and str(model).lower().startswith("laya:azure"))
        ):
            endpoint = settings.LAYA_REMOTE_URL.replace("/decisions", "/decisions/batch")
            headers = {
                "Authorization": f"Bearer {settings.LAYA_API_KEY}",
                "Content-Type": "application/json",
            }
            client = self._get_client()
            payload = {
                "items": [
                    {
                        "state": item["state"],
                        "questions": item["questions"],
                        "model": "laya:azure:t4",
                    }
                    for item in batch_items
                ]
            }
            t0 = time.perf_counter()
            resp = await client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            total_dur = (time.perf_counter() - t0) * 1000.0
            per_item_dur = round(total_dur / max(1, len(batch_items)), 2)

            results = []
            for item, res in zip(batch_items, data.get("results", [])):
                usage = res.get("usage", {})
                results.append({
                    "provider": "laya_azure",
                    "model": data.get("model", "laya:azure:t4:finetuned-10k"),
                    "answers": res.get("answers", {}),
                    "usage": {
                        "input_tokens": int(usage.get("input_tokens", 0)),
                        "output_tokens": 0,
                    },
                    "duration_ms": per_item_dur,
                    "cost_usd": 0.0,
                    "request_payload": item,
                    "raw_response": res,
                })
            return results
        else:
            return await asyncio.gather(*[
                self.evaluate_article(state=it["state"], questions=it["questions"], model=model, provider=provider)
                for it in batch_items
            ])

        if effective_provider in ("laya_local",) or (model and (str(model).lower().startswith("local:"))):
            return await laya_runner.evaluate_article(
                state=state,
                questions=questions,
                model=target_model,
            )
        if self.provider == "openrouter":
            root = self.api_base.rstrip("/")
            if root.endswith("/api/v1"):
                root = root[:-len("/api/v1")]
            endpoint = f"{root}/api/alpha/decisions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": settings.OPENROUTER_HTTP_REFERER or "https://muckrack.com/curation-engine",
                "X-Title": settings.OPENROUTER_APP_TITLE,
                "X-OpenRouter-Title": settings.OPENROUTER_APP_TITLE,
                "User-Agent": settings.OPENROUTER_USER_AGENT or settings.OPENROUTER_APP_TITLE,
            }
        else:
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
        data = None
        max_retries = 3
        client = self._get_client()
        for attempt in range(max_retries + 1):
            try:
                resp = await client.post(endpoint, json=payload, headers=headers)
                if resp.status_code == 400 and "max_tokens_exceeded" in resp.text and attempt < max_retries:
                    current_state = payload.get("state", "")
                    new_len = len(current_state) // 2
                    logger.warning(
                        f"Jev {self.provider} max_tokens_exceeded on attempt {attempt + 1}. Throttling state from {len(current_state)} to {new_len} chars and retrying..."
                    )
                    payload["state"] = current_state[:new_len] + "\n\n[... truncated due to max_tokens_exceeded ...]"
                    await asyncio.sleep(1.0)
                    continue
                if resp.status_code in (500, 502, 503, 504, 520, 521, 522, 524, 429) and attempt < max_retries:
                    wait_sec = (attempt + 1) * 2.0
                    logger.warning(
                        f"Jev {self.provider} API {resp.status_code} on attempt {attempt + 1}, retrying in {wait_sec}s..."
                    )
                    await asyncio.sleep(wait_sec)
                    continue
                if resp.is_error:
                    raise RuntimeError(f"Jev {self.provider} API {resp.status_code}: {resp.text[:1000]}")
                data = resp.json()
                break
            except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                if attempt < max_retries:
                    wait_sec = (attempt + 1) * 2.0
                    logger.warning(
                        f"Jev {self.provider} network error on attempt {attempt + 1}: {net_err}, retrying in {wait_sec}s..."
                    )
                    await asyncio.sleep(wait_sec)
                    continue
                raise
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
