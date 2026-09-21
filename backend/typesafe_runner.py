import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from config import settings
from laya_runner import laya_runner

logger = logging.getLogger(__name__)

# System 1 engines answer over HTTP with the same decision contract, so the only
# thing that varies between them is where to post and how to read the envelope.
LAYA_AZURE = "laya_azure"
LAYA_DATABRICKS = "laya_databricks"
LAYA_LOCAL = "laya_local"


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

    # ------------------------------------------------------------------
    # Engine routing
    # ------------------------------------------------------------------

    def _resolve_engine(
        self, model: Optional[str], provider: Optional[str]
    ) -> tuple[str, str]:
        """Returns (engine, target_model) for a UI selection.

        The frontend sends the model string as the source of truth; `provider` is
        only a hint. Both are checked so an explicit provider still wins when the
        caller omits the model.
        """
        hint = (provider or self.provider).lower()
        target = model or (settings.LAYA_MODEL if hint.startswith("laya") else self.model)
        name = str(target or "").lower()

        # Checked first: the downloaded-weights variant is a prefix of the
        # served variant, so ordering decides which branch wins.
        if hint == LAYA_LOCAL or name.startswith(("local:", "laya:local", "laya:databricks:local")):
            return LAYA_LOCAL, target
        if hint == LAYA_DATABRICKS or name.startswith("laya:databricks"):
            return LAYA_DATABRICKS, target
        if hint in (LAYA_AZURE, "laya") or name.startswith("laya:azure"):
            return LAYA_AZURE, target
        return "jev", target

    def _laya_transport(self, engine: str) -> dict[str, Any]:
        """Endpoint, auth and wire model name for a remote System 1 engine."""
        if engine == LAYA_DATABRICKS:
            host = settings.DATABRICKS_HOST.rstrip("/")
            endpoint = settings.LAYA_DATABRICKS_ENDPOINT
            if not host or not endpoint:
                raise RuntimeError(
                    "Databricks engine selected but DATABRICKS_HOST or "
                    "LAYA_DATABRICKS_ENDPOINT is unset."
                )
            if not settings.DATABRICKS_TOKEN:
                raise RuntimeError("Databricks engine selected but DATABRICKS_TOKEN is unset.")
            return {
                "url": f"{host}/serving-endpoints/{endpoint}/invocations",
                "api_key": settings.DATABRICKS_TOKEN,
                "batch_url": f"{host}/serving-endpoints/{endpoint}/invocations",
                "wire_model": "laya:databricks",
                "default_model": f"laya:databricks:{endpoint}",
                # Databricks Model Serving speaks the MLflow scoring protocol
                # rather than the bespoke shape the Azure service exposes.
                "protocol": "mlflow",
            }
        return {
            "url": settings.LAYA_REMOTE_URL,
            "api_key": settings.LAYA_API_KEY,
            "batch_url": settings.LAYA_REMOTE_URL.replace("/decisions", "/decisions/batch"),
            "wire_model": "laya:azure:t4",
            "default_model": "laya:azure:t4:finetuned-10k",
            "protocol": "native",
        }

    @staticmethod
    def _unwrap_mlflow(data: Any) -> Dict[str, Any]:
        """Pulls the decision envelope out of an MLflow scoring response."""
        if isinstance(data, dict):
            preds = data.get("predictions", data)
        else:
            preds = data
        if isinstance(preds, list):
            preds = preds[0] if preds else {}
        if isinstance(preds, str):
            preds = json.loads(preds)
        return preds if isinstance(preds, dict) else {}

    # ------------------------------------------------------------------
    # Single article
    # ------------------------------------------------------------------

    async def evaluate_article(
        self,
        state: str,
        questions: Dict[str, Dict[str, Any]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        engine, target_model = self._resolve_engine(model, provider)

        if engine == LAYA_LOCAL:
            return await laya_runner.evaluate_article(
                state=state,
                questions=questions,
                model=target_model,
            )

        if engine in (LAYA_AZURE, LAYA_DATABRICKS):
            return await self._evaluate_laya_remote(engine, state, questions)

        return await self._evaluate_jev(state, questions, target_model)

    async def _evaluate_laya_remote(
        self,
        engine: str,
        state: str,
        questions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        transport = self._laya_transport(engine)
        t0 = time.perf_counter()

        if transport["protocol"] == "mlflow":
            payload = {
                "dataframe_records": [
                    {"state": state, "questions": json.dumps(questions)}
                ]
            }
        else:
            payload = {
                "state": state,
                "model": transport["wire_model"],
                "questions": questions,
            }

        headers = {
            "Authorization": f"Bearer {transport['api_key']}",
            "Content-Type": "application/json",
        }
        client = self._get_client()
        resp = await client.post(transport["url"], json=payload, headers=headers)
        resp.raise_for_status()
        raw = resp.json()
        data = self._unwrap_mlflow(raw) if transport["protocol"] == "mlflow" else raw

        duration_ms = (time.perf_counter() - t0) * 1000.0
        usage = data.get("usage", {}) or {}
        return {
            "provider": engine,
            "model": data.get("model", transport["default_model"]),
            "answers": data.get("answers", {}),
            "usage": {
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": 0,
            },
            "duration_ms": round(duration_ms, 2),
            "cost_usd": 0.0,
            "request_payload": payload,
            "raw_response": raw,
        }

    async def _evaluate_jev(
        self,
        state: str,
        questions: Dict[str, Dict[str, Any]],
        target_model: str,
    ) -> Dict[str, Any]:
        if self.provider == "openrouter":
            root = self.api_base.rstrip("/")
            if root.endswith("/api/v1"):
                root = root[: -len("/api/v1")]
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
            # The direct TypeSafe provider has never worked: the original code
            # assigned `endpoint` only on the OpenRouter branch, so this path
            # raised NameError. Every plausible URL under TYPESAFE_API_BASE
            # returns 404, so there is nothing to point it at. Fail with a
            # message that says so rather than posting into the void.
            raise RuntimeError(
                f"JEV_PROVIDER={self.provider!r} has no working endpoint. "
                "Set JEV_PROVIDER=openrouter, or supply the real direct-API "
                "path before using this provider."
            )

        payload = {
            "state": state,
            "model": target_model,
            "questions": questions,
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
                        f"Jev {self.provider} max_tokens_exceeded on attempt {attempt + 1}. "
                        f"Throttling state from {len(current_state)} to {new_len} chars and retrying..."
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

        if data is None:
            raise RuntimeError(f"Jev {self.provider} API exhausted {max_retries} retries without a response.")

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
                "output_tokens": output_tokens,
            },
            "duration_ms": round(duration_ms, 2),
            "cost_usd": cost_usd,
            # Observability: the exact wire payload and full raw response.
            "request_payload": payload,
            "raw_response": data,
        }

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    async def evaluate_article_batch(
        self,
        batch_items: List[Dict[str, Any]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluates multiple articles through a GPU batch endpoint where one exists."""
        engine, _ = self._resolve_engine(model, provider)

        if engine in (LAYA_AZURE, LAYA_DATABRICKS):
            return await self._evaluate_laya_remote_batch(engine, batch_items)

        return await asyncio.gather(*[
            self.evaluate_article(
                state=it["state"], questions=it["questions"], model=model, provider=provider
            )
            for it in batch_items
        ])

    async def _evaluate_laya_remote_batch(
        self,
        engine: str,
        batch_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        transport = self._laya_transport(engine)
        headers = {
            "Authorization": f"Bearer {transport['api_key']}",
            "Content-Type": "application/json",
        }

        if transport["protocol"] == "mlflow":
            payload = {
                "dataframe_records": [
                    {"state": item["state"], "questions": json.dumps(item["questions"])}
                    for item in batch_items
                ]
            }
        else:
            payload = {
                "items": [
                    {
                        "state": item["state"],
                        "questions": item["questions"],
                        "model": transport["wire_model"],
                    }
                    for item in batch_items
                ]
            }

        client = self._get_client()
        t0 = time.perf_counter()
        resp = await client.post(transport["batch_url"], json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        total_dur = (time.perf_counter() - t0) * 1000.0
        per_item_dur = round(total_dur / max(1, len(batch_items)), 2)

        if transport["protocol"] == "mlflow":
            preds = data.get("predictions", []) if isinstance(data, dict) else data
            entries = [p if isinstance(p, dict) else json.loads(p) for p in (preds or [])]
            model_name = transport["default_model"]
        else:
            entries = data.get("results", [])
            model_name = data.get("model", transport["default_model"])

        results = []
        for item, res in zip(batch_items, entries):
            usage = res.get("usage", {}) or {}
            results.append({
                "provider": engine,
                "model": res.get("model", model_name),
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


typesafe_runner = TypeSafeRunner()
