import asyncio
import logging
import os
import time
from typing import Any

import laya
import torch

logger = logging.getLogger(__name__)


class LayaRunner:
    """Local System 1 decision runner using Laya non-autoregressive decision models.

    Executes typed questions (choice, score, noul) in parallel forward passes
    on Apple Silicon Metal (MPS) or CPU.
    """

    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._agents: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._eval_lock = asyncio.Lock()
        logger.info(f"Initialized LayaRunner on device: {self.device}")

    def _resolve_model_target(self, model_name: str | None) -> dict[str, Any]:
        """Resolves user-supplied model identifier to repo/subfolder/path."""
        raw = (model_name or "laya:typed-decisions").strip()

        # Check for local directory path (e.g. fine-tuned checkpoint)
        if os.path.isdir(raw) or os.path.exists(raw):
            return {
                "model_id_or_path": os.path.abspath(raw),
                "subfolder": None,
                "display_name": f"local:{os.path.basename(raw)}",
            }

        clean = raw.lower()
        finetuned_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs", "laya_finetuned"))
        if clean in (
            "laya:fine-tuned",
            "laya:finetuned",
            "fine-tuned",
            "finetuned",
            "laya-fine-tuned",
            "laya-finetuned",
            "laya:local",
            "local",
        ) and os.path.isdir(finetuned_dir):
                return {
                    "model_id_or_path": finetuned_dir,
                    "subfolder": None,
                    "display_name": "laya-finetuned",
                }
        clean = raw.lower()
        if clean in ("laya", "laya:typed-decisions", "typed-decisions"):
            return {
                "model_id_or_path": "convaiinnovations/laya",
                "subfolder": "typed-decisions",
                "display_name": "laya-typed-decisions",
            }
        elif clean in ("laya:multilingual", "multilingual"):
            return {
                "model_id_or_path": "convaiinnovations/laya",
                "subfolder": "multilingual",
                "display_name": "laya-multilingual",
            }
        elif clean in ("laya:english", "english"):
            return {
                "model_id_or_path": "convaiinnovations/laya",
                "subfolder": None,
                "display_name": "laya-english",
            }
        elif clean.startswith(("laya:local:", "local:")):
            path = raw.split(":", 2)[-1]
            return {
                "model_id_or_path": os.path.abspath(path),
                "subfolder": None,
                "display_name": f"local:{os.path.basename(path)}",
            }
        else:
            # Fallback to direct model ID or path
            return {
                "model_id_or_path": raw,
                "subfolder": None,
                "display_name": raw,
            }

    def _get_agent(self, target: dict[str, Any]) -> Any:
        cache_key = f"{target['model_id_or_path']}::{target.get('subfolder')}"
        if cache_key in self._agents:
            return self._agents[cache_key]

        logger.info(f"Loading Laya model '{target['display_name']}' on {self.device}...")
        t0 = time.perf_counter()
        agent = laya.load(
            target["model_id_or_path"],
            subfolder=target.get("subfolder"),
            device=self.device,
        )
        t_load = (time.perf_counter() - t0) * 1000.0
        logger.info(f"Loaded Laya model '{target['display_name']}' in {t_load:.1f} ms")
        # Expand context budget: 2048 sequence tokens, 384 head tokens
        agent.cfg["max_len"] = 2048
        agent.cfg["head_max_len"] = 384
        self._agents[cache_key] = agent
        return agent

    def _sanitize_questions_for_laya(
        self, questions: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Ensures question schema is strictly compatible with Laya's primitives."""
        sanitized: dict[str, dict[str, Any]] = {}
        for qid, q in questions.items():
            q_type = str(q.get("type", "choice")).lower()
            if q_type not in ("choice", "score", "noul"):
                q_type = "choice"

            ins = q.get("instructions", "")
            crit = q.get("criteria")

            # Validate criteria by type
            if q_type == "choice" and not isinstance(crit, (dict, list)):
                crit = {"valid": "valid", "invalid": "invalid"}
            elif q_type == "score" and not isinstance(crit, list):
                crit = ["level 0", "level 1", "level 2"]
            elif q_type == "noul" and not isinstance(crit, dict):
                crit = {
                    "true": "yes, the statement holds",
                    "false": "no, the statement does not hold",
                }

            sanitized[qid] = {
                "type": q_type,
                "instructions": ins,
                "criteria": crit,
            }
        return sanitized

    def _calibrate_priors(self, answers: dict[str, Any]) -> dict[str, Any]:
        """Applies empirical base-rate prior adjustments for editorial news reporting."""
        for qid, ans in answers.items():
            if qid.endswith("_prominence") and ans.get("type") == "choice":
                probs = dict(ans.get("probabilities", {}))
                if probs and "passing" in probs:
                    probs["passing"] = probs.get("passing", 0.0) * 1.5
                    probs["significant"] = probs.get("significant", 0.0) * 0.8
                    probs["primary"] = probs.get("primary", 0.0) * 0.7
                    s = sum(probs.values())
                    if s > 0:
                        probs = {k: round(v / s, 4) for k, v in probs.items()}
                    ans["probabilities"] = probs
                    ans["choice"] = max(probs, key=probs.get)

            elif qid.endswith("_sentiment") and ans.get("type") == "choice":
                probs = dict(ans.get("probabilities", {}))
                if probs and "neutral" in probs:
                    probs["neutral"] = probs.get("neutral", 0.0) * 1.6
                    probs["positive"] = probs.get("positive", 0.0) * 0.7
                    probs["negative"] = probs.get("negative", 0.0) * 0.8
                    probs["balanced"] = probs.get("balanced", 0.0) * 0.6
                    s = sum(probs.values())
                    if s > 0:
                        probs = {k: round(v / s, 4) for k, v in probs.items()}
                    ans["probabilities"] = probs
                    ans["choice"] = max(probs, key=probs.get)
        return answers

    def _predict_sync(
        self,
        agent: Any,
        state: str,
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Synchronous prediction executed inside threadpool with two-stage gating."""
        val_questions = {k: v for k, v in questions.items() if k.endswith("_valid")}

        # If there are both validation and downstream questions, execute two-stage pipeline
        if val_questions and len(val_questions) < len(questions):
            res_val = agent.predict(state, val_questions)
            answers = dict(res_val.get("answers", {}))
            input_tokens = int(res_val.get("usage", {}).get("input_tokens", 0))

            # Identify valid subject IDs
            valid_subjects = set()
            for qid, ans in answers.items():
                s_id = qid.replace("subj_", "").replace("_valid", "")
                if ans.get("noul", 0.0) >= 0.50:
                    valid_subjects.add(s_id)

            stage2_questions: dict[str, dict[str, Any]] = {}
            for qid, qdef in questions.items():
                if qid.endswith("_valid"):
                    continue
                s_id = ""
                if qid.startswith(("subj_", "tag_")):
                    parts = qid.split("_")
                    if len(parts) > 1:
                        s_id = parts[1]

                if s_id in valid_subjects:
                    stage2_questions[qid] = qdef
                else:
                    # Default passing, neutral, or false for invalid subject
                    if qid.endswith("_prominence"):
                        answers[qid] = {
                            "type": "choice",
                            "choice": "passing",
                            "probabilities": {"primary": 0.02, "significant": 0.08, "passing": 0.90},
                            "confidence": 0.92,
                        }
                    elif qid.endswith("_sentiment"):
                        answers[qid] = {
                            "type": "choice",
                            "choice": "neutral",
                            "probabilities": {"positive": 0.05, "negative": 0.05, "neutral": 0.88, "balanced": 0.02},
                            "confidence": 0.90,
                        }
                    elif qid.startswith("tag_"):
                        answers[qid] = {
                            "type": "noul",
                            "noul": 0.0,
                            "confidence": 1.0,
                        }

            # Only run Stage 2 if any subjects passed validation
            if stage2_questions:
                res_stage2 = agent.predict(state, stage2_questions)
                answers.update(res_stage2.get("answers", {}))
                input_tokens += int(res_stage2.get("usage", {}).get("input_tokens", 0))

            calibrated_answers = self._calibrate_priors(answers)
            return {
                "model": agent.cfg.get("model_name", "laya-rl-agent"),
                "answers": calibrated_answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            }

        # Otherwise evaluate all questions in one forward pass
        raw_result = agent.predict(state, questions)
        raw_result["answers"] = self._calibrate_priors(raw_result.get("answers", {}))
        return raw_result

    async def evaluate_article(
        self,
        state: str,
        questions: dict[str, dict[str, Any]],
        model: str | None = None,
    ) -> dict[str, Any]:
        """Asynchronously evaluates one article using local Laya System 1 model."""
        target = self._resolve_model_target(model)

        async with self._lock:
            agent = self._get_agent(target)

        sanitized_questions = self._sanitize_questions_for_laya(questions)

        payload = {
            "state": state,
            "model": target["display_name"],
            "questions": sanitized_questions,
        }

        start_time = time.perf_counter()

        # Protect MPS Metal command encoder from multi-threaded concurrency
        async with self._eval_lock:
            result = await asyncio.to_thread(
                self._predict_sync, agent, state, sanitized_questions
            )
            if self.device == "mps":
                torch.mps.synchronize()
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        usage = result.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        # Local inference cost is $0.00
        cost_usd = 0.0

        return {
            "provider": "laya_local",
            "model": target["display_name"],
            "answers": result.get("answers", {}),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "duration_ms": round(duration_ms, 2),
            "cost_usd": cost_usd,
            "request_payload": payload,
            "raw_response": result,
        }


laya_runner = LayaRunner()
