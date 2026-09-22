from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import main
from config import settings
from prompt_optimizer import PromptOptimizationError, PromptOptimizer
from question_converter import ASSEMBLER_VERSION, convert_config_to_jev_questions


class PromptOptimizerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.optimizer = PromptOptimizer()
        self.optimizer.cache_dir = self.temp_dir.name
        self.snapshot = {
            "subjects": [
                {
                    "id": 1,
                    "name": "Acme",
                    "validation_prompt": "Include only substantive coverage of Acme.",
                    "prominence_prompt": "Classify Acme prominence.",
                    "sentiment_prompt": "Classify Acme sentiment.",
                    "tag_evaluations": [],
                }
            ]
        }
        self.metadata = {
            "numeric_config_id": 10,
            "config_id": "acme",
            "version_number": 3,
        }
        self.source_questions = {
            "subj_1_valid": {
                "type": "noul",
                "instructions": 'Is "Acme" relevant? Respond with true or false only.',
                "criteria": {
                    "true": 'Include only substantive "Acme" coverage for 24 clinics.',
                    "false": "Exclude unrelated coverage.",
                },
            }
        }
        self.optimized_questions = {
            "subj_1_valid": {
                "type": "noul",
                "instructions": 'Is "Acme" relevant?',
                "criteria": {
                    "true": 'Include only substantive "Acme": 24 clinics.',
                    "false": "Exclude unrelated.",
                },
            }
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_cache_reuse_corruption_and_identity_invalidation(self) -> None:
        response = {
            "text": json.dumps({"questions": self.optimized_questions}),
            "input_tokens": 100,
            "output_tokens": 20,
        }
        call = AsyncMock(return_value=response)
        with (
            patch.object(settings, "GEMINI_API_KEY", "test-key"),
            patch.object(self.optimizer, "_call_gemini", call),
        ):
            first_questions, first_metadata = await self.optimizer.optimize_questions(
                self.source_questions, self.snapshot, self.metadata, ASSEMBLER_VERSION
            )
            second_questions, second_metadata = await self.optimizer.optimize_questions(
                self.source_questions, self.snapshot, self.metadata, ASSEMBLER_VERSION
            )

            self.assertEqual(first_questions, second_questions)
            self.assertFalse(first_metadata["cached"])
            self.assertTrue(second_metadata["cached"])
            self.assertEqual(call.await_count, 1)
            self.assertGreater(first_metadata["original_question_chars"], first_metadata["optimized_question_chars"])

            cache_path = os.path.join(
                self.optimizer.cache_dir, f'{first_metadata["cache_key"]}.json'
            )
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                cache_file.write("{corrupt")
            rebuilt_questions, rebuilt_metadata = await self.optimizer.optimize_questions(
                self.source_questions, self.snapshot, self.metadata, ASSEMBLER_VERSION
            )
            self.assertEqual(rebuilt_questions, self.optimized_questions)
            self.assertFalse(rebuilt_metadata["cached"])
            self.assertEqual(call.await_count, 2)

            newer_metadata = {**self.metadata, "version_number": 4}
            await self.optimizer.optimize_questions(
                self.source_questions, self.snapshot, newer_metadata, ASSEMBLER_VERSION
            )
            self.assertEqual(call.await_count, 3)

    def test_rejects_every_unsafe_output_class(self) -> None:
        cases: dict[str, dict[str, dict]] = {}

        cases["dropped question"] = {}

        changed_type = copy.deepcopy(self.optimized_questions)
        changed_type["subj_1_valid"]["type"] = "choice"
        cases["changed type"] = changed_type

        changed_label = copy.deepcopy(self.optimized_questions)
        changed_label["subj_1_valid"]["criteria"] = {
            "true": changed_label["subj_1_valid"]["criteria"]["true"],
            "no": changed_label["subj_1_valid"]["criteria"]["false"],
        }
        cases["changed label"] = changed_label

        lost_literal = copy.deepcopy(self.optimized_questions)
        lost_literal["subj_1_valid"]["instructions"] = "Is the company relevant?"
        lost_literal["subj_1_valid"]["criteria"]["true"] = "Include only substantive coverage: 24 clinics."
        cases["lost protected literal"] = lost_literal

        output_prose = copy.deepcopy(self.optimized_questions)
        output_prose["subj_1_valid"]["instructions"] = 'Is "Acme" relevant? Return exactly JSON.'
        cases["output-format prose"] = output_prose

        cases["not smaller"] = copy.deepcopy(self.source_questions)

        for label, candidate in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(PromptOptimizationError):
                    self.optimizer._validate_question_map(
                        candidate, self.source_questions, self.snapshot
                    )

    async def test_defective_question_is_preserved_while_safe_question_is_optimized(self) -> None:
        snapshot = {
            "subjects": [
                {
                    "id": 1,
                    "name": "Acme",
                    "entity_definition": "Include \"Acme\" but exclude \"Acme\".",
                    "validation_prompt": "",
                    "sentiment_prompt": "Classify sentiment.",
                    "tag_evaluations": [],
                }
            ]
        }
        source = {
            "subj_1_valid": {
                "type": "noul",
                "instructions": 'Is "Acme" valid?',
                "criteria": {"true": 'Include "Acme".', "false": 'Exclude "Acme".'},
            },
            "subj_1_sentiment": {
                "type": "choice",
                "instructions": 'Choose sentiment for "Acme". Output only one word.',
                "criteria": {"positive": "Favorable coverage.", "negative": "Adverse coverage."},
            },
        }
        optimized_safe = {
            "subj_1_sentiment": {
                "type": "choice",
                "instructions": 'Sentiment for "Acme"?',
                "criteria": {"positive": "Favorable.", "negative": "Adverse."},
            }
        }
        call = AsyncMock(
            return_value={
                "text": json.dumps({"questions": optimized_safe}),
                "input_tokens": 10,
                "output_tokens": 5,
            }
        )
        with (
            patch.object(settings, "GEMINI_API_KEY", "test-key"),
            patch.object(self.optimizer, "_call_gemini", call),
        ):
            optimized, metadata = await self.optimizer.optimize_questions(
                source, snapshot, self.metadata, ASSEMBLER_VERSION
            )

        self.assertEqual(optimized["subj_1_valid"], source["subj_1_valid"])
        self.assertEqual(optimized["subj_1_sentiment"], optimized_safe["subj_1_sentiment"])
        self.assertEqual(
            metadata["diagnostics"],
            [
                {
                    "question_id": "subj_1_valid",
                    "code": "contradictory_policy",
                    "message": "Source applies both include and exclude semantics to the same literal.",
                }
            ],
        )
        optimizer_prompt = call.await_args_list[0].args[0]
        sent_questions = json.loads(
            optimizer_prompt.split("ASSEMBLED JEV QUESTIONS:\n", 1)[1]
        )
        self.assertEqual(list(sent_questions), ["subj_1_sentiment"])

    def test_detects_configurator_and_tag_name_anomalies(self) -> None:
        snapshot = {
            "subjects": [
                {
                    "id": 1,
                    "name": "Acme",
                    "validation_prompt": "Please specify the criteria for this subject.",
                    "tag_evaluations": [
                        {
                            "tag_id": "partnership",
                            "tag_name": "Partnership",
                            "evaluation_type": "llm",
                            "prompt_text": 'Evaluate the tag "Layoffs".',
                        },
                        {
                            "tag_id": "layoffs",
                            "tag_name": "Layoffs",
                            "evaluation_type": "llm",
                            "prompt_text": "Evaluate Layoffs.",
                        },
                    ],
                }
            ]
        }
        questions = {
            "subj_1_valid": self.source_questions["subj_1_valid"],
            "tag_1_partnership": self.source_questions["subj_1_valid"],
        }
        diagnostics = self.optimizer._detect_anomalies(questions, snapshot)
        self.assertEqual(
            [(item["question_id"], item["code"]) for item in diagnostics],
            [
                ("subj_1_valid", "configurator_facing"),
                ("tag_1_partnership", "tag_name_mismatch"),
            ],
        )

    async def test_preview_and_run_assembly_share_the_cached_artifact(self) -> None:
        response = {
            "text": json.dumps({"questions": self.optimized_questions}),
            "input_tokens": 10,
            "output_tokens": 5,
        }
        call = AsyncMock(return_value=response)
        with (
            patch.object(settings, "GEMINI_API_KEY", "test-key"),
            patch.object(self.optimizer, "_call_gemini", call),
            patch.object(main, "prompt_optimizer", self.optimizer),
            patch.object(main, "convert_config_to_jev_questions", return_value=self.source_questions),
        ):
            preview_questions, preview_metadata = await main._assemble_questions(
                self.snapshot,
                self.metadata,
                optimize_prompts=True,
                model="jev-latest",
                provider="typesafe",
            )
            run_questions, run_metadata = await main._assemble_questions(
                self.snapshot,
                self.metadata,
                optimize_prompts=True,
                model="jev-latest",
                provider="typesafe",
            )

        self.assertEqual(preview_questions, run_questions)
        self.assertFalse(preview_metadata["cached"])
        self.assertTrue(run_metadata["cached"])
        self.assertEqual(call.await_count, 1)

    def test_laya_optimization_is_rejected_and_canonical_questions_stay_raw(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            main._validate_prompt_optimization_request(
                True, "laya:azure:t4", "laya_azure"
            )
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("only for Jev", caught.exception.detail)


class QuestionAssemblyTests(unittest.TestCase):
    def test_every_subject_and_only_llm_tags_are_assembled(self) -> None:
        snapshot = {
            "subjects": [
                {
                    "id": 7,
                    "name": "Alpha",
                    "validation_prompt": "Alpha validation",
                    "prominence_prompt": "Alpha prominence",
                    "sentiment_prompt": "Alpha sentiment",
                    "tag_evaluations": [
                        {
                            "tag_id": "llm",
                            "tag_name": "Risk",
                            "evaluation_type": "llm",
                            "prompt_text": "Risk criteria",
                        },
                        {
                            "tag_id": "bool",
                            "tag_name": "Alias",
                            "evaluation_type": "boolean",
                            "boolean_expression": '"Alpha"',
                        },
                    ],
                },
                {
                    "id": 8,
                    "name": "Beta",
                    "validation_prompt": "Beta validation",
                    "prominence_prompt": "Beta prominence",
                    "sentiment_prompt": "Beta sentiment",
                    "tag_evaluations": [],
                },
            ]
        }

        questions = convert_config_to_jev_questions(snapshot)

        self.assertEqual(
            list(questions),
            [
                "subj_7_valid",
                "subj_7_prominence",
                "subj_7_sentiment",
                "tag_7_llm",
                "subj_8_valid",
                "subj_8_prominence",
                "subj_8_sentiment",
            ],
        )
        self.assertEqual(questions["subj_7_valid"]["type"], "noul")
        self.assertEqual(
            list(questions["subj_7_prominence"]["criteria"]),
            ["primary", "significant", "passing"],
        )
        self.assertEqual(
            list(questions["subj_7_sentiment"]["criteria"]),
            ["positive", "negative", "neutral", "balanced"],
        )
        self.assertEqual(questions["tag_7_llm"]["type"], "noul")
        self.assertNotIn("tag_7_bool", questions)


if __name__ == "__main__":
    unittest.main()
