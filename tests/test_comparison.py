import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendistillation.comparison import (
    BeforeAfterComparisonEngine,
    ComparisonConfigurationError,
    ComparisonDependencyError,
    build_comparison_request,
)
from opendistillation.dataset import DatasetValidationError
from opendistillation.engines import TrainingResult
from opendistillation.fact_ledger import score_fact_outputs
from opendistillation.training import DEFAULT_STUDENT_MODEL, SFTLoRAConfig


class FakeGeneratedIds:
    def __init__(self, text):
        self.text = text

    def __getitem__(self, key):
        return self


class FakeInputIds:
    shape = (1, 4)

    def to(self, device):
        self.device = device
        return self


class FakeModel:
    def __init__(self, generated_text):
        self.generated_text = generated_text
        self.device = "cpu"
        self.eval_called = False
        self.generate_kwargs = None
        self.adapter_enabled = False

    def eval(self):
        self.eval_called = True

    def generate(self, input_ids, **kwargs):
        self.generate_kwargs = kwargs
        if self.adapter_enabled:
            return FakeGeneratedIds("trained answer")
        return FakeGeneratedIds(self.generated_text)


class FakePeftModel:
    def __init__(self, base_model):
        self.base_model = base_model
        self.device = base_model.device
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def disable_adapter(self):
        peft_model = self

        class DisabledAdapter:
            def __enter__(self):
                self.previous_state = peft_model.base_model.adapter_enabled
                peft_model.base_model.adapter_enabled = False

            def __exit__(self, exc_type, exc, traceback):
                peft_model.base_model.adapter_enabled = self.previous_state
                return False

        return DisabledAdapter()

    def generate(self, input_ids, **kwargs):
        return self.base_model.generate(input_ids, **kwargs)


class FakeTokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 151645
    pad_token = None
    pad_token_id = None

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, *, add_generation_prompt, return_tensors):
        self.messages = messages
        self.add_generation_prompt = add_generation_prompt
        self.return_tensors = return_tensors
        return FakeInputIds()

    def batch_decode(self, generated_ids, *, skip_special_tokens):
        return [generated_ids.text]


class FakeTokenizerFactory:
    tokenizer = FakeTokenizer()
    calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append((model_id, kwargs))
        return cls.tokenizer


class FakeAutoModelFactory:
    calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append((model_id, kwargs))
        return FakeModel("base answer")


class FakePeftModelFactory:
    calls = []

    @classmethod
    def from_pretrained(cls, base_model, adapter_path, **kwargs):
        cls.calls.append((base_model, adapter_path, kwargs))
        base_model.adapter_enabled = True
        return FakePeftModel(base_model)


class FakeTorch:
    class no_grad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False


class FakeModule:
    def __init__(self, **attributes):
        self.__dict__.update(attributes)


class ComparisonPathTests(unittest.TestCase):
    def test_comparison_module_does_not_import_heavy_ml_dependencies(self):
        heavy_dependencies = {"torch", "transformers", "peft", "accelerate"}

        self.assertTrue(heavy_dependencies.isdisjoint(sys.modules))

    def test_build_comparison_request_uses_first_dataset_question_and_adapter_output(self):
        rows = [
            {
                "instruction": " What is the main point? ",
                "response": " The notes say to keep the demo narrow. ",
                "source_chunk_id": " chunk-0001 ",
            }
        ]
        training_result = TrainingResult(
            engine_name="trl-sfttrainer-peft-lora",
            output_path=Path("outputs/notes-lora/adapter"),
            created_model_artifact=True,
        )

        request = build_comparison_request(rows, training_result, config=SFTLoRAConfig(), max_new_tokens=24)

        self.assertEqual(request.question, "What is the main point?")
        self.assertEqual(request.reference_response, "The notes say to keep the demo narrow.")
        self.assertEqual(request.source_chunk_id, "chunk-0001")
        self.assertEqual(request.student_model, DEFAULT_STUDENT_MODEL)
        self.assertEqual(request.adapter_path, Path("outputs/notes-lora/adapter"))
        self.assertEqual(request.max_new_tokens, 24)

    def test_build_comparison_request_selects_bounded_question_set(self):
        rows = [
            {
                "instruction": f"Question {index}",
                "response": f"Reference answer {index} from the notes.",
                "source_chunk_id": f"chunk-000{index}",
            }
            for index in range(1, 5)
        ]
        training_result = TrainingResult(
            engine_name="trl-sfttrainer-peft-lora",
            output_path=Path("outputs/notes-lora/adapter"),
            created_model_artifact=True,
        )

        request = build_comparison_request(rows, training_result, max_examples=3)

        self.assertEqual(len(request.examples), 3)
        self.assertEqual([example.question for example in request.examples], ["Question 1", "Question 2", "Question 3"])
        self.assertEqual(request.question, "Question 1")

    def test_build_comparison_request_prefers_distinct_source_chunks(self):
        rows = [
            {
                "instruction": "Question 1 from chunk one",
                "response": "Reference answer 1 from the notes.",
                "source_chunk_id": "chunk-0001",
            },
            {
                "instruction": "Question 2 from chunk one",
                "response": "Reference answer 2 from the notes.",
                "source_chunk_id": "chunk-0001",
            },
            {
                "instruction": "Question 3 from chunk two",
                "response": "Reference answer 3 from the notes.",
                "source_chunk_id": "chunk-0002",
            },
            {
                "instruction": "Question 4 from chunk three",
                "response": "Reference answer 4 from the notes.",
                "source_chunk_id": "chunk-0003",
            },
        ]
        training_result = TrainingResult(
            engine_name="trl-sfttrainer-peft-lora",
            output_path=Path("outputs/notes-lora/adapter"),
            created_model_artifact=True,
        )

        request = build_comparison_request(rows, training_result, max_examples=3)

        self.assertEqual(
            [example.question for example in request.examples],
            ["Question 1 from chunk one", "Question 3 from chunk two", "Question 4 from chunk three"],
        )
        self.assertEqual(
            [example.source_chunk_id for example in request.examples],
            ["chunk-0001", "chunk-0002", "chunk-0003"],
        )

    def test_build_comparison_request_preserves_fact_metadata_after_reordering(self):
        rows = [
            {
                "instruction": "Question 1 from chunk one",
                "response": "Reference answer 1 from the notes.",
                "source_chunk_id": "chunk-0001",
                "row_id": "eval-000001",
                "fact_id": "fact-0001",
                "label": "Project codename",
                "row_style": "held_out_direct_recall",
                "expected_terms": ["Glass Harbor"],
            },
            {
                "instruction": "Question 2 from chunk one",
                "response": "Reference answer 2 from the notes.",
                "source_chunk_id": "chunk-0001",
                "row_id": "eval-000002",
                "fact_id": "fact-0002",
                "label": "Demo owner alias",
                "row_style": "held_out_direct_recall",
                "expected_terms": ["Mira Vale"],
            },
            {
                "instruction": "Question 3 from chunk two",
                "response": "Reference answer 3 from the notes.",
                "source_chunk_id": "chunk-0002",
                "row_id": "eval-000003",
                "fact_id": "fact-0003",
                "label": "Notebook signal phrase",
                "row_style": "held_out_direct_recall",
                "expected_terms": ["copper-lantern-47"],
            },
            {
                "instruction": "Question 4 from chunk three",
                "response": "Reference answer 4 from the notes.",
                "source_chunk_id": "chunk-0003",
                "row_id": "eval-000004",
                "fact_id": "fact-0004",
                "label": "Review ritual color",
                "row_style": "held_out_direct_recall",
                "expected_terms": ["ultramarine"],
            },
        ]
        training_result = TrainingResult(
            engine_name="trl-sfttrainer-peft-lora",
            output_path=Path("outputs/notes-lora/adapter"),
            created_model_artifact=True,
        )

        request = build_comparison_request(rows, training_result, max_examples=3)

        self.assertEqual([example.question for example in request.examples], [
            "Question 1 from chunk one",
            "Question 3 from chunk two",
            "Question 4 from chunk three",
        ])
        self.assertEqual([example.fact_id for example in request.examples], ["fact-0001", "fact-0003", "fact-0004"])
        self.assertEqual([example.label for example in request.examples], [
            "Project codename",
            "Notebook signal phrase",
            "Review ritual color",
        ])
        self.assertEqual([example.expected_terms for example in request.examples], [
            ("Glass Harbor",),
            ("copper-lantern-47",),
            ("ultramarine",),
        ])
        self.assertTrue(all(example.row_style == "held_out_direct_recall" for example in request.examples))

    def test_build_comparison_request_rejects_non_positive_example_limit(self):
        training_result = TrainingResult(
            engine_name="trl-sfttrainer-peft-lora",
            output_path=Path("outputs/notes-lora/adapter"),
            created_model_artifact=True,
        )

        with self.assertRaisesRegex(ComparisonConfigurationError, "max_examples must be at least 1"):
            build_comparison_request(
                [
                    {
                        "instruction": "Question",
                        "response": "Answer",
                        "source_chunk_id": "chunk-0001",
                    }
                ],
                training_result,
                max_examples=0,
            )

    def test_build_comparison_request_rejects_invalid_dataset_rows(self):
        training_result = TrainingResult(
            engine_name="trl-sfttrainer-peft-lora",
            output_path=Path("outputs/notes-lora/adapter"),
            created_model_artifact=True,
        )

        with self.assertRaises(DatasetValidationError):
            build_comparison_request([{"instruction": "Question"}], training_result)

    def test_build_comparison_request_rejects_training_result_without_artifact(self):
        training_result = TrainingResult(
            engine_name="trl-sfttrainer-peft-lora",
            output_path=Path("outputs/notes-lora/adapter"),
            created_model_artifact=False,
        )

        with self.assertRaises(ComparisonConfigurationError):
            build_comparison_request(
                [
                    {
                        "instruction": "Question",
                        "response": "Answer",
                        "source_chunk_id": "chunk-0001",
                    }
                ],
                training_result,
            )

    def test_engine_plan_describes_before_after_run_without_loading_models(self):
        training_result = TrainingResult(
            engine_name="trl-sfttrainer-peft-lora",
            output_path=Path("outputs/notes-lora/adapter"),
            created_model_artifact=True,
        )
        request = build_comparison_request(
            [
                {
                    "instruction": "Question",
                    "response": "Answer",
                    "source_chunk_id": "chunk-0001",
                }
            ],
            training_result,
        )
        engine = BeforeAfterComparisonEngine()

        plan = engine.describe(request)

        self.assertEqual(plan["engine"], "transformers-peft-before-after")
        self.assertEqual(plan["student_model"], DEFAULT_STUDENT_MODEL)
        self.assertEqual(plan["question"], "Question")
        self.assertEqual(plan["adapter_path"], "outputs/notes-lora/adapter")
        self.assertIn("transformers", plan["dependencies"])
        self.assertIn("peft", plan["dependencies"])

    def test_compare_rejects_missing_adapter_path_before_loading_dependencies(self):
        training_result = TrainingResult(
            engine_name="trl-sfttrainer-peft-lora",
            output_path=Path("outputs/notes-lora/missing-adapter"),
            created_model_artifact=True,
        )
        request = build_comparison_request(
            [
                {
                    "instruction": "Question",
                    "response": "Answer",
                    "source_chunk_id": "chunk-0001",
                }
            ],
            training_result,
        )

        with patch("opendistillation.comparison.import_module") as import_module:
            with self.assertRaises(ComparisonConfigurationError):
                BeforeAfterComparisonEngine().compare(request)

        import_module.assert_not_called()

    def test_missing_comparison_dependencies_error_lists_optional_packages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            adapter_path.mkdir()
            request = build_comparison_request(
                [
                    {
                        "instruction": "Question",
                        "response": "Answer",
                        "source_chunk_id": "chunk-0001",
                    }
                ],
                TrainingResult(
                    engine_name="trl-sfttrainer-peft-lora",
                    output_path=adapter_path,
                    created_model_artifact=True,
                ),
            )

            def fail_import(module_name):
                raise ModuleNotFoundError(name=module_name)

            with patch("opendistillation.comparison.import_module", side_effect=fail_import):
                with self.assertRaises(ComparisonDependencyError) as context:
                    BeforeAfterComparisonEngine().compare(request)

        message = str(context.exception)
        for dependency in ("torch", "transformers", "peft", "accelerate"):
            self.assertIn(dependency, message)
        self.assertIn("transformers<5", message)
        self.assertNotIn("pip install -U torch", message)

    def test_comparison_dependency_error_reports_installed_package_import_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            adapter_path.mkdir()
            request = build_comparison_request(
                [
                    {
                        "instruction": "Question",
                        "response": "Answer",
                        "source_chunk_id": "chunk-0001",
                    }
                ],
                TrainingResult(
                    engine_name="trl-sfttrainer-peft-lora",
                    output_path=adapter_path,
                    created_model_artifact=True,
                ),
            )

            def fail_import(module_name):
                if module_name == "peft":
                    raise RuntimeError("operator torchvision::nms does not exist")
                return object()

            with patch("opendistillation.comparison.import_module", side_effect=fail_import):
                with self.assertRaises(ComparisonDependencyError) as context:
                    BeforeAfterComparisonEngine().compare(request)

        message = str(context.exception)
        self.assertIn("peft: RuntimeError: operator torchvision::nms does not exist", message)
        self.assertIn("without upgrading Colab's preinstalled torch", message)

    def test_compare_generates_base_and_adapter_answers_with_fake_dependencies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            adapter_path.mkdir()
            request = build_comparison_request(
                [
                    {
                        "instruction": "Question from generated data",
                        "response": "Reference answer from mock teacher",
                        "source_chunk_id": "chunk-0001",
                    }
                ],
                TrainingResult(
                    engine_name="trl-sfttrainer-peft-lora",
                    output_path=adapter_path,
                    created_model_artifact=True,
                ),
                max_new_tokens=12,
            )

            def fake_import(module_name):
                if module_name == "torch":
                    return FakeTorch
                if module_name == "transformers":
                    return FakeModule(
                        AutoModelForCausalLM=FakeAutoModelFactory,
                        AutoTokenizer=FakeTokenizerFactory,
                    )
                if module_name == "peft":
                    return FakeModule(PeftModel=FakePeftModelFactory)
                if module_name == "accelerate":
                    return FakeModule()
                raise ModuleNotFoundError(name=module_name)

            with patch("opendistillation.comparison.import_module", side_effect=fake_import):
                result = BeforeAfterComparisonEngine().compare(request)

        self.assertEqual(result.question, "Question from generated data")
        self.assertEqual(result.reference_response, "Reference answer from mock teacher")
        self.assertEqual(result.base_answer, "base answer")
        self.assertEqual(result.trained_answer, "trained answer")
        self.assertEqual(FakeTokenizerFactory.tokenizer.messages[0]["content"], "Question from generated data")
        self.assertEqual(FakeAutoModelFactory.calls[-1][0], DEFAULT_STUDENT_MODEL)
        self.assertEqual(FakePeftModelFactory.calls[-1][1], str(adapter_path))

    def test_compare_disables_adapter_when_generating_base_answer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            adapter_path.mkdir()
            request = build_comparison_request(
                [
                    {
                        "instruction": "Question from generated data",
                        "response": "Reference answer from mock teacher",
                        "source_chunk_id": "chunk-0001",
                    }
                ],
                TrainingResult(
                    engine_name="trl-sfttrainer-peft-lora",
                    output_path=adapter_path,
                    created_model_artifact=True,
                ),
            )

            def fake_import(module_name):
                if module_name == "torch":
                    return FakeTorch
                if module_name == "transformers":
                    return FakeModule(
                        AutoModelForCausalLM=FakeAutoModelFactory,
                        AutoTokenizer=FakeTokenizerFactory,
                    )
                if module_name == "peft":
                    return FakeModule(PeftModel=FakePeftModelFactory)
                if module_name == "accelerate":
                    return FakeModule()
                raise ModuleNotFoundError(name=module_name)

            with patch("opendistillation.comparison.import_module", side_effect=fake_import):
                result = BeforeAfterComparisonEngine().compare(request)

        self.assertEqual(result.base_answer, "base answer")
        self.assertEqual(result.trained_answer, "trained answer")

    def test_compare_generates_bounded_multi_question_quality_items(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            adapter_path.mkdir()
            request = build_comparison_request(
                [
                    {
                        "instruction": "Question one",
                        "response": "Reference answer one from the notes.",
                        "source_chunk_id": "chunk-0001",
                    },
                    {
                        "instruction": "Question two",
                        "response": "Reference answer two from the notes.",
                        "source_chunk_id": "chunk-0002",
                    },
                ],
                TrainingResult(
                    engine_name="trl-sfttrainer-peft-lora",
                    output_path=adapter_path,
                    created_model_artifact=True,
                ),
                max_examples=2,
            )

            def fake_import(module_name):
                if module_name == "torch":
                    return FakeTorch
                if module_name == "transformers":
                    return FakeModule(
                        AutoModelForCausalLM=FakeAutoModelFactory,
                        AutoTokenizer=FakeTokenizerFactory,
                    )
                if module_name == "peft":
                    return FakeModule(PeftModel=FakePeftModelFactory)
                if module_name == "accelerate":
                    return FakeModule()
                raise ModuleNotFoundError(name=module_name)

            with patch("opendistillation.comparison.import_module", side_effect=fake_import):
                result = BeforeAfterComparisonEngine().compare(request)

        self.assertEqual(len(result.items), 2)
        self.assertEqual([item.question for item in result.items], ["Question one", "Question two"])
        self.assertEqual(result.items[0].base_answer, "base answer")
        self.assertEqual(result.items[0].trained_answer, "trained answer")
        self.assertIsInstance(result.items[0].base_reference_overlap, float)
        self.assertIsInstance(result.items[0].trained_reference_overlap, float)
        self.assertGreaterEqual(result.items[0].base_reference_overlap, 0.0)
        self.assertLessEqual(result.items[0].base_reference_overlap, 1.0)
        self.assertEqual(result.question, "Question one")

    def test_compare_result_fact_outputs_keep_expected_terms_with_reordered_items(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            adapter_path.mkdir()
            request = build_comparison_request(
                [
                    {
                        "instruction": "Question 1 from chunk one",
                        "response": "Reference answer 1 from the notes.",
                        "source_chunk_id": "chunk-0001",
                        "fact_id": "fact-0001",
                        "label": "Project codename",
                        "expected_terms": ["base answer"],
                        "row_style": "held_out_direct_recall",
                    },
                    {
                        "instruction": "Question 2 from chunk one",
                        "response": "Reference answer 2 from the notes.",
                        "source_chunk_id": "chunk-0001",
                        "fact_id": "fact-0002",
                        "label": "Demo owner alias",
                        "expected_terms": ["Mira Vale"],
                        "row_style": "held_out_direct_recall",
                    },
                    {
                        "instruction": "Question 3 from chunk two",
                        "response": "Reference answer 3 from the notes.",
                        "source_chunk_id": "chunk-0002",
                        "fact_id": "fact-0003",
                        "label": "Notebook signal phrase",
                        "expected_terms": ["trained answer"],
                        "row_style": "held_out_direct_recall",
                    },
                ],
                TrainingResult(
                    engine_name="trl-sfttrainer-peft-lora",
                    output_path=adapter_path,
                    created_model_artifact=True,
                ),
                max_examples=2,
            )

            def fake_import(module_name):
                if module_name == "torch":
                    return FakeTorch
                if module_name == "transformers":
                    return FakeModule(
                        AutoModelForCausalLM=FakeAutoModelFactory,
                        AutoTokenizer=FakeTokenizerFactory,
                    )
                if module_name == "peft":
                    return FakeModule(PeftModel=FakePeftModelFactory)
                if module_name == "accelerate":
                    return FakeModule()
                raise ModuleNotFoundError(name=module_name)

            with patch("opendistillation.comparison.import_module", side_effect=fake_import):
                result = BeforeAfterComparisonEngine().compare(request)

        self.assertEqual([item.fact_id for item in result.items], ["fact-0001", "fact-0003"])
        self.assertEqual([item.expected_terms for item in result.items], [("base answer",), ("trained answer",)])

        trained_score = score_fact_outputs(result.fact_outputs("trained"))

        self.assertEqual(trained_score.answer_count, 2)
        self.assertEqual(trained_score.hit_count, 1)
        self.assertFalse(trained_score.items[0].hit)
        self.assertTrue(trained_score.items[1].hit)


if __name__ == "__main__":
    unittest.main()
