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

    def eval(self):
        self.eval_called = True

    def generate(self, input_ids, **kwargs):
        self.generate_kwargs = kwargs
        return FakeGeneratedIds(self.generated_text)


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
        return FakeModel("trained answer")


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


if __name__ == "__main__":
    unittest.main()
