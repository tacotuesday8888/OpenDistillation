import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendistillation.dataset import DatasetValidationError
from opendistillation.training import (
    DEFAULT_STUDENT_MODEL,
    SFTLoRAConfig,
    SFTLoRATrainingEngine,
    TrainingConfigurationError,
    TrainingDependencyError,
    build_lora_config_kwargs,
    build_sft_config_kwargs,
    build_sft_preview_rows,
    build_training_request,
    format_sft_preview_report,
    format_sft_rows,
)


class TrainingPathTests(unittest.TestCase):
    def test_training_module_does_not_import_heavy_ml_dependencies(self):
        heavy_dependencies = {"torch", "transformers", "trl", "peft", "datasets", "accelerate"}

        self.assertTrue(heavy_dependencies.isdisjoint(sys.modules))

    def test_default_config_selects_one_bounded_qwen_lora_path(self):
        config = SFTLoRAConfig()

        self.assertEqual(config.student_model, DEFAULT_STUDENT_MODEL)
        self.assertEqual(config.student_model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(config.max_steps, 10)
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.lora_r, 8)
        self.assertEqual(config.eos_token, "<|im_end|>")
        self.assertFalse(config.use_bitsandbytes)
        self.assertFalse(config.use_unsloth)

    def test_build_training_request_validates_rows_and_uses_config_defaults(self):
        rows = [
            {
                "instruction": " What does OpenDistillation do? ",
                "response": " It turns notes into a small training dataset. ",
                "source_chunk_id": " chunk-0001 ",
            }
        ]
        config = SFTLoRAConfig(max_steps=7)

        request = build_training_request(rows, Path("/tmp/opendistillation-run"), config=config)

        self.assertEqual(request.student_model, DEFAULT_STUDENT_MODEL)
        self.assertEqual(request.output_dir, Path("/tmp/opendistillation-run"))
        self.assertEqual(request.max_steps, 7)
        self.assertEqual(
            request.dataset_rows,
            [
                {
                    "instruction": "What does OpenDistillation do?",
                    "response": "It turns notes into a small training dataset.",
                    "source_chunk_id": "chunk-0001",
                }
            ],
        )

    def test_build_training_request_rejects_invalid_dataset_rows(self):
        rows = [{"instruction": "Question", "source_chunk_id": "chunk-0001"}]

        with self.assertRaises(DatasetValidationError):
            build_training_request(rows, Path("/tmp/opendistillation-run"))

    def test_format_sft_rows_converts_dataset_schema_to_conversation_pairs(self):
        rows = [
            {
                "instruction": "Answer from the notes only.",
                "response": "The notes say to keep v0 narrow.",
                "source_chunk_id": "chunk-0001",
            }
        ]

        formatted = format_sft_rows(rows)

        self.assertEqual(
            formatted,
            [
                {
                    "prompt": [{"role": "user", "content": "Answer from the notes only."}],
                    "completion": [{"role": "assistant", "content": "The notes say to keep v0 narrow."}],
                }
            ],
        )

    def test_sft_preview_rows_show_exact_prompt_completion_and_fact_metadata(self):
        rows = [
            {
                "instruction": "Answer only with the exact saved value for project codename.",
                "response": "Glass Harbor",
                "source_chunk_id": "chunk-0001",
            }
        ]
        manifest_rows = [
            {
                "instruction": "Answer only with the exact saved value for project codename.",
                "response": "Glass Harbor",
                "source_chunk_id": "chunk-0001",
                "row_id": "train-000001",
                "fact_id": "fact-0001",
                "label": "Project codename",
                "row_style": "exact_value_answer_only",
                "expected_terms": ["Glass Harbor"],
            }
        ]

        preview = build_sft_preview_rows(rows, manifest_rows=manifest_rows)
        lines = format_sft_preview_report(preview)

        self.assertEqual(len(preview), 1)
        self.assertEqual(preview[0].prompt, "Answer only with the exact saved value for project codename.")
        self.assertEqual(preview[0].completion, "Glass Harbor")
        self.assertEqual(preview[0].fact_id, "fact-0001")
        self.assertEqual(preview[0].label, "Project codename")
        self.assertEqual(preview[0].row_style, "exact_value_answer_only")
        self.assertEqual(preview[0].expected_terms, ("Glass Harbor",))
        self.assertIn("Exact SFT preview", lines[0])
        self.assertTrue(any("Prompt: Answer only with the exact saved value" in line for line in lines))
        self.assertTrue(any("Completion: Glass Harbor" in line for line in lines))

    def test_engine_plan_describes_colab_gpu_training_without_running_it(self):
        rows = [
            {
                "instruction": "Question",
                "response": "Answer",
                "source_chunk_id": "chunk-0001",
            }
        ]
        request = build_training_request(rows, Path("outputs/notes-lora"), max_steps=3)
        engine = SFTLoRATrainingEngine()

        plan = engine.describe(request)

        self.assertEqual(plan["engine"], "trl-sfttrainer-peft-lora")
        self.assertEqual(plan["student_model"], DEFAULT_STUDENT_MODEL)
        self.assertEqual(plan["output_dir"], "outputs/notes-lora")
        self.assertEqual(plan["max_steps"], 3)
        self.assertTrue(plan["requires_gpu"])
        self.assertIn("trl", plan["dependencies"])
        self.assertIn("peft", plan["dependencies"])
        self.assertIn("transformers", plan["dependencies"])

    def test_training_kwargs_are_short_and_do_not_push_to_hub(self):
        rows = [
            {
                "instruction": "Question",
                "response": "Answer",
                "source_chunk_id": "chunk-0001",
            }
        ]
        config = SFTLoRAConfig(max_steps=4, learning_rate=0.0002)
        request = build_training_request(rows, Path("outputs/notes-lora"), config=config)

        sft_kwargs = build_sft_config_kwargs(request, config)
        lora_kwargs = build_lora_config_kwargs(config)

        self.assertEqual(sft_kwargs["max_steps"], 4)
        self.assertEqual(sft_kwargs["learning_rate"], 0.0002)
        self.assertEqual(sft_kwargs["per_device_train_batch_size"], 1)
        self.assertEqual(sft_kwargs["gradient_accumulation_steps"], 4)
        self.assertEqual(sft_kwargs["max_length"], 512)
        self.assertEqual(sft_kwargs["eos_token"], "<|im_end|>")
        self.assertEqual(sft_kwargs["completion_only_loss"], True)
        self.assertNotIn("assistant_only_loss", sft_kwargs)
        self.assertEqual(sft_kwargs["push_to_hub"], False)
        self.assertEqual(sft_kwargs["report_to"], "none")
        self.assertEqual(lora_kwargs["r"], 8)
        self.assertEqual(lora_kwargs["lora_alpha"], 16)
        self.assertEqual(lora_kwargs["bias"], "none")
        self.assertEqual(lora_kwargs["task_type"], "CAUSAL_LM")

    def test_engine_rejects_non_default_student_model_to_keep_v0_single_path(self):
        rows = [
            {
                "instruction": "Question",
                "response": "Answer",
                "source_chunk_id": "chunk-0001",
            }
        ]
        config = SFTLoRAConfig(student_model="another/model")
        request = build_training_request(rows, Path("outputs/notes-lora"), config=config)
        engine = SFTLoRATrainingEngine(config)

        with self.assertRaises(TrainingConfigurationError):
            engine.train(request)

    def test_missing_training_dependencies_error_lists_full_optional_package_set(self):
        rows = [
            {
                "instruction": "Question",
                "response": "Answer",
                "source_chunk_id": "chunk-0001",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            request = build_training_request(rows, Path(tmpdir) / "notes-lora")
            engine = SFTLoRATrainingEngine()

            def fail_import(module_name):
                raise ModuleNotFoundError(name=module_name)

            with patch("opendistillation.training.import_module", side_effect=fail_import):
                with self.assertRaises(TrainingDependencyError) as context:
                    engine.train(request)

        message = str(context.exception)
        for dependency in ("torch", "transformers", "datasets", "trl", "peft", "accelerate"):
            self.assertIn(dependency, message)
        self.assertIn("transformers<5", message)
        self.assertNotIn("pip install -U torch", message)

    def test_training_dependency_error_reports_installed_package_import_failure(self):
        rows = [
            {
                "instruction": "Question",
                "response": "Answer",
                "source_chunk_id": "chunk-0001",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            request = build_training_request(rows, Path(tmpdir) / "notes-lora")
            engine = SFTLoRATrainingEngine()

            def fail_import(module_name):
                if module_name == "peft":
                    raise RuntimeError("operator torchvision::nms does not exist")
                return object()

            with patch("opendistillation.training.import_module", side_effect=fail_import):
                with self.assertRaises(TrainingDependencyError) as context:
                    engine.train(request)

        message = str(context.exception)
        self.assertIn("peft: RuntimeError: operator torchvision::nms does not exist", message)
        self.assertIn("without upgrading Colab's preinstalled torch", message)


if __name__ == "__main__":
    unittest.main()
