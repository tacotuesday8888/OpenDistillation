import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendistillation.engines import ExportRequest, ExportResult, TrainingRequest, TrainingResult


class EngineBoundaryTests(unittest.TestCase):
    def test_training_request_carries_validated_dataset_and_future_backend_settings(self):
        rows = [{"instruction": "Question", "response": "Answer", "source_chunk_id": "chunk-0001"}]

        request = TrainingRequest(
            dataset_rows=rows,
            student_model="future-small-student",
            output_dir=Path("/tmp/opendistillation-output"),
            max_steps=10,
        )

        self.assertEqual(request.dataset_rows, rows)
        self.assertEqual(request.student_model, "future-small-student")
        self.assertEqual(request.output_dir, Path("/tmp/opendistillation-output"))
        self.assertEqual(request.max_steps, 10)

    def test_training_result_does_not_require_model_artifact_to_exist_in_skeleton(self):
        result = TrainingResult(
            engine_name="future-training-engine",
            output_path=Path("/tmp/opendistillation-output/adapter"),
            notes=["placeholder boundary only"],
        )

        self.assertEqual(result.engine_name, "future-training-engine")
        self.assertFalse(result.created_model_artifact)

    def test_export_request_and_result_describe_future_local_runtime_boundary(self):
        request = ExportRequest(
            training_output_path=Path("/tmp/opendistillation-output/adapter"),
            target_format="gguf",
            local_runtime="llama.cpp",
        )
        result = ExportResult(
            engine_name="future-export-engine",
            output_path=None,
            local_command=None,
            verified=False,
            notes=["not implemented in skeleton"],
        )

        self.assertEqual(request.target_format, "gguf")
        self.assertEqual(request.local_runtime, "llama.cpp")
        self.assertFalse(result.verified)
        self.assertIsNone(result.output_path)
        self.assertIsNone(result.local_command)


if __name__ == "__main__":
    unittest.main()
