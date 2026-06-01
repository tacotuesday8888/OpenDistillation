import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendistillation.dataset import DatasetValidationError, rows_to_jsonl, validate_dataset, validate_dataset_row


class DatasetTests(unittest.TestCase):
    def test_validate_dataset_row_returns_required_fields_only(self):
        row = {
            "instruction": "What does OpenDistillation do?",
            "response": "It turns notes into training examples for a tiny local model.",
            "source_chunk_id": "chunk-0001",
            "debug": "not part of the public schema",
        }

        validated = validate_dataset_row(row)

        self.assertEqual(
            validated,
            {
                "instruction": "What does OpenDistillation do?",
                "response": "It turns notes into training examples for a tiny local model.",
                "source_chunk_id": "chunk-0001",
            },
        )

    def test_validate_dataset_row_rejects_missing_required_field(self):
        with self.assertRaisesRegex(DatasetValidationError, "missing required field: response"):
            validate_dataset_row({"instruction": "Question", "source_chunk_id": "chunk-0001"})

    def test_validate_dataset_row_rejects_blank_values(self):
        with self.assertRaisesRegex(DatasetValidationError, "instruction must be a non-empty string"):
            validate_dataset_row({"instruction": " ", "response": "Answer", "source_chunk_id": "chunk-0001"})

    def test_validate_dataset_reports_row_number(self):
        rows = [
            {"instruction": "Question", "response": "Answer", "source_chunk_id": "chunk-0001"},
            {"instruction": "Question", "response": "", "source_chunk_id": "chunk-0002"},
        ]

        with self.assertRaisesRegex(DatasetValidationError, "row 2"):
            validate_dataset(rows)

    def test_rows_to_jsonl_writes_valid_one_line_json_objects(self):
        rows = [
            {"instruction": "Question A", "response": "Answer A", "source_chunk_id": "chunk-0001"},
            {"instruction": "Question B", "response": "Answer B", "source_chunk_id": "chunk-0002"},
        ]

        jsonl = rows_to_jsonl(rows)

        lines = jsonl.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(jsonl.endswith("\n"))
        self.assertEqual(json.loads(lines[0])["source_chunk_id"], "chunk-0001")


if __name__ == "__main__":
    unittest.main()
