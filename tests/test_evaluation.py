import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendistillation.dataset import validate_dataset
from opendistillation.quality import analyze_dataset_quality
from opendistillation.evaluation import build_sample_fact_comparison_rows
from opendistillation.teacher import MockTeacherEngine, TeacherRequest
from opendistillation.text import chunk_text, load_text_document


class SampleFactEvaluationTests(unittest.TestCase):
    def test_sample_notes_have_concrete_unusual_facts_in_stable_chunks(self):
        sample_path = Path(__file__).resolve().parents[1] / "examples" / "sample-notes.md"
        document = load_text_document(sample_path.name, sample_path.read_text(encoding="utf-8"))

        chunks = chunk_text(document.text, max_chars=300)
        combined = "\n".join(chunk.text for chunk in chunks)

        self.assertEqual([chunk.id for chunk in chunks], ["chunk-0001", "chunk-0002", "chunk-0003", "chunk-0004"])
        self.assertIn("Project codename: Glass Harbor", combined)
        self.assertIn("Notebook signal phrase: copper-lantern-47", combined)
        self.assertIn("Review ritual time: 4:17 PM", combined)
        self.assertIn("Review ritual color: ultramarine", combined)
        self.assertIn("Local runner label: llama-harbor-alpha", combined)

    def test_sample_fact_comparison_rows_are_held_out_from_mock_training_wording(self):
        sample_path = Path(__file__).resolve().parents[1] / "examples" / "sample-notes.md"
        document = load_text_document(sample_path.name, sample_path.read_text(encoding="utf-8"))
        chunks = chunk_text(document.text, max_chars=300)
        training_rows = MockTeacherEngine().generate(TeacherRequest(chunks=chunks, examples_per_chunk=6))

        held_out_rows = build_sample_fact_comparison_rows(document.filename, text=document.text)

        validate_dataset(held_out_rows)
        self.assertEqual(len(held_out_rows), 4)
        self.assertEqual(
            [row["source_chunk_id"] for row in held_out_rows],
            ["chunk-0001", "chunk-0002", "chunk-0003", "chunk-0004"],
        )
        training_questions = {row["instruction"] for row in training_rows}
        self.assertTrue(all(row["instruction"] not in training_questions for row in held_out_rows))
        self.assertTrue(any("Glass Harbor" in row["response"] for row in held_out_rows))
        self.assertTrue(any("copper-lantern-47" in row["response"] for row in held_out_rows))
        self.assertTrue(any("4:17 PM" in row["response"] and "ultramarine" in row["response"] for row in held_out_rows))
        self.assertTrue(any("llama-harbor-alpha" in row["response"] for row in held_out_rows))

    def test_sample_fact_training_rows_pass_dataset_quality_checks(self):
        sample_path = Path(__file__).resolve().parents[1] / "examples" / "sample-notes.md"
        document = load_text_document(sample_path.name, sample_path.read_text(encoding="utf-8"))
        chunks = chunk_text(document.text, max_chars=300)

        training_rows = MockTeacherEngine().generate(TeacherRequest(chunks=chunks, examples_per_chunk=6))
        report = analyze_dataset_quality(training_rows, expected_chunk_ids=[chunk.id for chunk in chunks])

        self.assertEqual(report.row_count, 24)
        self.assertEqual(report.valid_row_count, 24)
        self.assertEqual(len(report.covered_chunk_ids), 4)
        self.assertEqual(report.duplicate_question_count, 0)
        self.assertEqual(report.near_duplicate_question_count, 0)
        self.assertEqual(report.short_answer_count, 0)
        self.assertEqual(report.long_answer_count, 0)
        self.assertTrue(report.passes_required_checks)

    def test_sample_fact_comparison_rows_skip_uploaded_or_unknown_files(self):
        self.assertEqual(build_sample_fact_comparison_rows("uploaded-notes.md"), [])
        self.assertEqual(build_sample_fact_comparison_rows("notes.txt"), [])
        self.assertEqual(build_sample_fact_comparison_rows("sample-notes.md"), [])
        self.assertEqual(build_sample_fact_comparison_rows("sample-notes.md", text="Project codename: different."), [])


if __name__ == "__main__":
    unittest.main()
