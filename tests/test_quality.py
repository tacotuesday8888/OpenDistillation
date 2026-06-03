import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendistillation.quality import analyze_dataset_quality, format_dataset_quality_report


class DatasetQualityTests(unittest.TestCase):
    def test_quality_report_counts_rows_coverage_duplicates_and_answer_sanity(self):
        rows = [
            {
                "instruction": "What is photosynthesis?",
                "response": "Photosynthesis turns light energy into chemical energy stored as sugar.",
                "source_chunk_id": "chunk-0001",
            },
            {
                "instruction": "What is photosynthesis?",
                "response": "It turns light into sugar.",
                "source_chunk_id": "chunk-0001",
            },
            {
                "instruction": "What is the purpose of a chloroplast?",
                "response": "It captures light energy.",
                "source_chunk_id": "chunk-0002",
            },
            {
                "instruction": "What is the chloroplast purpose?",
                "response": "Yes",
                "source_chunk_id": "chunk-0002",
            },
            {
                "instruction": "Which chunk ID is invalid?",
                "response": "This row points at a chunk that was not produced by the chunker.",
                "source_chunk_id": "chunk-9999",
            },
            {
                "instruction": "What field is missing?",
                "source_chunk_id": "chunk-0002",
            },
        ]

        report = analyze_dataset_quality(
            rows,
            expected_chunk_ids=["chunk-0001", "chunk-0002", "chunk-0003"],
            min_answer_words=4,
        )

        self.assertEqual(report.row_count, 6)
        self.assertEqual(report.valid_row_count, 5)
        self.assertEqual(report.missing_field_count, 1)
        self.assertEqual(report.duplicate_question_count, 1)
        self.assertEqual(report.near_duplicate_question_count, 1)
        self.assertEqual(report.short_answer_count, 1)
        self.assertEqual(report.covered_chunk_ids, ("chunk-0001", "chunk-0002"))
        self.assertEqual(report.missing_chunk_ids, ("chunk-0003",))
        self.assertEqual(report.extra_source_chunk_ids, ("chunk-9999",))
        self.assertFalse(report.passes_required_checks)

        issue_codes = {issue.code for issue in report.issues}
        self.assertIn("missing_required_field", issue_codes)
        self.assertIn("duplicate_question", issue_codes)
        self.assertIn("near_duplicate_question", issue_codes)
        self.assertIn("short_answer", issue_codes)
        self.assertIn("missing_chunk_coverage", issue_codes)
        self.assertIn("unexpected_source_chunk_id", issue_codes)

    def test_quality_report_for_clean_dataset_has_beginner_readable_summary(self):
        rows = [
            {
                "instruction": "What should the demo prove?",
                "response": "The demo should prove that notes can become grounded training examples.",
                "source_chunk_id": "chunk-0001",
            },
            {
                "instruction": "Explain why v0 stays narrow.",
                "response": "V0 stays narrow so the notes model path can be verified before more profiles are added.",
                "source_chunk_id": "chunk-0002",
            },
        ]

        report = analyze_dataset_quality(rows, expected_chunk_ids=["chunk-0001", "chunk-0002"])
        lines = format_dataset_quality_report(report)

        self.assertTrue(report.passes_required_checks)
        self.assertIn("Dataset quality report", lines[0])
        self.assertTrue(any("Rows: 2 total, 2 schema-valid" in line for line in lines))
        self.assertTrue(any("Chunk coverage: 2/2" in line for line in lines))
        self.assertTrue(any("No required dataset-quality problems found" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
