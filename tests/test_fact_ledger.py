import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendistillation.dataset import validate_dataset
from opendistillation.fact_ledger import (
    analyze_fact_quality_gate,
    build_fact_train_eval_split,
    extract_fact_ledger,
    format_fact_quality_report,
    score_fact_answer,
    score_fact_outputs,
)
from opendistillation.text import TextChunk, chunk_text, load_text_document


class FactLedgerTests(unittest.TestCase):
    def test_extract_fact_ledger_builds_stable_fact_cards_from_sample_notes(self):
        sample_path = Path(__file__).resolve().parents[1] / "examples" / "sample-notes.md"
        document = load_text_document(sample_path.name, sample_path.read_text(encoding="utf-8"))
        chunks = chunk_text(document.text, max_chars=300)

        facts = extract_fact_ledger(chunks)

        self.assertEqual(len(facts), 8)
        self.assertEqual(facts[0].fact_id, "fact-0001")
        self.assertEqual(facts[0].source_chunk_id, "chunk-0001")
        self.assertEqual(facts[0].label, "Project codename")
        self.assertEqual(facts[0].value, "Glass Harbor")
        self.assertEqual(facts[0].expected_terms, ("Glass Harbor",))
        self.assertEqual(facts[0].fact_kind, "label_value")
        self.assertEqual(len(facts[0].source_hash), 12)
        labels = {fact.label for fact in facts}
        self.assertIn("Notebook signal phrase", labels)
        self.assertIn("Safety boundary phrase", labels)
        self.assertIn("Local runner label", labels)
        self.assertIn("Review ritual time", labels)
        self.assertIn("Review ritual color", labels)

    def test_build_fact_train_eval_split_creates_separate_rows_and_manifest(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text="Project codename: Glass Harbor. Review ritual color: ultramarine.",
                char_count=68,
                word_count=8,
            )
        ]
        facts = extract_fact_ledger(chunks)

        split = build_fact_train_eval_split(facts, train_examples_per_fact=3)

        self.assertEqual(len(split.facts), 2)
        self.assertEqual(len(split.train_rows), 6)
        self.assertEqual(len(split.eval_rows), 2)
        self.assertEqual(len(split.manifest_rows), 8)
        validate_dataset(split.train_rows)
        validate_dataset(split.eval_rows)
        train_questions = {row["instruction"] for row in split.train_rows}
        eval_questions = {row["instruction"] for row in split.eval_rows}
        self.assertTrue(train_questions.isdisjoint(eval_questions))
        self.assertTrue(all(row["split"] in {"train", "eval"} for row in split.manifest_rows))
        self.assertTrue(all(row["expected_terms"] for row in split.manifest_rows))

    def test_fact_quality_gate_passes_clean_split_and_formats_plain_language_report(self):
        sample_path = Path(__file__).resolve().parents[1] / "examples" / "sample-notes.md"
        document = load_text_document(sample_path.name, sample_path.read_text(encoding="utf-8"))
        chunks = chunk_text(document.text, max_chars=300)
        split = build_fact_train_eval_split(extract_fact_ledger(chunks), train_examples_per_fact=3)

        report = analyze_fact_quality_gate(split)
        lines = format_fact_quality_report(report)

        self.assertTrue(report.passes_required_checks)
        self.assertEqual(report.fact_count, 8)
        self.assertEqual(report.train_row_count, 24)
        self.assertEqual(report.eval_row_count, 8)
        self.assertEqual(report.exact_leak_count, 0)
        self.assertEqual(report.near_leak_count, 0)
        self.assertEqual(report.missing_expected_term_count, 0)
        self.assertIn("Fact-ledger quality gate", lines[0])
        self.assertTrue(any("Facts: 8" in line for line in lines))
        self.assertTrue(any("Train/eval leakage: 0 exact, 0 near-duplicate" in line for line in lines))
        self.assertTrue(any("ready for a bounded training smoke" in line for line in lines))

    def test_fact_quality_gate_flags_exact_and_near_duplicate_eval_leakage(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text="Project codename: Glass Harbor.",
                char_count=32,
                word_count=4,
            )
        ]
        split = build_fact_train_eval_split(extract_fact_ledger(chunks), train_examples_per_fact=1)
        leaked_eval_rows = [
            {
                **split.eval_rows[0],
                "instruction": split.train_rows[0]["instruction"],
            },
            {
                **split.eval_rows[0],
                "instruction": split.train_rows[0]["instruction"].replace("note", "notes"),
            },
        ]
        leaked_split = split.replace(eval_rows=leaked_eval_rows)

        report = analyze_fact_quality_gate(leaked_split, near_duplicate_threshold=0.80)

        self.assertFalse(report.passes_required_checks)
        self.assertEqual(report.exact_leak_count, 1)
        self.assertEqual(report.near_leak_count, 1)
        issue_codes = {issue.code for issue in report.issues}
        self.assertIn("train_eval_exact_leak", issue_codes)
        self.assertIn("train_eval_near_leak", issue_codes)

    def test_fact_hit_scoring_requires_all_expected_terms(self):
        single = score_fact_answer("The project codename is Glass Harbor.", ["Glass Harbor"])
        partial = score_fact_answer("The ritual uses ultramarine.", ["4:17", "ultramarine"])
        paired = score_fact_answer("The ritual pairs 4:17 PM with ultramarine.", ["4:17", "ultramarine"])

        self.assertTrue(single.hit)
        self.assertFalse(partial.hit)
        self.assertTrue(paired.hit)
        self.assertEqual(partial.missing_terms, ("4:17",))

    def test_score_fact_outputs_counts_hits_and_preserves_raw_answers(self):
        outputs = [
            {
                "question": "What is the project codename?",
                "answer": "The project codename is Glass Harbor.",
                "expected_terms": ["Glass Harbor"],
            },
            {
                "question": "What is the ritual pair?",
                "answer": "The ritual uses violet.",
                "expected_terms": ["4:17", "ultramarine"],
            },
        ]

        summary = score_fact_outputs(outputs)

        self.assertEqual(summary.answer_count, 2)
        self.assertEqual(summary.hit_count, 1)
        self.assertEqual(summary.miss_count, 1)
        self.assertEqual(summary.items[0].question, "What is the project codename?")
        self.assertEqual(summary.items[1].missing_terms, ("4:17", "ultramarine"))


if __name__ == "__main__":
    unittest.main()
