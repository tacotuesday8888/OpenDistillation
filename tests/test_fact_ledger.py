import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendistillation.dataset import validate_dataset
from opendistillation.fact_ledger import (
    DEFAULT_FACT_TRAIN_EXAMPLES_PER_FACT,
    analyze_fact_quality_gate,
    analyze_fact_readiness,
    build_fact_comparison_rows,
    build_fact_train_eval_split,
    extract_fact_ledger,
    FactTrainEvalSplit,
    format_fact_score_report,
    format_fact_quality_report,
    format_fact_readiness_report,
    compare_fact_scores,
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

    def test_extract_fact_ledger_captures_safe_bullet_list_facts(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text=(
                    "Project codename: Glass Harbor\n"
                    "- Safety phrase - stay local first\n"
                    "* Review ritual = 4:17 PM\n"
                    "1. Accent color - ultramarine\n"
                    "- This bullet is just prose and should stay out\n"
                    "- This label has far too many words for a safe fact - skipped\n"
                    "## Heading - Not a fact\n"
                    "- Project codename: Glass Harbor\n"
                ),
                char_count=250,
                word_count=45,
            )
        ]

        facts = extract_fact_ledger(chunks)

        self.assertEqual(
            [(fact.label, fact.value, fact.fact_kind) for fact in facts],
            [
                ("Project codename", "Glass Harbor", "label_value"),
                ("Safety phrase", "stay local first", "list_pair"),
                ("Review ritual", "4:17 PM", "list_pair"),
                ("Accent color", "ultramarine", "list_pair"),
            ],
        )
        self.assertEqual([fact.fact_id for fact in facts], ["fact-0001", "fact-0002", "fact-0003", "fact-0004"])

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
        self.assertEqual(split.manifest_rows[0]["row_style"], "canonical_label_value_statement")
        self.assertEqual(split.manifest_rows[-1]["row_style"], "held_out_direct_recall")
        self.assertEqual(split.manifest_rows[0]["value"], "Glass Harbor")

    def test_build_fact_train_eval_split_defaults_to_six_label_value_rows_per_fact(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text="Project codename: Glass Harbor.",
                char_count=32,
                word_count=4,
            )
        ]

        split = build_fact_train_eval_split(extract_fact_ledger(chunks))

        self.assertEqual(DEFAULT_FACT_TRAIN_EXAMPLES_PER_FACT, 6)
        self.assertEqual(len(split.train_rows), 6)
        self.assertEqual(
            [row["row_style"] for row in split.manifest_rows if row["split"] == "train"],
            [
                "canonical_label_value_statement",
                "exact_value_from_label",
                "label_value_flashcard",
                "closed_book_label_value",
                "source_note_label_value",
                "review_quiz_label_value",
            ],
        )
        self.assertTrue(
            all(
                "Project codename: Glass Harbor" in str(row["response"])
                for row in split.manifest_rows
                if row["split"] == "train"
            )
        )

    def test_train_rows_include_same_chunk_disambiguation_for_label_value_swaps(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text="Project codename: Glass Harbor. Demo owner alias: Mira Vale.",
                char_count=61,
                word_count=8,
            )
        ]

        split = build_fact_train_eval_split(extract_fact_ledger(chunks))

        self.assertEqual(len(split.train_rows), 12)
        self.assertTrue(
            all(set(row) == {"instruction", "response", "source_chunk_id"} for row in split.train_rows)
        )
        project_rows = [
            row
            for row in split.manifest_rows
            if row["split"] == "train" and row["label"] == "Project codename"
        ]
        self.assertEqual(
            [row["row_style"] for row in project_rows],
            [
                "canonical_label_value_statement",
                "exact_value_from_label",
                "label_value_flashcard",
                "closed_book_label_value",
                "same_chunk_label_disambiguation",
                "swapped_value_correction",
            ],
        )
        disambiguation_rows = [
            row
            for row in project_rows
            if row["row_style"] in {"same_chunk_label_disambiguation", "swapped_value_correction"}
        ]
        self.assertEqual(len(disambiguation_rows), 2)
        for row in disambiguation_rows:
            self.assertEqual(row["contrast_label"], "Demo owner alias")
            self.assertEqual(row["contrast_value"], "Mira Vale")
            self.assertEqual(row["contrast_fact_id"], "fact-0002")
            self.assertIn("Project codename: Glass Harbor", str(row["response"]))
            self.assertIn("Mira Vale belongs to Demo owner alias", str(row["response"]))
            self.assertIn("Glass Harbor", str(row["response"]))
        swapped_row = disambiguation_rows[1]
        self.assertIn("Project codename: Mira Vale", str(swapped_row["instruction"]))
        self.assertIn("Incorrect swap", str(swapped_row["response"]))

    def test_build_fact_comparison_rows_adds_sidecar_metadata_without_changing_public_eval_rows(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text="Project codename: Glass Harbor. Review ritual color: ultramarine.",
                char_count=68,
                word_count=8,
            )
        ]
        split = build_fact_train_eval_split(extract_fact_ledger(chunks), train_examples_per_fact=3)

        comparison_rows = build_fact_comparison_rows(split)

        self.assertEqual(
            set(split.eval_rows[0]),
            {"instruction", "response", "source_chunk_id"},
        )
        self.assertEqual(comparison_rows[0]["instruction"], split.eval_rows[0]["instruction"])
        self.assertEqual(comparison_rows[0]["response"], split.eval_rows[0]["response"])
        self.assertEqual(comparison_rows[0]["source_chunk_id"], "chunk-0001")
        self.assertEqual(comparison_rows[0]["row_id"], "eval-000001")
        self.assertEqual(comparison_rows[0]["fact_id"], "fact-0001")
        self.assertEqual(comparison_rows[0]["label"], "Project codename")
        self.assertEqual(comparison_rows[0]["value"], "Glass Harbor")
        self.assertEqual(comparison_rows[0]["row_style"], "held_out_direct_recall")
        self.assertEqual(comparison_rows[0]["expected_terms"], ["Glass Harbor"])

    def test_build_fact_comparison_rows_matches_sidecar_by_full_public_row(self):
        eval_rows = (
            {
                "instruction": "What exact value should be recalled?",
                "response": "Exact answer: Glass Harbor.",
                "source_chunk_id": "chunk-0001",
            },
            {
                "instruction": "What exact value should be recalled?",
                "response": "Exact answer: Mira Vale.",
                "source_chunk_id": "chunk-0002",
            },
        )
        split = FactTrainEvalSplit(
            facts=(),
            train_rows=(),
            eval_rows=eval_rows,
            manifest_rows=(
                {
                    **eval_rows[1],
                    "split": "eval",
                    "row_id": "eval-000002",
                    "fact_id": "fact-0002",
                    "label": "Demo owner alias",
                    "value": "Mira Vale",
                    "row_style": "held_out_direct_recall",
                    "expected_terms": ["Mira Vale"],
                },
                {
                    **eval_rows[0],
                    "split": "eval",
                    "row_id": "eval-000001",
                    "fact_id": "fact-0001",
                    "label": "Project codename",
                    "value": "Glass Harbor",
                    "row_style": "held_out_direct_recall",
                    "expected_terms": ["Glass Harbor"],
                },
            ),
        )

        comparison_rows = build_fact_comparison_rows(split)

        self.assertEqual([row["fact_id"] for row in comparison_rows], ["fact-0001", "fact-0002"])
        self.assertEqual([row["expected_terms"] for row in comparison_rows], [["Glass Harbor"], ["Mira Vale"]])

    def test_build_fact_train_eval_split_keeps_question_wording_diverse(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text="Project codename: Glass Harbor.",
                char_count=32,
                word_count=4,
            )
        ]
        split = build_fact_train_eval_split(extract_fact_ledger(chunks), train_examples_per_fact=6)

        train_questions = [row["instruction"] for row in split.train_rows]

        self.assertEqual(len(train_questions), 6)
        self.assertEqual(len(set(train_questions)), 6)
        self.assertIn("closed-book check", split.eval_rows[0]["instruction"].lower())
        self.assertTrue(set(train_questions).isdisjoint({split.eval_rows[0]["instruction"]}))

    def test_fact_train_rows_frontload_exact_values_for_tiny_sft(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text="Project codename: Glass Harbor.",
                char_count=32,
                word_count=4,
            )
        ]
        split = build_fact_train_eval_split(extract_fact_ledger(chunks), train_examples_per_fact=3)
        responses = [row["response"] for row in split.train_rows]

        self.assertEqual(responses[0], "Exact answer: Glass Harbor. Project codename: Glass Harbor.")
        self.assertTrue(all("Glass Harbor" in response for response in responses))
        self.assertTrue(
            all(
                response.startswith("Exact answer: Glass Harbor")
                for response in responses
            )
        )
        self.assertTrue(all("Project codename: Glass Harbor" in response for response in responses))

    def test_eval_rows_use_direct_recall_instead_of_note_field_matching_wording(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text="Project codename: Glass Harbor.",
                char_count=32,
                word_count=4,
            )
        ]
        split = build_fact_train_eval_split(extract_fact_ledger(chunks), train_examples_per_fact=3)

        eval_question = split.eval_rows[0]["instruction"].lower()

        self.assertIn("closed-book", eval_question)
        self.assertIn("exact", eval_question)
        self.assertIn("project codename", eval_question)
        self.assertNotIn("which answer belongs", eval_question)
        self.assertNotIn("note field", eval_question)

    def test_fact_quality_gate_passes_clean_split_and_formats_plain_language_report(self):
        sample_path = Path(__file__).resolve().parents[1] / "examples" / "sample-notes.md"
        document = load_text_document(sample_path.name, sample_path.read_text(encoding="utf-8"))
        chunks = chunk_text(document.text, max_chars=300)
        split = build_fact_train_eval_split(extract_fact_ledger(chunks))

        report = analyze_fact_quality_gate(split)
        lines = format_fact_quality_report(report)

        self.assertTrue(report.passes_required_checks)
        self.assertEqual(report.fact_count, 8)
        self.assertEqual(report.train_row_count, 48)
        self.assertEqual(report.eval_row_count, 8)
        self.assertEqual(report.exact_leak_count, 0)
        self.assertEqual(report.near_leak_count, 0)
        self.assertEqual(report.missing_expected_term_count, 0)
        self.assertEqual(report.missing_manifest_metadata_count, 0)
        self.assertIn("Fact-ledger quality gate", lines[0])
        self.assertTrue(any("Facts: 8" in line for line in lines))
        self.assertTrue(any("Train/eval leakage: 0 exact, 0 near-duplicate" in line for line in lines))
        self.assertTrue(any("A leakage failure means" in line for line in lines))
        self.assertTrue(any("Expected terms are the exact note details" in line for line in lines))
        self.assertTrue(any("safe enough for a bounded training smoke" in line for line in lines))
        self.assertTrue(any("does not prove the model will learn" in line for line in lines))

    def test_fact_readiness_report_says_ready_only_for_clean_label_value_split(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text="Project codename: Glass Harbor.",
                char_count=32,
                word_count=4,
            )
        ]
        split = build_fact_train_eval_split(extract_fact_ledger(chunks))

        readiness = analyze_fact_readiness(split, sft_preview_row_count=6)
        lines = format_fact_readiness_report(readiness)

        self.assertTrue(readiness.ready_for_gpu_smoke)
        self.assertEqual(readiness.fact_count, 1)
        self.assertEqual(readiness.train_examples_per_fact, 6)
        self.assertEqual(readiness.label_value_fact_coverage, 1)
        self.assertEqual(readiness.contrastable_fact_count, 0)
        self.assertEqual(readiness.disambiguation_fact_coverage, 0)
        self.assertEqual(readiness.disambiguation_train_row_count, 0)
        self.assertTrue(any("Fact-ledger label/value readiness report" in line for line in lines))
        self.assertTrue(any("Train rows: 6, 6 per fact" in line for line in lines))
        self.assertTrue(any("Canonical Label: value bindings: 1/1 facts covered, 6 total rows" in line for line in lines))
        self.assertTrue(any("Label/value disambiguation rows: 0/0 contrastable facts, 0 total rows" in line for line in lines))
        self.assertTrue(any("SFT preview: 6 exact prompt/completion row(s)" in line for line in lines))
        self.assertTrue(any("Verdict: ready for one bounded GPU training smoke" in line for line in lines))
        self.assertTrue(any("does not claim model quality" in line for line in lines))

    def test_fact_readiness_report_says_sample_notes_have_disambiguation_signal(self):
        sample_path = Path(__file__).resolve().parents[1] / "examples" / "sample-notes.md"
        document = load_text_document(sample_path.name, sample_path.read_text(encoding="utf-8"))
        chunks = chunk_text(document.text, max_chars=300)
        split = build_fact_train_eval_split(extract_fact_ledger(chunks))

        readiness = analyze_fact_readiness(split, sft_preview_row_count=6)
        lines = format_fact_readiness_report(readiness)

        self.assertTrue(readiness.ready_for_gpu_smoke)
        self.assertEqual(readiness.contrastable_fact_count, 8)
        self.assertEqual(readiness.disambiguation_fact_coverage, 8)
        self.assertEqual(readiness.disambiguation_train_row_count, 16)
        self.assertTrue(
            any(
                "Label/value disambiguation rows: 8/8 contrastable facts, 16 total rows" in line
                for line in lines
            )
        )

    def test_fact_readiness_rejects_missing_disambiguation_rows_for_contrastable_facts(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text="Project codename: Glass Harbor. Demo owner alias: Mira Vale.",
                char_count=61,
                word_count=8,
            )
        ]
        split = build_fact_train_eval_split(extract_fact_ledger(chunks))
        manifest_rows = []
        for row in split.manifest_rows:
            if row.get("row_style") in {"same_chunk_label_disambiguation", "swapped_value_correction"}:
                stale_row = {
                    key: value
                    for key, value in row.items()
                    if key not in {"contrast_label", "contrast_value", "contrast_fact_id"}
                }
                manifest_rows.append({**stale_row, "row_style": "source_note_label_value"})
            else:
                manifest_rows.append(row)
        weakened_split = split.replace(manifest_rows=tuple(manifest_rows))

        readiness = analyze_fact_readiness(weakened_split, sft_preview_row_count=6)
        lines = format_fact_readiness_report(readiness)

        self.assertTrue(readiness.quality_report.passes_required_checks)
        self.assertEqual(readiness.contrastable_fact_count, 2)
        self.assertEqual(readiness.disambiguation_fact_coverage, 0)
        self.assertEqual(readiness.disambiguation_train_row_count, 0)
        self.assertFalse(readiness.ready_for_gpu_smoke)
        self.assertEqual(readiness.skip_reason, "missing_label_value_disambiguation_signal")
        self.assertTrue(any("contrastable facts need rows that distinguish nearby labels" in line for line in lines))

    def test_fact_readiness_requires_both_disambiguation_row_styles_per_contrastable_fact(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text="Project codename: Glass Harbor. Demo owner alias: Mira Vale.",
                char_count=61,
                word_count=8,
            )
        ]
        split = build_fact_train_eval_split(extract_fact_ledger(chunks))
        manifest_rows = [
            {**row, "row_style": "same_chunk_label_disambiguation"}
            if row.get("row_style") == "swapped_value_correction"
            else row
            for row in split.manifest_rows
        ]
        weakened_split = split.replace(manifest_rows=tuple(manifest_rows))

        readiness = analyze_fact_readiness(weakened_split, sft_preview_row_count=6)

        self.assertTrue(readiness.quality_report.passes_required_checks)
        self.assertEqual(readiness.contrastable_fact_count, 2)
        self.assertEqual(readiness.disambiguation_fact_coverage, 0)
        self.assertEqual(readiness.disambiguation_train_row_count, 4)
        self.assertFalse(readiness.ready_for_gpu_smoke)
        self.assertEqual(readiness.skip_reason, "missing_label_value_disambiguation_signal")

    def test_fact_readiness_rejects_stale_contrast_fact_metadata(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text="Project codename: Glass Harbor. Demo owner alias: Mira Vale.",
                char_count=61,
                word_count=8,
            )
        ]
        split = build_fact_train_eval_split(extract_fact_ledger(chunks))
        manifest_rows = [
            {**row, "contrast_fact_id": str(row["fact_id"])}
            if row.get("row_style") in {"same_chunk_label_disambiguation", "swapped_value_correction"}
            else row
            for row in split.manifest_rows
        ]
        weakened_split = split.replace(manifest_rows=tuple(manifest_rows))

        readiness = analyze_fact_readiness(weakened_split, sft_preview_row_count=6)

        self.assertTrue(readiness.quality_report.passes_required_checks)
        self.assertEqual(readiness.contrastable_fact_count, 2)
        self.assertEqual(readiness.disambiguation_fact_coverage, 0)
        self.assertEqual(readiness.disambiguation_train_row_count, 0)
        self.assertFalse(readiness.ready_for_gpu_smoke)
        self.assertEqual(readiness.skip_reason, "missing_label_value_disambiguation_signal")

    def test_fact_readiness_rejects_partial_label_value_binding_coverage(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text="Project codename: Glass Harbor.",
                char_count=32,
                word_count=4,
            )
        ]
        split = build_fact_train_eval_split(extract_fact_ledger(chunks))
        train_rows = []
        manifest_rows = []
        first_train_row_kept = False
        for manifest_row in split.manifest_rows:
            if manifest_row["split"] != "train":
                manifest_rows.append(manifest_row)
                continue

            if not first_train_row_kept:
                first_train_row_kept = True
                public_row = {
                    "instruction": manifest_row["instruction"],
                    "response": manifest_row["response"],
                    "source_chunk_id": manifest_row["source_chunk_id"],
                }
            else:
                public_row = {
                    "instruction": str(manifest_row["instruction"]),
                    "response": "Exact answer: Glass Harbor.",
                    "source_chunk_id": str(manifest_row["source_chunk_id"]),
                }
            train_rows.append(public_row)
            manifest_rows.append({**manifest_row, **public_row})
        weakened_split = split.replace(train_rows=tuple(train_rows), manifest_rows=tuple(manifest_rows))

        readiness = analyze_fact_readiness(weakened_split, sft_preview_row_count=6)
        lines = format_fact_readiness_report(readiness)

        self.assertTrue(readiness.quality_report.passes_required_checks)
        self.assertEqual(readiness.train_examples_per_fact, 6)
        self.assertEqual(readiness.label_value_fact_coverage, 1)
        self.assertEqual(readiness.label_value_train_row_count, 1)
        self.assertFalse(readiness.ready_for_gpu_smoke)
        self.assertEqual(readiness.skip_reason, "missing_label_value_training_signal")
        self.assertTrue(any("canonical Label: value" in line for line in lines))

    def test_empty_fact_ledger_reports_no_facts_without_crashing(self):
        split = build_fact_train_eval_split([])

        readiness = analyze_fact_readiness(split, sft_preview_row_count=0)
        lines = format_fact_readiness_report(readiness)

        self.assertEqual(split.facts, ())
        self.assertEqual(split.train_rows, ())
        self.assertEqual(split.eval_rows, ())
        self.assertEqual(build_fact_comparison_rows(split), ())
        self.assertFalse(readiness.ready_for_gpu_smoke)
        self.assertEqual(readiness.skip_reason, "no_fact_ledger_facts")
        self.assertTrue(any("no explicit facts were extracted" in line for line in lines))

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
                "instruction": split.train_rows[0]["instruction"].replace("exact", "precise"),
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

    def test_fact_quality_gate_flags_token_overlap_leakage(self):
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
        overlapped_split = split.replace(
            eval_rows=[
                {
                    **split.eval_rows[0],
                    "instruction": "learn this exact note fact label value pair project codename",
                }
            ]
        )

        report = analyze_fact_quality_gate(overlapped_split, near_duplicate_threshold=0.99)

        self.assertFalse(report.passes_required_checks)
        self.assertEqual(report.near_leak_count, 1)
        self.assertIn("train_eval_near_leak", {issue.code for issue in report.issues})

    def test_fact_quality_gate_fails_when_eval_row_has_no_matching_manifest_metadata(self):
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
        stale_eval_split = split.replace(
            eval_rows=[
                {
                    **split.eval_rows[0],
                    "response": "Exact answer: Mira Vale.",
                }
            ]
        )

        report = analyze_fact_quality_gate(stale_eval_split)

        self.assertFalse(report.passes_required_checks)
        self.assertEqual(report.missing_manifest_metadata_count, 1)
        self.assertIn("missing_eval_manifest_metadata", {issue.code for issue in report.issues})

    def test_fact_hit_scoring_requires_all_expected_terms(self):
        single = score_fact_answer("The project codename is Glass Harbor.", ["Glass Harbor"])
        partial = score_fact_answer("The ritual uses ultramarine.", ["4:17", "ultramarine"])
        paired = score_fact_answer("The ritual pairs 4:17 PM with ultramarine.", ["4:17", "ultramarine"])
        punctuation_case = score_fact_answer("The codename is glass harbor!", ["Glass Harbor"])
        partial_word = score_fact_answer("The codename is glass harboring.", ["Glass Harbor"])
        exact_identifier = score_fact_answer("The token is copper-lantern-47.", ["copper-lantern-47"])
        changed_identifier = score_fact_answer("The token is copper lantern 47.", ["copper-lantern-47"])
        unscored = score_fact_answer("Any answer should not pass without expected terms.", [])

        self.assertTrue(single.hit)
        self.assertFalse(partial.hit)
        self.assertTrue(paired.hit)
        self.assertTrue(punctuation_case.hit)
        self.assertFalse(partial_word.hit)
        self.assertTrue(exact_identifier.hit)
        self.assertFalse(changed_identifier.hit)
        self.assertFalse(unscored.hit)
        self.assertFalse(unscored.scored)
        self.assertEqual(partial.missing_terms, ("4:17",))
        self.assertEqual(unscored.unscored_reason, "missing_expected_terms")

    def test_score_fact_outputs_counts_hits_and_preserves_raw_answers_and_metadata(self):
        outputs = [
            {
                "question": "What is the project codename?",
                "answer": "The project codename is Glass Harbor.",
                "expected_terms": ["Glass Harbor"],
                "fact_id": "fact-0001",
                "label": "Project codename",
                "source_chunk_id": "chunk-0001",
                "row_style": "held_out_direct_recall",
            },
            {
                "question": "What is the ritual pair?",
                "answer": "The ritual uses violet.",
                "expected_terms": ["4:17", "ultramarine"],
                "fact_id": "fact-0002",
                "label": "Review ritual pair",
                "source_chunk_id": "chunk-0002",
                "row_style": "held_out_direct_recall",
            },
        ]

        summary = score_fact_outputs(outputs)

        self.assertEqual(summary.answer_count, 2)
        self.assertEqual(summary.hit_count, 1)
        self.assertEqual(summary.miss_count, 1)
        self.assertEqual(summary.items[0].question, "What is the project codename?")
        self.assertEqual(summary.items[0].fact_id, "fact-0001")
        self.assertEqual(summary.items[0].label, "Project codename")
        self.assertEqual(summary.items[0].source_chunk_id, "chunk-0001")
        self.assertEqual(summary.items[0].row_style, "held_out_direct_recall")
        self.assertEqual(summary.items[1].missing_terms, ("4:17", "ultramarine"))

    def test_fact_score_report_detects_mixed_learned_and_worse_when_total_hits_match(self):
        base_score = score_fact_outputs(
            [
                {
                    "question": "What is the project codename?",
                    "answer": "The project codename is Glass Harbor.",
                    "expected_terms": ["Glass Harbor"],
                    "fact_id": "fact-0001",
                    "label": "Project codename",
                },
                {
                    "question": "What is the review ritual color?",
                    "answer": "A generic color.",
                    "expected_terms": ["ultramarine"],
                    "fact_id": "fact-0002",
                    "label": "Review ritual color",
                },
            ]
        )
        trained_score = score_fact_outputs(
            [
                {
                    "question": "What is the project codename?",
                    "answer": "The project codename is Mira Vale.",
                    "expected_terms": ["Glass Harbor"],
                    "fact_id": "fact-0001",
                    "label": "Project codename",
                },
                {
                    "question": "What is the review ritual color?",
                    "answer": "The review ritual color is ultramarine.",
                    "expected_terms": ["ultramarine"],
                    "fact_id": "fact-0002",
                    "label": "Review ritual color",
                },
            ]
        )

        comparison = compare_fact_scores(base_score, trained_score)
        lines = format_fact_score_report(base_score, trained_score, changed_answer_count=2)

        self.assertEqual(comparison.learned_count, 1)
        self.assertEqual(comparison.worse_count, 1)
        self.assertEqual(comparison.missed_count, 0)
        self.assertEqual(comparison.unchanged_count, 0)
        self.assertTrue(any("Outcome counts: learned 1, missed 0, unchanged 0, worse 1" in line for line in lines))
        self.assertTrue(any("Learned fact: fact-0002 | Review ritual color" in line for line in lines))
        self.assertTrue(any("Worse fact: fact-0001 | Project codename" in line for line in lines))
        self.assertTrue(any("trained answer: The project codename is Mira Vale." in line for line in lines))

    def test_format_fact_score_report_calls_changed_wrong_answers_a_failure(self):
        base_score = score_fact_outputs(
            [
                {
                    "question": "What is the project codename?",
                    "answer": "Generic project guidance.",
                    "expected_terms": ["Glass Harbor"],
                },
                {
                    "question": "What is the review ritual color?",
                    "answer": "A generic color.",
                    "expected_terms": ["ultramarine"],
                },
            ]
        )
        trained_score = score_fact_outputs(
            [
                {
                    "question": "What is the project codename?",
                    "answer": "Exact answer: Mira Vale.",
                    "expected_terms": ["Glass Harbor"],
                },
                {
                    "question": "What is the review ritual color?",
                    "answer": "Exact answer: magenta.",
                    "expected_terms": ["ultramarine"],
                },
            ]
        )

        lines = format_fact_score_report(base_score, trained_score, changed_answer_count=2)

        self.assertIn("Exact fact-hit quality report", lines[0])
        self.assertTrue(any("Base exact fact hits: 0/2" in line for line in lines))
        self.assertTrue(any("Trained exact fact hits: 0/2" in line for line in lines))
        self.assertTrue(any("Changed answers: 2/2" in line for line in lines))
        self.assertTrue(any("Judgment: unchanged" in line for line in lines))
        self.assertTrue(any("Changed answers with wrong facts are still a failure" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
