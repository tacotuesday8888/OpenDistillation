import copy
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendistillation.experiment_manifest import (
    ANTI_INVENTION_MAX_STEPS,
    ANTI_INVENTION_REQUIRED_EXACT_HITS,
    ANTI_INVENTION_SMOKE_NAME,
    ANTI_INVENTION_SMOKE_SCHEMA_VERSION,
    build_anti_invention_smoke_manifest,
    format_anti_invention_smoke_report,
    validate_anti_invention_smoke_manifest,
)
from opendistillation.training import DEFAULT_STUDENT_MODEL


class ExperimentManifestTests(unittest.TestCase):
    def test_anti_invention_manifest_captures_exact_next_t4_smoke_contract(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = build_anti_invention_smoke_manifest(
            repo_root / "examples" / "sample-notes.md",
            repo_root=repo_root,
            git_state={"root": str(repo_root), "commit": "test-commit", "branch": "main", "dirty": False},
        )

        self.assertEqual(manifest["schema_version"], ANTI_INVENTION_SMOKE_SCHEMA_VERSION)
        self.assertEqual(manifest["experiment_name"], ANTI_INVENTION_SMOKE_NAME)
        self.assertTrue(manifest["validation"]["ready"])
        self.assertEqual(manifest["validation"]["errors"], [])
        self.assertEqual(manifest["source"]["extension"], ".md")
        self.assertEqual(manifest["source"]["chunk_max_chars"], 300)
        self.assertEqual(manifest["source"]["chunk_count"], 4)

        data = manifest["data"]
        self.assertEqual(data["fact_count"], 8)
        self.assertEqual(data["train_row_count"], 48)
        self.assertEqual(data["eval_row_count"], 8)
        self.assertEqual(data["train_row_styles"]["same_chunk_label_disambiguation"], 8)
        self.assertEqual(data["train_row_styles"]["known_values_only_label_value"], 8)
        self.assertTrue(data["public_schema"]["train_schema_valid"])
        self.assertTrue(data["public_schema"]["eval_schema_valid"])
        self.assertEqual(len(data["hashes"]["train_rows_sha256"]), 64)
        self.assertEqual(data["facts"][0]["label"], "Project codename")
        self.assertEqual(data["facts"][0]["expected_terms"], ["Glass Harbor"])

        quality_gate = manifest["quality_gate"]
        self.assertTrue(quality_gate["passes_required_checks"])
        self.assertEqual(quality_gate["exact_leak_count"], 0)
        self.assertEqual(quality_gate["near_leak_count"], 0)
        self.assertEqual(quality_gate["missing_expected_term_count"], 0)
        self.assertEqual(quality_gate["missing_manifest_metadata_count"], 0)

        readiness = manifest["readiness"]
        self.assertTrue(readiness["ready_for_gpu_smoke"])
        self.assertEqual(readiness["train_examples_per_fact"], 6)
        self.assertEqual(readiness["disambiguation_train_row_count"], 8)
        self.assertEqual(readiness["known_values_only_train_row_count"], 8)
        self.assertEqual(readiness["sft_preview_row_count"], 6)

        anti_invention_signal = manifest["anti_invention_signal"]
        self.assertEqual(anti_invention_signal["known_values_only_row_count"], 8)
        self.assertEqual(anti_invention_signal["warning_text_row_count"], 8)
        self.assertTrue(anti_invention_signal["all_known_values_only_rows_have_warning"])
        self.assertTrue(anti_invention_signal["all_known_values_only_rows_use_same_chunk_contrast"])

        self.assertEqual(len(manifest["sft_preview"]), 6)
        self.assertEqual(len(manifest["sft_preview_sha256"]), 64)
        first_preview = manifest["sft_preview"][0]
        self.assertEqual(first_preview["fact_id"], "fact-0001")
        self.assertIn("Project codename: Glass Harbor", first_preview["completion"])
        self.assertEqual(first_preview["expected_terms"], ["Glass Harbor"])

        training = manifest["training"]
        self.assertEqual(training["student_model"], DEFAULT_STUDENT_MODEL)
        self.assertEqual(training["max_steps"], ANTI_INVENTION_MAX_STEPS)
        self.assertEqual(training["sft_config"]["completion_only_loss"], True)
        self.assertEqual(training["lora_config"]["r"], 8)
        self.assertTrue(training["requires_gpu"])

        comparison = manifest["comparison"]
        self.assertEqual(comparison["question_count"], 8)
        self.assertEqual(len(comparison["questions_sha256"]), 64)
        self.assertEqual(comparison["questions"][0]["fact_id"], "fact-0001")
        self.assertEqual(comparison["questions"][0]["expected_terms"], ["Glass Harbor"])

        quality_rule = manifest["quality_rule"]
        self.assertEqual(quality_rule["previous_best_trained_exact_hits"], 1)
        self.assertEqual(quality_rule["required_trained_exact_hits"], ANTI_INVENTION_REQUIRED_EXACT_HITS)
        self.assertGreater(
            quality_rule["required_trained_exact_hits"],
            quality_rule["previous_best_trained_exact_hits"],
        )
        self.assertIn("Changed answers", quality_rule["failure_rule"])
        self.assertEqual(len(manifest["manifest_sha256"]), 64)

        report_lines = format_anti_invention_smoke_report(manifest)
        self.assertIn("Ready for GPU smoke: yes", report_lines)
        self.assertTrue(any("Pass condition: trained exact fact hits" in line for line in report_lines))
        self.assertTrue(any("Changed answers with wrong facts are failure evidence" in line for line in report_lines))

    def test_manifest_validation_rejects_stale_or_incomplete_contract(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = build_anti_invention_smoke_manifest(
            repo_root / "examples" / "sample-notes.md",
            repo_root=repo_root,
            git_state={"root": str(repo_root), "commit": "test-commit", "branch": "main", "dirty": False},
        )
        stale_manifest = copy.deepcopy(manifest)
        stale_manifest["readiness"]["ready_for_gpu_smoke"] = False
        stale_manifest["readiness"]["known_values_only_train_row_count"] = 0
        stale_manifest["comparison"]["questions"][0]["expected_terms"] = []

        report = validate_anti_invention_smoke_manifest(stale_manifest)

        self.assertFalse(report.ready)
        self.assertIn("readiness report is not ready for GPU smoke", report.errors)
        self.assertIn("known_values_only_train_row_count expected 8, got 0", report.errors)
        self.assertIn("comparison question 1 is missing fact identity or expected terms", report.errors)

    def test_manifest_validation_warns_when_repo_state_is_dirty(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = build_anti_invention_smoke_manifest(
            repo_root / "examples" / "sample-notes.md",
            repo_root=repo_root,
            git_state={"root": str(repo_root), "commit": "test-commit", "branch": "main", "dirty": True},
        )

        report = validate_anti_invention_smoke_manifest(manifest)

        self.assertTrue(report.ready)
        self.assertIn("repo is dirty; commit the runner before using the manifest as GPU evidence", report.warnings)


if __name__ == "__main__":
    unittest.main()
