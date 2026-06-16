import contextlib
import io
import json
import os
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "opendistillation_v0_demo.ipynb"


class NotebookSkeletonTests(unittest.TestCase):
    def _run_notebook_code_cells(self):
        repo_root = NOTEBOOK_PATH.parents[1]
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        state = {"__name__": "__main__"}
        output = io.StringIO()
        old_cwd = Path.cwd()
        try:
            os.chdir(repo_root)
            with contextlib.redirect_stdout(output):
                for index, cell in enumerate(notebook["cells"], start=1):
                    if cell.get("cell_type") != "code":
                        continue
                    code = "".join(cell.get("source", []))
                    # This executes only the repository's trusted pure-Python notebook cells,
                    # not user-uploaded notes or arbitrary runtime input.
                    exec(compile(code, f"notebook-cell-{index}", "exec"), state)
        finally:
            os.chdir(old_cwd)
        return state, output.getvalue()

    def test_notebook_has_notes_flow_sections_and_no_saved_outputs(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("A personal model factory for the AI PC and AI phone era.", sources)
        self.assertIn("OPEN_DISTILLATION_REPO_URL", sources)
        self.assertIn("https://github.com/tacotuesday8888/OpenDistillation.git", sources)
        self.assertIn("STATUS_LOG_PATH", sources)
        self.assertIn("opendistillation_status.jsonl", sources)
        self.assertIn("record_status", sources)
        self.assertIn("OD_STATUS", sources)
        self.assertIn("cat /tmp/opendistillation_status.jsonl", sources)
        self.assertIn("not model quality", sources)
        self.assertIn("fact-ledger quality gate", sources)
        self.assertIn('"clone", "--depth", "1"', sources)
        self.assertIn("Upload or load a TXT/MD file", sources)
        self.assertIn("USE_SAMPLE_NOTES = True", sources)
        self.assertIn("Set `USE_SAMPLE_NOTES = False`", sources)
        self.assertIn("Chunk the document", sources)
        self.assertIn("Generate mock training examples", sources)
        self.assertIn("RUN_REAL_TEACHER = False", sources)
        self.assertIn("HuggingFaceLocalTeacherEngine", sources)
        self.assertIn("DEFAULT_REAL_TEACHER_MODEL", sources)
        self.assertIn("explain_teacher_failure", sources)
        self.assertIn('"teacher"', sources)
        self.assertIn('"configured"', sources)
        self.assertIn('"succeeded"', sources)
        self.assertIn("Optional short student fine-tuning", sources)
        self.assertIn("INSTALL_TRAINING_DEPS = False", sources)
        self.assertIn('"install"', sources)
        self.assertIn("OPTIONAL_TRAINING_INSTALL_PACKAGES", sources)
        self.assertIn("OPTIONAL_TRAINING_PACKAGES", sources)
        self.assertIn("preinstalled GPU `torch`", sources)
        self.assertIn("RUN_TRAINING = False", sources)
        self.assertIn('"training"', sources)
        self.assertIn('"runtime_check_finished"', sources)
        self.assertIn("check_training_runtime", sources)
        self.assertIn("format_runtime_check", sources)
        self.assertIn("explain_runtime_failure", sources)
        self.assertIn("extract_fact_ledger", sources)
        self.assertIn("build_fact_train_eval_split", sources)
        self.assertIn("analyze_fact_quality_gate", sources)
        self.assertIn("format_fact_quality_report", sources)
        self.assertIn("format_fact_readiness_report", sources)
        self.assertIn("build_fact_comparison_rows", sources)
        self.assertIn("build_sft_preview_rows", sources)
        self.assertIn("format_sft_preview_report", sources)
        self.assertIn("format_fact_score_report", sources)
        self.assertIn("DEFAULT_FACT_TRAIN_EXAMPLES_PER_FACT", sources)
        self.assertIn("analyze_fact_readiness", sources)
        self.assertIn("score_fact_outputs", sources)
        self.assertIn('"fact_ledger"', sources)
        self.assertIn("fact_training_rows", sources)
        self.assertIn("held_out_fact_eval_rows", sources)
        self.assertIn("fact_readiness_skip_reason", sources)
        self.assertIn("no_fact_ledger_facts", sources)
        self.assertIn("fact_readiness_report.skip_reason", sources)
        self.assertIn("fact_readiness_not_ready", sources)
        self.assertIn("if RUN_TRAINING and fact_training_rows and not fact_readiness_report.ready_for_gpu_smoke", sources)
        self.assertIn("no explicit facts were extracted", sources)
        self.assertIn("format_fact_readiness_report(fact_readiness_report)", sources)
        self.assertIn("DEFAULT_FACT_TRAIN_EXAMPLES_PER_FACT", sources)
        self.assertNotIn("train_examples_per_fact=3", sources)
        self.assertIn("fact-ledger train rows", sources)
        self.assertIn("held-out fact-ledger eval questions", sources)
        self.assertIn("SFT preview before any training", sources)
        self.assertIn("Exact fact-hit report", sources)
        self.assertIn("Before/after comparison", sources)
        self.assertIn("build_comparison_request", sources)
        self.assertIn('"comparison"', sources)
        self.assertIn("training_result is None", sources)
        self.assertIn("Manual Colab smoke-test checklist", sources)
        self.assertIn("Export placeholder", sources)
        self.assertTrue(all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code"))
        self.assertTrue(all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code"))

    def test_safe_notebook_path_builds_gpu_ready_fact_rows_without_training(self):
        state, output = self._run_notebook_code_cells()

        self.assertFalse(state["INSTALL_TRAINING_DEPS"])
        self.assertTrue(state["USE_SAMPLE_NOTES"])
        self.assertFalse(state["RUN_REAL_TEACHER"])
        self.assertFalse(state["RUN_TRAINING"])
        self.assertIsNone(state["training_result"])

        self.assertEqual(len(state["chunks"]), 4)
        self.assertEqual(len(state["rows"]), 24)
        self.assertEqual(len(state["fact_ledger"]), 8)
        self.assertEqual(len(state["fact_training_rows"]), 48)
        self.assertEqual(len(state["held_out_fact_eval_rows"]), 8)
        self.assertEqual(state["training_row_source"], "fact-ledger train rows")
        self.assertEqual(len(state["sft_preview_rows"]), 6)

        fact_quality_report = state["fact_quality_report"]
        self.assertTrue(fact_quality_report.passes_required_checks)
        self.assertEqual(fact_quality_report.exact_leak_count, 0)
        self.assertEqual(fact_quality_report.near_leak_count, 0)
        self.assertEqual(fact_quality_report.missing_expected_term_count, 0)

        fact_readiness_report = state["fact_readiness_report"]
        self.assertTrue(fact_readiness_report.ready_for_gpu_smoke)
        self.assertEqual(fact_readiness_report.skip_reason, "")
        self.assertEqual(fact_readiness_report.train_examples_per_fact, 6)
        self.assertEqual(fact_readiness_report.label_value_fact_coverage, 8)
        self.assertEqual(fact_readiness_report.label_value_train_row_count, 48)
        self.assertEqual(fact_readiness_report.contrastable_fact_count, 8)
        self.assertEqual(fact_readiness_report.disambiguation_fact_coverage, 8)
        self.assertEqual(fact_readiness_report.disambiguation_train_row_count, 16)

        self.assertIn("SFT preview before any training:", output)
        self.assertIn("Exact SFT preview", output)
        self.assertIn("Label/value disambiguation rows: 8/8 contrastable facts, 16 total rows", output)
        self.assertIn("Verdict: ready for one bounded GPU training smoke", output)
        self.assertIn("Training skipped. Set RUN_TRAINING = True", output)
        self.assertIn('OD_STATUS stage=training status=skipped', output)


if __name__ == "__main__":
    unittest.main()
