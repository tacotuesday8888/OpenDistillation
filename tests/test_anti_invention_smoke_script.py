import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class AntiInventionSmokeScriptTests(unittest.TestCase):
    def test_prepare_script_writes_manifest_and_stable_marker(self):
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "prepare_anti_invention_smoke.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--repo-root",
                    str(repo_root),
                    "--output",
                    str(manifest_path),
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": str(repo_root / "src"),
                },
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Anti-invention T4 smoke preflight", result.stdout)
        self.assertIn("Ready for GPU smoke: yes", result.stdout)
        self.assertIn("Manifest written:", result.stdout)
        marker_line = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("OD_ANTI_INVENTION_SMOKE_MANIFEST ")
        )
        marker = json.loads(marker_line.split(" ", 1)[1])
        self.assertTrue(marker["ready"])
        self.assertEqual(marker["facts"], 8)
        self.assertEqual(marker["train_rows"], 48)
        self.assertEqual(marker["eval_rows"], 8)
        self.assertEqual(marker["known_values_only_rows"], 8)
        self.assertEqual(marker["required_trained_exact_hits"], 3)
        self.assertEqual(marker["maximum_invented_value_misses"], 5)

        self.assertTrue(manifest["validation"]["ready"])
        self.assertEqual(manifest["comparison"]["question_count"], 8)
        self.assertEqual(manifest["training"]["max_steps"], 30)

    def test_prepare_script_exits_nonzero_when_contract_is_not_the_sample_smoke(self):
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "prepare_anti_invention_smoke.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            notes_path = Path(tmpdir) / "tiny.md"
            notes_path.write_text("Project codename: Glass Harbor\n", encoding="utf-8")
            manifest_path = Path(tmpdir) / "manifest.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--repo-root",
                    str(repo_root),
                    "--notes",
                    str(notes_path),
                    "--output",
                    str(manifest_path),
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": str(repo_root / "src"),
                },
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Ready for GPU smoke: no", result.stdout)
        self.assertIn("fact_count expected 8, got 1", result.stdout)
        marker_line = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("OD_ANTI_INVENTION_SMOKE_MANIFEST ")
        )
        marker = json.loads(marker_line.split(" ", 1)[1])
        self.assertFalse(marker["ready"])
        self.assertEqual(marker["facts"], 1)
        self.assertEqual(marker["train_rows"], 6)


if __name__ == "__main__":
    unittest.main()
