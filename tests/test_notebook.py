import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "opendistillation_v0_demo.ipynb"


class NotebookSkeletonTests(unittest.TestCase):
    def test_notebook_has_notes_flow_sections_and_no_saved_outputs(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("A personal model factory for the AI PC and AI phone era.", sources)
        self.assertIn("OPEN_DISTILLATION_REPO_URL", sources)
        self.assertIn("https://github.com/tacotuesday8888/OpenDistillation.git", sources)
        self.assertIn('"clone", "--depth", "1"', sources)
        self.assertIn("Upload or load a TXT/MD file", sources)
        self.assertIn("Chunk the document", sources)
        self.assertIn("Generate mock training examples", sources)
        self.assertIn("Optional short student fine-tuning", sources)
        self.assertIn("INSTALL_TRAINING_DEPS = False", sources)
        self.assertIn("OPTIONAL_TRAINING_INSTALL_PACKAGES", sources)
        self.assertIn("OPTIONAL_TRAINING_PACKAGES", sources)
        self.assertIn("preinstalled GPU `torch`", sources)
        self.assertIn("RUN_TRAINING = False", sources)
        self.assertIn("check_training_runtime", sources)
        self.assertIn("format_runtime_check", sources)
        self.assertIn("explain_runtime_failure", sources)
        self.assertIn("Before/after comparison", sources)
        self.assertIn("build_comparison_request", sources)
        self.assertIn("training_result is None", sources)
        self.assertIn("Manual Colab smoke-test checklist", sources)
        self.assertIn("Export placeholder", sources)
        self.assertTrue(all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code"))
        self.assertTrue(all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code"))


if __name__ == "__main__":
    unittest.main()
