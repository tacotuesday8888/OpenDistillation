import unittest
from pathlib import Path


GITIGNORE = Path(__file__).resolve().parents[1] / ".gitignore"


class GitignoreTests(unittest.TestCase):
    def test_generated_training_outputs_and_caches_are_ignored(self):
        patterns = set(GITIGNORE.read_text(encoding="utf-8").splitlines())

        expected_patterns = {
            "models/",
            "checkpoints/",
            "datasets/generated/",
            "outputs/",
            "adapters/",
            "hf_cache/",
            "huggingface_cache/",
            ".cache/",
            "wandb/",
            "mlruns/",
            "events.out.tfevents*",
            "trainer_state.json",
            "*.safetensors",
            "*.bin",
            "*.gguf",
            "*.pt",
            "*.pth",
            "*.ckpt",
            "*.onnx",
        }
        self.assertTrue(expected_patterns.issubset(patterns))


if __name__ == "__main__":
    unittest.main()
