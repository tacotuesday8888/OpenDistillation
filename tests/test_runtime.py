import sys
import unittest


sys.path.insert(0, "src")

from opendistillation.runtime import (
    OPTIONAL_TRAINING_PACKAGES,
    build_pip_install_command,
    check_training_runtime,
    explain_runtime_failure,
    format_runtime_check,
)


class FakeCuda:
    def __init__(self, available, name="Fake GPU"):
        self._available = available
        self._name = name

    def is_available(self):
        return self._available

    def get_device_name(self, index):
        return self._name


class FakeTorch:
    def __init__(self, available=True, name="Fake GPU"):
        self.cuda = FakeCuda(available, name)


class RuntimeTests(unittest.TestCase):
    def test_runtime_module_does_not_import_heavy_ml_dependencies(self):
        heavy_dependencies = {"torch", "transformers", "datasets", "trl", "peft", "accelerate"}

        self.assertTrue(heavy_dependencies.isdisjoint(sys.modules))

    def test_install_command_lists_one_training_package_set(self):
        self.assertEqual(
            build_pip_install_command(),
            "python -m pip install -U torch transformers datasets trl peft accelerate",
        )
        self.assertEqual(OPTIONAL_TRAINING_PACKAGES, ("torch", "transformers", "datasets", "trl", "peft", "accelerate"))

    def test_runtime_check_reports_missing_packages_and_install_command(self):
        def missing_importer(module_name):
            raise ModuleNotFoundError(name=module_name)

        result = check_training_runtime(importer=missing_importer)
        lines = format_runtime_check(result)

        self.assertFalse(result.can_run_training)
        self.assertEqual(result.missing_packages, OPTIONAL_TRAINING_PACKAGES)
        self.assertIn("Missing optional training packages: torch, transformers, datasets, trl, peft, accelerate", lines)
        self.assertIn("Install command: python -m pip install -U torch transformers datasets trl peft accelerate", lines)

    def test_runtime_check_reports_ready_gpu_runtime(self):
        def importer(module_name):
            if module_name == "torch":
                return FakeTorch(available=True, name="NVIDIA T4")
            return object()

        result = check_training_runtime(importer=importer)
        lines = format_runtime_check(result)

        self.assertTrue(result.can_run_training)
        self.assertEqual(result.gpu_name, "NVIDIA T4")
        self.assertIn("GPU detected: NVIDIA T4", lines)
        self.assertIn("Runtime is ready for the optional short training run.", lines)

    def test_runtime_check_reports_no_cuda_runtime(self):
        def importer(module_name):
            if module_name == "torch":
                return FakeTorch(available=False)
            return object()

        result = check_training_runtime(importer=importer)
        lines = format_runtime_check(result)

        self.assertFalse(result.can_run_training)
        self.assertFalse(result.cuda_available)
        self.assertIn("No CUDA GPU detected.", lines)
        self.assertIn("In Colab, choose Runtime > Change runtime type > GPU, then rerun the runtime check.", lines)

    def test_explain_runtime_failure_mentions_out_of_memory_next_steps(self):
        lines = explain_runtime_failure(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB."))

        self.assertIn("The GPU ran out of memory.", lines)
        self.assertIn("Restart the runtime, keep RUN_TRAINING = True, and rerun from setup.", lines)
        self.assertIn("If it fails again, reduce max_steps or max_length before retrying.", lines)


if __name__ == "__main__":
    unittest.main()
