"""Runtime readiness helpers for the optional Colab training path.

These helpers keep heavyweight ML packages lazy. Importing this module should
not import torch, Transformers, TRL, PEFT, datasets, or Accelerate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any


OPTIONAL_TRAINING_PACKAGES = ("torch", "transformers", "datasets", "trl", "peft", "accelerate")
OPTIONAL_TRAINING_INSTALL_PACKAGES = (
    "transformers<5",
    "datasets",
    "trl<1",
    "peft<0.19",
    "accelerate",
)
OPTIONAL_COMPARISON_PACKAGES = ("torch", "transformers", "peft", "accelerate")


@dataclass(frozen=True)
class RuntimeCheck:
    """A plain-English readiness summary for optional Colab training."""

    missing_packages: tuple[str, ...]
    import_errors: dict[str, str]
    cuda_available: bool
    gpu_name: str | None
    install_command: str

    @property
    def can_run_training(self) -> bool:
        return not self.missing_packages and not self.import_errors and self.cuda_available


def build_pip_install_command(packages: Sequence[str] = OPTIONAL_TRAINING_INSTALL_PACKAGES) -> str:
    """Return a stable install command suitable for Colab and local notebooks."""

    return "python -m pip install -U " + " ".join(_quote_package(package) for package in packages)


def _quote_package(package: str) -> str:
    if any(character in package for character in "<>=!~"):
        return "'" + package + "'"
    return package


def check_training_runtime(
    *,
    importer: Callable[[str], Any] = import_module,
) -> RuntimeCheck:
    """Check optional packages and CUDA without downloading models."""

    missing: list[str] = []
    import_errors: dict[str, str] = {}
    modules: dict[str, Any] = {}
    for package in OPTIONAL_TRAINING_PACKAGES:
        try:
            modules[package] = importer(package)
        except ModuleNotFoundError as exc:
            missing_name = exc.name or package
            if missing_name == package:
                if missing_name not in missing:
                    missing.append(missing_name)
            else:
                import_errors[package] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            import_errors[package] = f"{type(exc).__name__}: {exc}"

    torch_module = modules.get("torch")
    cuda_available = False
    gpu_name = None
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None:
        try:
            cuda_available = bool(cuda.is_available())
            if cuda_available:
                gpu_name = str(cuda.get_device_name(0))
        except Exception:
            cuda_available = False
            gpu_name = None

    return RuntimeCheck(
        missing_packages=tuple(missing),
        import_errors=import_errors,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        install_command=build_pip_install_command(),
    )


def format_runtime_check(check: RuntimeCheck) -> list[str]:
    """Format a runtime check for notebook printing."""

    lines: list[str] = []
    if check.missing_packages:
        lines.append("Missing optional training packages: " + ", ".join(check.missing_packages))
        lines.append("Install command: " + check.install_command)
    else:
        lines.append("Optional training packages are importable.")

    if check.import_errors:
        lines.append("Optional training package import failures:")
        for package, error in check.import_errors.items():
            lines.append(f"- {package}: {error}")
        if any("torchvision::nms" in error for error in check.import_errors.values()):
            lines.append(
                "This can happen when Colab's torch/torchvision packages are mismatched after upgrading torch."
            )
            lines.append("Restart the Colab runtime and use the notebook install command without upgrading torch.")

    if check.cuda_available:
        lines.append("GPU detected: " + (check.gpu_name or "CUDA device"))
    else:
        lines.append("No CUDA GPU detected.")
        lines.append("In Colab, choose Runtime > Change runtime type > GPU, then rerun the runtime check.")

    if check.can_run_training:
        lines.append("Runtime is ready for the optional short training run.")
    else:
        lines.append("Training remains skipped until packages and GPU runtime are ready.")

    return lines


def explain_runtime_failure(exc: BaseException) -> list[str]:
    """Return beginner-readable next steps for common optional runtime failures."""

    message = str(exc)
    lowered = message.lower()
    lines = ["Optional training/comparison failed."]

    if "missing optional" in lowered or "modulenotfounderror" in lowered:
        lines.append("A required optional package is missing.")
        lines.append("Run: " + build_pip_install_command())

    if "out of memory" in lowered or ("cuda" in lowered and "memory" in lowered):
        lines.append("The GPU ran out of memory.")
        lines.append("Restart the runtime, keep RUN_TRAINING = True, and rerun from setup.")
        lines.append("If it fails again, reduce max_steps or max_length before retrying.")

    if "no cuda" in lowered or "cuda gpu" in lowered or "not compiled with cuda" in lowered:
        lines.append("A CUDA GPU is not available.")
        lines.append("In Colab, choose Runtime > Change runtime type > GPU, then rerun the runtime check.")

    if "adapter path" in lowered or "adapter" in lowered and "does not exist" in lowered:
        lines.append("The trained adapter was not found.")
        lines.append("Run the training cell first and confirm it prints an adapter output path under outputs/.")

    if len(lines) == 1:
        lines.append("Read the error above, then rerun the training cell after checking setup.")

    return lines


__all__ = [
    "OPTIONAL_COMPARISON_PACKAGES",
    "OPTIONAL_TRAINING_INSTALL_PACKAGES",
    "OPTIONAL_TRAINING_PACKAGES",
    "RuntimeCheck",
    "build_pip_install_command",
    "check_training_runtime",
    "explain_runtime_failure",
    "format_runtime_check",
]
