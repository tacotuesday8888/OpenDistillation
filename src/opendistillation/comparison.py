"""Before/after comparison helpers for the optional v0 training path.

The comparison engine is intentionally lazy-loaded. Importing this module should
not import Transformers, PEFT, Accelerate, or torch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from .dataset import validate_dataset
from .engines import TrainingResult
from .runtime import build_pip_install_command
from .training import DEFAULT_STUDENT_MODEL, SFTLoRAConfig


COMPARISON_ENGINE_NAME = "transformers-peft-before-after"
COMPARISON_DEPENDENCIES = ("torch", "transformers", "peft", "accelerate")


class ComparisonDependencyError(RuntimeError):
    """Raised when optional comparison dependencies are unavailable."""


class ComparisonConfigurationError(ValueError):
    """Raised when comparison cannot run with the provided request."""


@dataclass(frozen=True)
class BeforeAfterComparisonRequest:
    """Input for comparing base-model and adapter-model answers."""

    question: str
    reference_response: str
    source_chunk_id: str
    student_model: str
    adapter_path: Path
    max_new_tokens: int = 96


@dataclass(frozen=True)
class BeforeAfterComparisonResult:
    """Output from a simple base-versus-adapter comparison."""

    question: str
    reference_response: str
    base_answer: str
    trained_answer: str
    adapter_path: Path
    source_chunk_id: str
    notes: list[str] = field(default_factory=list)


def build_comparison_request(
    rows: Iterable[Mapping[str, object]],
    training_result: TrainingResult,
    *,
    config: SFTLoRAConfig | None = None,
    max_new_tokens: int = 96,
) -> BeforeAfterComparisonRequest:
    """Pick one generated dataset question for a before/after comparison."""

    if max_new_tokens < 1:
        raise ComparisonConfigurationError("max_new_tokens must be at least 1")
    if not training_result.created_model_artifact:
        raise ComparisonConfigurationError("comparison requires a training result with a model artifact")

    selected_config = config or SFTLoRAConfig()
    validated_rows = validate_dataset(rows)
    first_row = validated_rows[0]
    return BeforeAfterComparisonRequest(
        question=first_row["instruction"],
        reference_response=first_row["response"],
        source_chunk_id=first_row["source_chunk_id"],
        student_model=selected_config.student_model,
        adapter_path=Path(training_result.output_path),
        max_new_tokens=max_new_tokens,
    )


class BeforeAfterComparisonEngine:
    """Compare one prompt against the base model and trained LoRA adapter."""

    name = COMPARISON_ENGINE_NAME
    requires_gpu = True
    dependencies = COMPARISON_DEPENDENCIES

    def describe(self, request: BeforeAfterComparisonRequest) -> dict[str, object]:
        """Describe the comparison without importing ML packages."""

        return {
            "engine": self.name,
            "student_model": request.student_model,
            "question": request.question,
            "source_chunk_id": request.source_chunk_id,
            "adapter_path": str(request.adapter_path),
            "max_new_tokens": request.max_new_tokens,
            "requires_gpu": self.requires_gpu,
            "dependencies": list(self.dependencies),
        }

    def compare(self, request: BeforeAfterComparisonRequest) -> BeforeAfterComparisonResult:
        """Generate one base answer and one adapter answer."""

        _validate_request(request)
        torch_module, AutoModelForCausalLM, AutoTokenizer, PeftModel = _load_comparison_dependencies()

        tokenizer = AutoTokenizer.from_pretrained(request.student_model, padding_side="left")
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(request.student_model, device_map="auto")
        base_model.eval()
        base_answer = _generate_chat_answer(
            base_model,
            tokenizer,
            torch_module,
            request.question,
            request.max_new_tokens,
        )

        trained_model = PeftModel.from_pretrained(base_model, str(request.adapter_path))
        trained_model.eval()
        trained_answer = _generate_chat_answer(
            trained_model,
            tokenizer,
            torch_module,
            request.question,
            request.max_new_tokens,
        )

        return BeforeAfterComparisonResult(
            question=request.question,
            reference_response=request.reference_response,
            base_answer=base_answer,
            trained_answer=trained_answer,
            adapter_path=request.adapter_path,
            source_chunk_id=request.source_chunk_id,
            notes=[
                "This is a qualitative sanity check, not a benchmark.",
                "The reference answer is generated by the current teacher path.",
            ],
        )


def _validate_request(request: BeforeAfterComparisonRequest) -> None:
    if request.student_model != DEFAULT_STUDENT_MODEL:
        raise ComparisonConfigurationError(f"comparison supports only {DEFAULT_STUDENT_MODEL}")
    if not request.adapter_path.exists():
        raise ComparisonConfigurationError(f"adapter path does not exist: {request.adapter_path}")
    if request.max_new_tokens < 1:
        raise ComparisonConfigurationError("max_new_tokens must be at least 1")


def _load_comparison_dependencies() -> tuple[Any, type[Any], type[Any], type[Any]]:
    missing = []
    import_errors: dict[str, str] = {}
    modules: dict[str, Any] = {}
    for module_name in COMPARISON_DEPENDENCIES:
        try:
            modules[module_name] = import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_name = exc.name or module_name
            if missing_name == module_name:
                if missing_name not in missing:
                    missing.append(missing_name)
            else:
                import_errors[module_name] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            import_errors[module_name] = f"{type(exc).__name__}: {exc}"

    if missing or import_errors:
        raise ComparisonDependencyError(
            _format_dependency_error("comparison", missing, import_errors)
        )

    try:
        return (
            modules["torch"],
            modules["transformers"].AutoModelForCausalLM,
            modules["transformers"].AutoTokenizer,
            modules["peft"].PeftModel,
        )
    except AttributeError as exc:
        raise ComparisonDependencyError(
            "Installed comparison packages do not expose the expected Transformers/PEFT APIs. "
            "In Colab, install with: "
            + build_pip_install_command()
            + " without upgrading Colab's preinstalled torch."
        ) from exc


def _format_dependency_error(kind: str, missing: list[str], import_errors: dict[str, str]) -> str:
    parts: list[str] = []
    if missing:
        parts.append(f"Missing optional {kind} packages: " + ", ".join(missing) + ".")
    if import_errors:
        failures = "; ".join(f"{package}: {error}" for package, error in import_errors.items())
        parts.append(f"Optional {kind} package import failures: {failures}.")
    parts.append(
        "In Colab, install with: "
        + build_pip_install_command()
        + " without upgrading Colab's preinstalled torch."
    )
    return " ".join(parts)


def _generate_chat_answer(
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    question: str,
    max_new_tokens: int,
) -> str:
    messages = [{"role": "user", "content": question}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_length = input_ids.shape[-1]
    if hasattr(input_ids, "to"):
        input_ids = input_ids.to(model.device)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    with torch_module.no_grad():
        generated_ids = model.generate(input_ids, **generation_kwargs)

    generated_answer_ids = generated_ids[:, input_length:]
    decoded = tokenizer.batch_decode(generated_answer_ids, skip_special_tokens=True)
    return decoded[0].strip()


__all__ = [
    "BeforeAfterComparisonEngine",
    "BeforeAfterComparisonRequest",
    "BeforeAfterComparisonResult",
    "COMPARISON_DEPENDENCIES",
    "COMPARISON_ENGINE_NAME",
    "ComparisonConfigurationError",
    "ComparisonDependencyError",
    "build_comparison_request",
]
