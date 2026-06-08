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
class ComparisonExample:
    """One generated dataset example selected for model comparison."""

    question: str
    reference_response: str
    source_chunk_id: str
    row_id: str = ""
    fact_id: str = ""
    label: str = ""
    value: str = ""
    expected_terms: tuple[str, ...] = ()
    row_style: str = ""


@dataclass(frozen=True)
class BeforeAfterComparisonRequest:
    """Input for comparing base-model and adapter-model answers."""

    examples: tuple[ComparisonExample, ...]
    student_model: str
    adapter_path: Path
    max_new_tokens: int = 96

    @property
    def question(self) -> str:
        return self.examples[0].question

    @property
    def reference_response(self) -> str:
        return self.examples[0].reference_response

    @property
    def source_chunk_id(self) -> str:
        return self.examples[0].source_chunk_id


@dataclass(frozen=True)
class BeforeAfterComparisonItem:
    """One before/after answer pair with simple quality signals."""

    question: str
    reference_response: str
    base_answer: str
    trained_answer: str
    source_chunk_id: str
    base_reference_overlap: float
    trained_reference_overlap: float
    row_id: str = ""
    fact_id: str = ""
    label: str = ""
    value: str = ""
    expected_terms: tuple[str, ...] = ()
    row_style: str = ""

    @property
    def overlap_delta(self) -> float:
        return self.trained_reference_overlap - self.base_reference_overlap


@dataclass(frozen=True)
class BeforeAfterComparisonResult:
    """Output from a bounded base-versus-adapter comparison."""

    items: tuple[BeforeAfterComparisonItem, ...]
    adapter_path: Path
    notes: list[str] = field(default_factory=list)

    @property
    def question(self) -> str:
        return self.items[0].question

    @property
    def reference_response(self) -> str:
        return self.items[0].reference_response

    @property
    def base_answer(self) -> str:
        return self.items[0].base_answer

    @property
    def trained_answer(self) -> str:
        return self.items[0].trained_answer

    @property
    def source_chunk_id(self) -> str:
        return self.items[0].source_chunk_id

    def fact_outputs(self, answer_kind: str) -> tuple[dict[str, object], ...]:
        """Return score-ready outputs using each item's own fact metadata."""

        if answer_kind not in {"base", "trained"}:
            raise ComparisonConfigurationError("answer_kind must be 'base' or 'trained'")
        answer_field = "base_answer" if answer_kind == "base" else "trained_answer"
        return tuple(
            {
                "question": item.question,
                "answer": getattr(item, answer_field),
                "expected_terms": item.expected_terms,
                "fact_id": item.fact_id,
                "label": item.label,
                "row_style": item.row_style,
                "source_chunk_id": item.source_chunk_id,
            }
            for item in self.items
        )


def build_comparison_request(
    rows: Iterable[Mapping[str, object]],
    training_result: TrainingResult,
    *,
    config: SFTLoRAConfig | None = None,
    max_new_tokens: int = 96,
    max_examples: int = 3,
) -> BeforeAfterComparisonRequest:
    """Pick a bounded set of generated dataset questions for comparison."""

    if max_new_tokens < 1:
        raise ComparisonConfigurationError("max_new_tokens must be at least 1")
    if max_examples < 1:
        raise ComparisonConfigurationError("max_examples must be at least 1")
    if not training_result.created_model_artifact:
        raise ComparisonConfigurationError("comparison requires a training result with a model artifact")

    selected_config = config or SFTLoRAConfig()
    input_rows = tuple(rows)
    validated_rows = validate_dataset(input_rows)
    comparison_rows = [
        _comparison_row_with_metadata(original, validated)
        for original, validated in zip(input_rows, validated_rows)
    ]
    selected_rows = _select_comparison_rows(comparison_rows, max_examples=max_examples)
    examples = tuple(
        ComparisonExample(
            question=str(row["instruction"]),
            reference_response=str(row["response"]),
            source_chunk_id=str(row["source_chunk_id"]),
            row_id=str(row.get("row_id", "")),
            fact_id=str(row.get("fact_id", "")),
            label=str(row.get("label", "")),
            value=str(row.get("value", "")),
            expected_terms=_expected_terms(row.get("expected_terms", ())),
            row_style=str(row.get("row_style", "")),
        )
        for row in selected_rows
    )
    return BeforeAfterComparisonRequest(
        examples=examples,
        student_model=selected_config.student_model,
        adapter_path=Path(training_result.output_path),
        max_new_tokens=max_new_tokens,
    )


def _select_comparison_rows(
    validated_rows: list[dict[str, object]],
    *,
    max_examples: int,
) -> list[dict[str, object]]:
    selected_rows: list[dict[str, object]] = []
    selected_indexes: set[int] = set()
    seen_source_chunks: set[str] = set()

    for index, row in enumerate(validated_rows):
        source_chunk_id = str(row["source_chunk_id"])
        if source_chunk_id in seen_source_chunks:
            continue
        selected_rows.append(row)
        selected_indexes.add(index)
        seen_source_chunks.add(source_chunk_id)
        if len(selected_rows) == max_examples:
            return selected_rows

    for index, row in enumerate(validated_rows):
        if index in selected_indexes:
            continue
        selected_rows.append(row)
        if len(selected_rows) == max_examples:
            return selected_rows

    return selected_rows


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
            "question_count": len(request.examples),
            "questions": [example.question for example in request.examples],
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
        trained_model = PeftModel.from_pretrained(base_model, str(request.adapter_path))
        trained_model.eval()
        disable_adapter = getattr(trained_model, "disable_adapter", None)
        if not callable(disable_adapter):
            raise ComparisonDependencyError(
                "Installed PEFT PeftModel does not expose disable_adapter(), "
                "which is required for a true base-versus-adapter comparison."
            )

        items: list[BeforeAfterComparisonItem] = []
        for example in request.examples:
            with disable_adapter():
                base_answer = _generate_chat_answer(
                    trained_model,
                    tokenizer,
                    torch_module,
                    example.question,
                    request.max_new_tokens,
                )
            trained_answer = _generate_chat_answer(
                trained_model,
                tokenizer,
                torch_module,
                example.question,
                request.max_new_tokens,
            )
            items.append(
                BeforeAfterComparisonItem(
                    question=example.question,
                    reference_response=example.reference_response,
                    base_answer=base_answer,
                    trained_answer=trained_answer,
                    source_chunk_id=example.source_chunk_id,
                    base_reference_overlap=_reference_overlap(example.reference_response, base_answer),
                    trained_reference_overlap=_reference_overlap(example.reference_response, trained_answer),
                    row_id=example.row_id,
                    fact_id=example.fact_id,
                    label=example.label,
                    value=example.value,
                    expected_terms=example.expected_terms,
                    row_style=example.row_style,
                )
            )

        return BeforeAfterComparisonResult(
            items=tuple(items),
            adapter_path=request.adapter_path,
            notes=[
                "This is a qualitative quality smoke report, not a benchmark.",
                "The base answer is generated with the LoRA adapter disabled; the trained answer uses the adapter enabled.",
                "The reference answer is generated by the current teacher path.",
                "Reference-overlap scores are crude lexical signals; read the answers before trusting them.",
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


def _comparison_row_with_metadata(
    original: Mapping[str, object],
    validated: Mapping[str, str],
) -> dict[str, object]:
    row: dict[str, object] = dict(validated)
    for field in ("row_id", "fact_id", "label", "value", "row_style"):
        value = original.get(field)
        if isinstance(value, str) and value.strip():
            row[field] = value.strip()
    expected_terms = _expected_terms(original.get("expected_terms", ()))
    if expected_terms:
        row["expected_terms"] = expected_terms
    return row


def _expected_terms(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Iterable):
        return tuple(str(term).strip() for term in value if str(term).strip())
    return ()


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


def _reference_overlap(reference: str, answer: str) -> float:
    reference_tokens = set(_tokenize_for_overlap(reference))
    answer_tokens = set(_tokenize_for_overlap(answer))
    if not reference_tokens or not answer_tokens:
        return 0.0
    return round(len(reference_tokens & answer_tokens) / len(reference_tokens), 3)


def _tokenize_for_overlap(text: str) -> list[str]:
    import re

    return re.findall(r"\b[\w'-]+\b", text.lower())


__all__ = [
    "BeforeAfterComparisonEngine",
    "BeforeAfterComparisonItem",
    "BeforeAfterComparisonRequest",
    "BeforeAfterComparisonResult",
    "COMPARISON_DEPENDENCIES",
    "COMPARISON_ENGINE_NAME",
    "ComparisonExample",
    "ComparisonConfigurationError",
    "ComparisonDependencyError",
    "build_comparison_request",
]
