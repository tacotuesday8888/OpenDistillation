"""Bounded supervised fine-tuning path for the v0 notes model.

The module keeps heavy ML dependencies out of normal imports. Local tests and
the CPU-only notebook path can inspect the plan without downloading models.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from .dataset import validate_dataset
from .engines import TrainingRequest, TrainingResult
from .runtime import build_pip_install_command


DEFAULT_STUDENT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAINING_ENGINE_NAME = "trl-sfttrainer-peft-lora"
TRAINING_DEPENDENCIES = ("torch", "transformers", "datasets", "trl", "peft", "accelerate")
QWEN_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class TrainingDependencyError(RuntimeError):
    """Raised when optional Colab training dependencies are unavailable."""


class TrainingConfigurationError(ValueError):
    """Raised when an unsupported training configuration is requested."""


@dataclass(frozen=True)
class SFTLoRAConfig:
    """Small default SFT/LoRA settings for the first Colab training path."""

    student_model: str = DEFAULT_STUDENT_MODEL
    max_steps: int = 10
    max_length: int = 512
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    logging_steps: int = 1
    save_total_limit: int = 1
    seed: int = 42
    eos_token: str = "<|im_end|>"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = QWEN_LORA_TARGET_MODULES
    use_bitsandbytes: bool = False
    use_unsloth: bool = False


@dataclass(frozen=True)
class SFTPreviewRow:
    """One local preview row showing the exact SFT prompt/completion text."""

    row_index: int
    source_chunk_id: str
    prompt: str
    completion: str
    row_id: str = ""
    fact_id: str = ""
    label: str = ""
    row_style: str = ""
    expected_terms: tuple[str, ...] = ()


def build_training_request(
    rows: Iterable[Mapping[str, object]],
    output_dir: str | Path,
    *,
    config: SFTLoRAConfig | None = None,
    max_steps: int | None = None,
) -> TrainingRequest:
    """Validate dataset rows and build the existing training boundary object."""

    selected_config = config or SFTLoRAConfig()
    selected_steps = selected_config.max_steps if max_steps is None else max_steps
    if selected_steps < 1:
        raise TrainingConfigurationError("max_steps must be at least 1")

    return TrainingRequest(
        dataset_rows=validate_dataset(rows),
        student_model=selected_config.student_model,
        output_dir=Path(output_dir),
        max_steps=selected_steps,
    )


def format_sft_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, list[dict[str, str]]]]:
    """Convert v0 dataset rows into TRL conversational prompt/completion rows."""

    return [
        {
            "prompt": [{"role": "user", "content": row["instruction"]}],
            "completion": [{"role": "assistant", "content": row["response"]}],
        }
        for row in validate_dataset(rows)
    ]


def build_sft_preview_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    manifest_rows: Iterable[Mapping[str, object]] = (),
    max_rows: int | None = None,
) -> tuple[SFTPreviewRow, ...]:
    """Build a local preview of the exact prompt/completion pairs used for SFT."""

    if max_rows is not None and max_rows < 1:
        raise TrainingConfigurationError("max_rows must be at least 1")

    validated_rows = validate_dataset(rows)
    if max_rows is not None:
        validated_rows = validated_rows[:max_rows]
    manifest_by_key = {
        _manifest_key(row): row
        for row in manifest_rows
        if isinstance(row.get("instruction"), str)
    }
    formatted_rows = format_sft_rows(validated_rows)

    preview_rows: list[SFTPreviewRow] = []
    for index, (dataset_row, sft_row) in enumerate(zip(validated_rows, formatted_rows), start=1):
        manifest_row = manifest_by_key.get(_manifest_key(dataset_row), {})
        preview_rows.append(
            SFTPreviewRow(
                row_index=index,
                source_chunk_id=dataset_row["source_chunk_id"],
                prompt=sft_row["prompt"][0]["content"],
                completion=sft_row["completion"][0]["content"],
                row_id=_optional_manifest_string(manifest_row, "row_id"),
                fact_id=_optional_manifest_string(manifest_row, "fact_id"),
                label=_optional_manifest_string(manifest_row, "label"),
                row_style=_optional_manifest_string(manifest_row, "row_style"),
                expected_terms=_manifest_expected_terms(manifest_row),
            )
        )
    return tuple(preview_rows)


def format_sft_preview_report(preview_rows: Iterable[SFTPreviewRow]) -> list[str]:
    """Format SFT preview rows for the notebook and local diagnostics."""

    rows = tuple(preview_rows)
    lines = [
        "Exact SFT preview",
        "These are the exact user prompts and assistant completions sent to TRL before chat templating/tokenization.",
    ]
    for row in rows:
        heading_parts = [f"Row {row.row_index}", row.source_chunk_id]
        if row.row_id:
            heading_parts.append(row.row_id)
        if row.fact_id:
            heading_parts.append(row.fact_id)
        if row.row_style:
            heading_parts.append(row.row_style)
        lines.append(" | ".join(heading_parts))
        if row.label:
            lines.append(f"Label: {row.label}")
        if row.expected_terms:
            lines.append("Expected term(s): " + ", ".join(row.expected_terms))
        lines.append(f"Prompt: {row.prompt}")
        lines.append(f"Completion: {row.completion}")
    return lines


def build_sft_config_kwargs(
    request: TrainingRequest,
    config: SFTLoRAConfig | None = None,
) -> dict[str, Any]:
    """Return serializable kwargs for ``trl.SFTConfig``."""

    selected_config = config or SFTLoRAConfig()
    return {
        "output_dir": str(request.output_dir),
        "do_train": True,
        "eval_strategy": "no",
        "per_device_train_batch_size": selected_config.per_device_train_batch_size,
        "gradient_accumulation_steps": selected_config.gradient_accumulation_steps,
        "learning_rate": selected_config.learning_rate,
        "max_steps": request.max_steps,
        "max_length": selected_config.max_length,
        "completion_only_loss": True,
        "eos_token": selected_config.eos_token,
        "packing": False,
        "logging_strategy": "steps",
        "logging_steps": selected_config.logging_steps,
        "save_strategy": "steps",
        "save_steps": request.max_steps,
        "save_total_limit": selected_config.save_total_limit,
        "report_to": "none",
        "push_to_hub": False,
        "seed": selected_config.seed,
    }


def build_lora_config_kwargs(config: SFTLoRAConfig | None = None) -> dict[str, Any]:
    """Return serializable kwargs for ``peft.LoraConfig``."""

    selected_config = config or SFTLoRAConfig()
    return {
        "r": selected_config.lora_r,
        "lora_alpha": selected_config.lora_alpha,
        "lora_dropout": selected_config.lora_dropout,
        "target_modules": list(selected_config.target_modules),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


class SFTLoRATrainingEngine:
    """Optional Colab training engine using TRL SFTTrainer and PEFT LoRA."""

    name = TRAINING_ENGINE_NAME
    requires_gpu = True
    dependencies = TRAINING_DEPENDENCIES

    def __init__(self, config: SFTLoRAConfig | None = None):
        self.config = config or SFTLoRAConfig()

    def describe(self, request: TrainingRequest) -> dict[str, object]:
        """Describe the run without importing ML packages or starting training."""

        return {
            "engine": self.name,
            "student_model": request.student_model,
            "output_dir": str(request.output_dir),
            "max_steps": request.max_steps,
            "max_length": self.config.max_length,
            "requires_gpu": self.requires_gpu,
            "dependencies": list(self.dependencies),
            "artifact": str(request.output_dir / "adapter"),
        }

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Run a short supervised fine-tuning job after explicit user opt-in."""

        self._validate_supported_config()
        if request.student_model != self.config.student_model:
            raise TrainingConfigurationError("training request must use the configured v0 student model")
        formatted_rows = format_sft_rows(request.dataset_rows)
        request.output_dir.mkdir(parents=True, exist_ok=True)

        Dataset, LoraConfig, SFTConfig, SFTTrainer = _load_training_dependencies()
        train_dataset = Dataset.from_list(formatted_rows)
        training_args = SFTConfig(**build_sft_config_kwargs(request, self.config))
        peft_config = LoraConfig(**build_lora_config_kwargs(self.config))

        trainer = SFTTrainer(
            model=request.student_model,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config,
        )
        trainer.train()

        adapter_dir = request.output_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        return TrainingResult(
            engine_name=self.name,
            output_path=adapter_dir,
            notes=[
                "Created a PEFT LoRA adapter from validated v0 notes dataset rows.",
                "This artifact is for Colab prototype testing and is not a GGUF export.",
            ],
            created_model_artifact=True,
        )

    def _validate_supported_config(self) -> None:
        if self.config.student_model != DEFAULT_STUDENT_MODEL:
            raise TrainingConfigurationError(
                f"the first v0 training path supports only {DEFAULT_STUDENT_MODEL}"
            )
        if self.config.use_bitsandbytes:
            raise TrainingConfigurationError("bitsandbytes is not enabled in the first v0 training path")
        if self.config.use_unsloth:
            raise TrainingConfigurationError("Unsloth is not enabled in the first v0 training path")


def _load_training_dependencies() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    missing = []
    import_errors: dict[str, str] = {}
    modules: dict[str, Any] = {}
    for module_name in TRAINING_DEPENDENCIES:
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
        raise TrainingDependencyError(
            _format_dependency_error("training", missing, import_errors)
        )

    try:
        return (
            modules["datasets"].Dataset,
            modules["peft"].LoraConfig,
            modules["trl"].SFTConfig,
            modules["trl"].SFTTrainer,
        )
    except AttributeError as exc:
        raise TrainingDependencyError(
            "Installed training packages do not expose the expected TRL/PEFT APIs. "
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


def _manifest_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("instruction", "")).strip(),
        str(row.get("response", "")).strip(),
        str(row.get("source_chunk_id", "")).strip(),
    )


def _optional_manifest_string(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    return value.strip() if isinstance(value, str) else ""


def _manifest_expected_terms(row: Mapping[str, object]) -> tuple[str, ...]:
    value = row.get("expected_terms", ())
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Iterable):
        return tuple(str(term).strip() for term in value if str(term).strip())
    return ()


__all__ = [
    "DEFAULT_STUDENT_MODEL",
    "QWEN_LORA_TARGET_MODULES",
    "SFTLoRAConfig",
    "SFTLoRATrainingEngine",
    "SFTPreviewRow",
    "TRAINING_DEPENDENCIES",
    "TRAINING_ENGINE_NAME",
    "TrainingConfigurationError",
    "TrainingDependencyError",
    "build_lora_config_kwargs",
    "build_sft_config_kwargs",
    "build_sft_preview_rows",
    "build_training_request",
    "format_sft_preview_report",
    "format_sft_rows",
]
