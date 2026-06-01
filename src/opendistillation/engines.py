"""Future training and export engine boundaries.

These are intentionally inert interfaces. They let later work add real engines
without changing the notebook's high-level flow.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class TrainingRequest:
    """Input boundary for future student fine-tuning engines."""

    dataset_rows: list[Mapping[str, str]]
    student_model: str
    output_dir: Path
    max_steps: int = 20


@dataclass(frozen=True)
class TrainingResult:
    """Output boundary for future student fine-tuning engines."""

    engine_name: str
    output_path: Path
    notes: list[str] = field(default_factory=list)
    created_model_artifact: bool = False


class TrainingEngine(Protocol):
    """Protocol future Unsloth/Transformers/PEFT/TRL integrations can satisfy."""

    name: str
    requires_gpu: bool

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Train or fine-tune a student model from validated dataset rows."""


@dataclass(frozen=True)
class ExportRequest:
    """Input boundary for future local-runtime export engines."""

    training_output_path: Path
    target_format: str = "gguf"
    local_runtime: str = "llama.cpp"


@dataclass(frozen=True)
class ExportResult:
    """Output boundary for future GGUF/Ollama-style export engines."""

    engine_name: str
    output_path: Path | None
    local_command: str | None
    verified: bool
    notes: list[str] = field(default_factory=list)


class ExportEngine(Protocol):
    """Protocol future llama.cpp/GGUF/Ollama export integrations can satisfy."""

    name: str
    target_format: str

    def export(self, request: ExportRequest) -> ExportResult:
        """Convert training output into a local-runtime artifact or instructions."""
