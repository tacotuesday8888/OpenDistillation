"""OpenDistillation v0 prototype helpers.

The default path intentionally starts with deterministic local helpers only.
Optional model training and comparison helpers stay lazy-loaded.
"""

from .dataset import DatasetValidationError, rows_to_jsonl, validate_dataset, validate_dataset_row
from .engines import ExportEngine, ExportRequest, ExportResult, TrainingEngine, TrainingRequest, TrainingResult
from .teacher import MockTeacherEngine, TeacherEngine, TeacherRequest, build_teacher_prompt, generate_mock_qa_pairs
from .text import LoadedTextDocument, TextChunk, TextValidationError, chunk_text, load_text_document
from .comparison import (
    BeforeAfterComparisonEngine,
    BeforeAfterComparisonRequest,
    BeforeAfterComparisonResult,
    ComparisonConfigurationError,
    ComparisonDependencyError,
    build_comparison_request,
)
from .runtime import (
    OPTIONAL_COMPARISON_PACKAGES,
    OPTIONAL_TRAINING_INSTALL_PACKAGES,
    OPTIONAL_TRAINING_PACKAGES,
    RuntimeCheck,
    build_pip_install_command,
    check_training_runtime,
    explain_runtime_failure,
    format_runtime_check,
)
from .training import (
    DEFAULT_STUDENT_MODEL,
    SFTLoRAConfig,
    SFTLoRATrainingEngine,
    TrainingConfigurationError,
    TrainingDependencyError,
    build_training_request,
    format_sft_rows,
)

__all__ = [
    "DatasetValidationError",
    "DEFAULT_STUDENT_MODEL",
    "BeforeAfterComparisonEngine",
    "BeforeAfterComparisonRequest",
    "BeforeAfterComparisonResult",
    "ComparisonConfigurationError",
    "ComparisonDependencyError",
    "OPTIONAL_COMPARISON_PACKAGES",
    "OPTIONAL_TRAINING_INSTALL_PACKAGES",
    "OPTIONAL_TRAINING_PACKAGES",
    "ExportEngine",
    "ExportRequest",
    "ExportResult",
    "LoadedTextDocument",
    "MockTeacherEngine",
    "TextChunk",
    "TextValidationError",
    "RuntimeCheck",
    "TeacherEngine",
    "TeacherRequest",
    "TrainingEngine",
    "TrainingRequest",
    "TrainingResult",
    "SFTLoRAConfig",
    "SFTLoRATrainingEngine",
    "TrainingConfigurationError",
    "TrainingDependencyError",
    "build_comparison_request",
    "build_pip_install_command",
    "build_teacher_prompt",
    "build_training_request",
    "check_training_runtime",
    "chunk_text",
    "explain_runtime_failure",
    "format_sft_rows",
    "format_runtime_check",
    "generate_mock_qa_pairs",
    "load_text_document",
    "rows_to_jsonl",
    "validate_dataset",
    "validate_dataset_row",
]
