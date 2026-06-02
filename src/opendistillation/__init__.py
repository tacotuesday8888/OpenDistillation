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
    "ExportEngine",
    "ExportRequest",
    "ExportResult",
    "LoadedTextDocument",
    "MockTeacherEngine",
    "TextChunk",
    "TextValidationError",
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
    "build_teacher_prompt",
    "build_training_request",
    "chunk_text",
    "format_sft_rows",
    "generate_mock_qa_pairs",
    "load_text_document",
    "rows_to_jsonl",
    "validate_dataset",
    "validate_dataset_row",
]
