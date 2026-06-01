"""OpenDistillation v0 prototype helpers.

The package intentionally starts with deterministic local helpers only. Real
teacher-model calls and model training belong to later milestones.
"""

from .dataset import DatasetValidationError, rows_to_jsonl, validate_dataset, validate_dataset_row
from .engines import ExportEngine, ExportRequest, ExportResult, TrainingEngine, TrainingRequest, TrainingResult
from .teacher import MockTeacherEngine, TeacherEngine, TeacherRequest, build_teacher_prompt, generate_mock_qa_pairs
from .text import LoadedTextDocument, TextChunk, TextValidationError, chunk_text, load_text_document

__all__ = [
    "DatasetValidationError",
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
    "build_teacher_prompt",
    "chunk_text",
    "generate_mock_qa_pairs",
    "load_text_document",
    "rows_to_jsonl",
    "validate_dataset",
    "validate_dataset_row",
]
