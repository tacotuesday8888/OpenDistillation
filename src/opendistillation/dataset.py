"""Dataset schema helpers for generated v0 training examples."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping


REQUIRED_FIELDS = ("instruction", "response", "source_chunk_id")


class DatasetValidationError(ValueError):
    """Raised when generated training data does not match the v0 schema."""


def validate_dataset_row(row: Mapping[str, object]) -> dict[str, str]:
    """Return a normalized dataset row with only public schema fields."""

    for field in REQUIRED_FIELDS:
        if field not in row:
            raise DatasetValidationError(f"missing required field: {field}")

    normalized: dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        value = row[field]
        if not isinstance(value, str) or not value.strip():
            raise DatasetValidationError(f"{field} must be a non-empty string")
        normalized[field] = value.strip()
    return normalized


def validate_dataset(rows: Iterable[Mapping[str, object]]) -> list[dict[str, str]]:
    """Validate all rows and include row numbers in error messages."""

    validated = []
    for index, row in enumerate(rows, start=1):
        try:
            validated.append(validate_dataset_row(row))
        except DatasetValidationError as exc:
            raise DatasetValidationError(f"row {index}: {exc}") from exc
    if not validated:
        raise DatasetValidationError("dataset must contain at least one row")
    return validated


def rows_to_jsonl(rows: Iterable[Mapping[str, object]]) -> str:
    """Serialize validated rows as one JSON object per line."""

    validated = validate_dataset(rows)
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in validated)
