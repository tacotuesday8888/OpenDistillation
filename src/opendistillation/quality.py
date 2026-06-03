"""Deterministic dataset-quality checks for the v0 notes dataset."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from .dataset import REQUIRED_FIELDS, validate_dataset_row


@dataclass(frozen=True)
class DatasetQualityIssue:
    """One beginner-readable dataset quality finding."""

    code: str
    severity: str
    message: str
    row_numbers: tuple[int, ...] = ()


@dataclass(frozen=True)
class DatasetQualityReport:
    """Deterministic report for generated training rows."""

    row_count: int
    valid_row_count: int
    expected_chunk_ids: tuple[str, ...]
    covered_chunk_ids: tuple[str, ...]
    missing_chunk_ids: tuple[str, ...]
    extra_source_chunk_ids: tuple[str, ...]
    missing_field_count: int
    duplicate_question_count: int
    near_duplicate_question_count: int
    short_answer_count: int
    long_answer_count: int
    issues: tuple[DatasetQualityIssue, ...]

    @property
    def passes_required_checks(self) -> bool:
        return not self.issues


def analyze_dataset_quality(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_chunk_ids: Sequence[str] = (),
    min_answer_words: int = 6,
    max_answer_words: int = 180,
    near_duplicate_threshold: float = 0.86,
) -> DatasetQualityReport:
    """Return a no-download quality report for generated v0 dataset rows."""

    materialized_rows = list(rows)
    expected_ids = tuple(dict.fromkeys(chunk_id.strip() for chunk_id in expected_chunk_ids if chunk_id.strip()))
    expected_id_set = set(expected_ids)
    issues: list[DatasetQualityIssue] = []
    valid_rows: list[tuple[int, dict[str, str]]] = []
    missing_field_count = 0
    short_answer_count = 0
    long_answer_count = 0

    for row_number, row in enumerate(materialized_rows, start=1):
        missing_fields = tuple(
            field
            for field in REQUIRED_FIELDS
            if field not in row or not isinstance(row[field], str) or not str(row[field]).strip()
        )
        if missing_fields:
            missing_field_count += len(missing_fields)
            issues.append(
                DatasetQualityIssue(
                    code="missing_required_field",
                    severity="error",
                    message=(
                        f"Row {row_number} is missing required field(s): "
                        + ", ".join(missing_fields)
                        + "."
                    ),
                    row_numbers=(row_number,),
                )
            )
            continue

        validated = validate_dataset_row(row)
        valid_rows.append((row_number, validated))
        answer_words = _word_count(validated["response"])
        if answer_words < min_answer_words:
            short_answer_count += 1
            issues.append(
                DatasetQualityIssue(
                    code="short_answer",
                    severity="warning",
                    message=(
                        f"Row {row_number} has a very short answer "
                        f"({answer_words} words). It may not teach much."
                    ),
                    row_numbers=(row_number,),
                )
            )
        if answer_words > max_answer_words:
            long_answer_count += 1
            issues.append(
                DatasetQualityIssue(
                    code="long_answer",
                    severity="warning",
                    message=(
                        f"Row {row_number} has a long answer "
                        f"({answer_words} words). It may be hard for a tiny demo run."
                    ),
                    row_numbers=(row_number,),
                )
            )

    covered_ids = tuple(
        chunk_id
        for chunk_id in expected_ids
        if any(row["source_chunk_id"] == chunk_id for _, row in valid_rows)
    )
    observed_source_ids = tuple(dict.fromkeys(row["source_chunk_id"] for _, row in valid_rows))
    extra_source_ids = tuple(
        source_id for source_id in observed_source_ids if expected_id_set and source_id not in expected_id_set
    )
    missing_source_ids = tuple(chunk_id for chunk_id in expected_ids if chunk_id not in set(covered_ids))

    if missing_source_ids:
        issues.append(
            DatasetQualityIssue(
                code="missing_chunk_coverage",
                severity="warning",
                message="No generated row covers chunk(s): " + ", ".join(missing_source_ids) + ".",
            )
        )
    if extra_source_ids:
        issues.append(
            DatasetQualityIssue(
                code="unexpected_source_chunk_id",
                severity="error",
                message="Generated rows reference unknown chunk ID(s): " + ", ".join(extra_source_ids) + ".",
            )
        )

    duplicate_count, duplicate_issues = _find_duplicate_questions(valid_rows)
    issues.extend(duplicate_issues)
    near_count, near_issues = _find_near_duplicate_questions(valid_rows, threshold=near_duplicate_threshold)
    issues.extend(near_issues)

    return DatasetQualityReport(
        row_count=len(materialized_rows),
        valid_row_count=len(valid_rows),
        expected_chunk_ids=expected_ids,
        covered_chunk_ids=covered_ids if expected_ids else observed_source_ids,
        missing_chunk_ids=missing_source_ids,
        extra_source_chunk_ids=extra_source_ids,
        missing_field_count=missing_field_count,
        duplicate_question_count=duplicate_count,
        near_duplicate_question_count=near_count,
        short_answer_count=short_answer_count,
        long_answer_count=long_answer_count,
        issues=tuple(issues),
    )


def format_dataset_quality_report(report: DatasetQualityReport) -> list[str]:
    """Format a dataset report for notebook users."""

    expected_count = len(report.expected_chunk_ids)
    covered_count = len(report.covered_chunk_ids)
    lines = [
        "Dataset quality report",
        f"Rows: {report.row_count} total, {report.valid_row_count} schema-valid",
    ]
    if expected_count:
        lines.append(f"Chunk coverage: {covered_count}/{expected_count}")
    else:
        lines.append(f"Source chunks represented: {covered_count}")
    lines.append(
        "Question checks: "
        f"{report.duplicate_question_count} duplicate, "
        f"{report.near_duplicate_question_count} near-duplicate"
    )
    lines.append(
        "Answer checks: "
        f"{report.short_answer_count} very short, "
        f"{report.long_answer_count} very long"
    )

    if report.passes_required_checks:
        lines.append("No required dataset-quality problems found for this small prototype dataset.")
        return lines

    lines.append("Issues to review before trusting a training run:")
    lines.extend(f"- [{issue.severity}] {issue.message}" for issue in report.issues)
    return lines


def _find_duplicate_questions(
    valid_rows: list[tuple[int, dict[str, str]]],
) -> tuple[int, list[DatasetQualityIssue]]:
    row_numbers_by_question: dict[str, list[int]] = {}
    for row_number, row in valid_rows:
        normalized = _normalize_question(row["instruction"])
        row_numbers_by_question.setdefault(normalized, []).append(row_number)

    issues: list[DatasetQualityIssue] = []
    duplicate_count = 0
    for row_numbers in row_numbers_by_question.values():
        if len(row_numbers) < 2:
            continue
        duplicate_count += len(row_numbers) - 1
        issues.append(
            DatasetQualityIssue(
                code="duplicate_question",
                severity="warning",
                message="Duplicate question text appears in rows: " + ", ".join(map(str, row_numbers)) + ".",
                row_numbers=tuple(row_numbers),
            )
        )
    return duplicate_count, issues


def _find_near_duplicate_questions(
    valid_rows: list[tuple[int, dict[str, str]]],
    *,
    threshold: float,
) -> tuple[int, list[DatasetQualityIssue]]:
    normalized_rows_by_source: dict[str, list[tuple[int, str]]] = {}
    for row_number, row in valid_rows:
        normalized_rows_by_source.setdefault(row["source_chunk_id"], []).append(
            (row_number, _normalize_question(row["instruction"]))
        )

    issues: list[DatasetQualityIssue] = []
    for normalized_rows in normalized_rows_by_source.values():
        for left_index, (left_row_number, left_question) in enumerate(normalized_rows):
            for right_row_number, right_question in normalized_rows[left_index + 1 :]:
                if left_question == right_question:
                    continue
                ratio = SequenceMatcher(None, left_question, right_question).ratio()
                if ratio >= threshold:
                    issues.append(
                        DatasetQualityIssue(
                            code="near_duplicate_question",
                            severity="warning",
                            message=(
                                "Near-duplicate questions appear in rows "
                                f"{left_row_number} and {right_row_number}."
                            ),
                            row_numbers=(left_row_number, right_row_number),
                        )
                    )
    return len(issues), issues


def _normalize_question(text: str) -> str:
    tokens = re.findall(r"\b[\w'-]+\b", text.lower())
    return " ".join(sorted(tokens))


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


__all__ = [
    "DatasetQualityIssue",
    "DatasetQualityReport",
    "analyze_dataset_quality",
    "format_dataset_quality_report",
]
