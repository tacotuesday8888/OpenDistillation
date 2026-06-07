"""Fact-ledger helpers for notes-model train/eval data quality.

The fact ledger is a deterministic, local intermediate representation. It is
not the public training JSONL. It lets the prototype know which concrete facts
training and eval rows are supposed to test.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace as dataclass_replace
from difflib import SequenceMatcher
import hashlib
import re

from .dataset import validate_dataset
from .text import TextChunk


@dataclass(frozen=True)
class FactCard:
    """One learnable atomic fact extracted from a notes chunk."""

    fact_id: str
    source_chunk_id: str
    label: str
    value: str
    expected_terms: tuple[str, ...]
    fact_kind: str
    source_text: str
    source_hash: str


@dataclass(frozen=True)
class FactTrainEvalSplit:
    """Public train/eval rows plus internal metadata for quality checks."""

    facts: tuple[FactCard, ...]
    train_rows: tuple[dict[str, str], ...]
    eval_rows: tuple[dict[str, str], ...]
    manifest_rows: tuple[dict[str, object], ...]

    def replace(self, **changes: object) -> "FactTrainEvalSplit":
        """Return a copy with selected fields changed, for tests and checks."""

        return dataclass_replace(self, **changes)


@dataclass(frozen=True)
class FactQualityIssue:
    """One train/eval quality gate issue."""

    code: str
    severity: str
    message: str
    row_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactQualityGateReport:
    """Deterministic report for the fact-ledger train/eval split."""

    fact_count: int
    train_row_count: int
    eval_row_count: int
    train_fact_coverage: int
    eval_fact_coverage: int
    exact_leak_count: int
    near_leak_count: int
    missing_expected_term_count: int
    unknown_source_chunk_count: int
    issues: tuple[FactQualityIssue, ...]

    @property
    def passes_required_checks(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class FactAnswerScore:
    """Exact expected-term hit score for one answer."""

    question: str
    answer: str
    expected_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]

    @property
    def hit(self) -> bool:
        return not self.missing_terms


@dataclass(frozen=True)
class FactOutputScore:
    """Exact fact-hit summary for multiple answers."""

    items: tuple[FactAnswerScore, ...]

    @property
    def answer_count(self) -> int:
        return len(self.items)

    @property
    def hit_count(self) -> int:
        return sum(1 for item in self.items if item.hit)

    @property
    def miss_count(self) -> int:
        return self.answer_count - self.hit_count


def extract_fact_ledger(chunks: Iterable[TextChunk]) -> list[FactCard]:
    """Extract simple stable facts from note chunks.

    V0 intentionally starts with explicit ``Label: value`` facts. That covers
    the sample notes and many beginner study notes without model calls.
    """

    facts: list[FactCard] = []
    for chunk in chunks:
        source_hash = _source_hash(chunk.text)
        for label, value in _extract_label_value_pairs(chunk.text):
            fact_id = f"fact-{len(facts) + 1:04d}"
            facts.append(
                FactCard(
                    fact_id=fact_id,
                    source_chunk_id=chunk.id,
                    label=label,
                    value=value,
                    expected_terms=(value,),
                    fact_kind="label_value",
                    source_text=chunk.text,
                    source_hash=source_hash,
                )
            )
    return facts


def build_fact_train_eval_split(
    facts: Iterable[FactCard],
    *,
    train_examples_per_fact: int = 3,
) -> FactTrainEvalSplit:
    """Build public train rows, held-out eval rows, and sidecar metadata."""

    if train_examples_per_fact < 1:
        raise ValueError("train_examples_per_fact must be at least 1")

    fact_tuple = tuple(facts)
    train_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, object]] = []

    for fact in fact_tuple:
        train_templates = _train_templates(fact)
        for index in range(train_examples_per_fact):
            instruction, response = train_templates[index % len(train_templates)]
            row = {
                "instruction": instruction,
                "response": response,
                "source_chunk_id": fact.source_chunk_id,
            }
            train_rows.append(row)
            manifest_rows.append(
                _manifest_row(
                    row_id=f"train-{len(train_rows):06d}",
                    split="train",
                    fact=fact,
                    row=row,
                )
            )

        eval_row = {
            "instruction": f"In your own words, what should the notes say for {fact.label.lower()}?",
            "response": f"The notes say that {fact.label.lower()} is {fact.value}.",
            "source_chunk_id": fact.source_chunk_id,
        }
        eval_rows.append(eval_row)
        manifest_rows.append(
            _manifest_row(
                row_id=f"eval-{len(eval_rows):06d}",
                split="eval",
                fact=fact,
                row=eval_row,
            )
        )

    return FactTrainEvalSplit(
        facts=fact_tuple,
        train_rows=tuple(validate_dataset(train_rows)),
        eval_rows=tuple(validate_dataset(eval_rows)),
        manifest_rows=tuple(manifest_rows),
    )


def analyze_fact_quality_gate(
    split: FactTrainEvalSplit,
    *,
    near_duplicate_threshold: float = 0.86,
) -> FactQualityGateReport:
    """Check the fact-ledger train/eval split for leakage and coverage."""

    issues: list[FactQualityIssue] = []
    train_rows = tuple(validate_dataset(split.train_rows))
    eval_rows = tuple(validate_dataset(split.eval_rows))
    known_fact_ids = {fact.fact_id for fact in split.facts}
    known_chunk_ids = {fact.source_chunk_id for fact in split.facts}
    manifest_by_instruction = {
        str(row["instruction"]): row
        for row in split.manifest_rows
        if isinstance(row.get("instruction"), str)
    }

    train_by_normalized = {_normalize_question(row["instruction"]): row for row in train_rows}
    exact_leak_count = 0
    near_leak_count = 0

    for eval_row in eval_rows:
        normalized_eval = _normalize_question(eval_row["instruction"])
        if normalized_eval in train_by_normalized:
            exact_leak_count += 1
            issues.append(
                FactQualityIssue(
                    code="train_eval_exact_leak",
                    severity="error",
                    message="An eval question is copied from the training questions.",
                    row_ids=_row_ids_for_instruction(manifest_by_instruction, eval_row["instruction"]),
                )
            )
            continue

        for train_row in train_rows:
            ratio = SequenceMatcher(
                None,
                normalized_eval,
                _normalize_question(train_row["instruction"]),
            ).ratio()
            if ratio >= near_duplicate_threshold:
                near_leak_count += 1
                issues.append(
                    FactQualityIssue(
                        code="train_eval_near_leak",
                        severity="error",
                        message=(
                            "An eval question is too similar to a training question "
                            f"({ratio:.2f} similarity)."
                        ),
                        row_ids=(
                            *_row_ids_for_instruction(manifest_by_instruction, train_row["instruction"]),
                            *_row_ids_for_instruction(manifest_by_instruction, eval_row["instruction"]),
                        ),
                    )
                )
                break

    missing_expected_term_count = 0
    for manifest_row in split.manifest_rows:
        expected_terms = tuple(str(term) for term in manifest_row.get("expected_terms", ()))
        response = str(manifest_row.get("response", ""))
        missing_terms = _missing_terms(response, expected_terms)
        if missing_terms:
            missing_expected_term_count += 1
            issues.append(
                FactQualityIssue(
                    code="missing_expected_term",
                    severity="error",
                    message="A train/eval response does not contain its expected fact term(s).",
                    row_ids=(str(manifest_row.get("row_id", "")),),
                )
            )

    unknown_source_chunk_count = 0
    for row in (*train_rows, *eval_rows):
        if row["source_chunk_id"] not in known_chunk_ids:
            unknown_source_chunk_count += 1
            issues.append(
                FactQualityIssue(
                    code="unknown_source_chunk_id",
                    severity="error",
                    message=f"Row references unknown source chunk ID: {row['source_chunk_id']}.",
                )
            )

    train_fact_ids = {
        str(row.get("fact_id"))
        for row in split.manifest_rows
        if row.get("split") == "train" and str(row.get("fact_id")) in known_fact_ids
    }
    eval_fact_ids = {
        str(row.get("fact_id"))
        for row in split.manifest_rows
        if row.get("split") == "eval" and str(row.get("fact_id")) in known_fact_ids
    }

    return FactQualityGateReport(
        fact_count=len(split.facts),
        train_row_count=len(train_rows),
        eval_row_count=len(eval_rows),
        train_fact_coverage=len(train_fact_ids),
        eval_fact_coverage=len(eval_fact_ids),
        exact_leak_count=exact_leak_count,
        near_leak_count=near_leak_count,
        missing_expected_term_count=missing_expected_term_count,
        unknown_source_chunk_count=unknown_source_chunk_count,
        issues=tuple(issues),
    )


def format_fact_quality_report(report: FactQualityGateReport) -> list[str]:
    """Format a beginner-readable fact-ledger quality gate report."""

    lines = [
        "Fact-ledger quality gate",
        f"Facts: {report.fact_count}",
        f"Train rows: {report.train_row_count}",
        f"Held-out eval rows: {report.eval_row_count}",
        f"Fact coverage: train {report.train_fact_coverage}/{report.fact_count}, eval {report.eval_fact_coverage}/{report.fact_count}",
        f"Train/eval leakage: {report.exact_leak_count} exact, {report.near_leak_count} near-duplicate",
        f"Expected-term checks: {report.missing_expected_term_count} missing expected term(s)",
    ]
    if report.passes_required_checks:
        lines.append("Fact-ledger checks passed; this split is ready for a bounded training smoke.")
        return lines

    lines.append("Fact-ledger checks failed:")
    lines.extend(f"- [{issue.severity}] {issue.message}" for issue in report.issues)
    return lines


def score_fact_answer(
    answer: str,
    expected_terms: Sequence[str],
    *,
    question: str = "",
) -> FactAnswerScore:
    """Score one answer by requiring every expected term to appear."""

    terms = tuple(term for term in expected_terms if str(term).strip())
    return FactAnswerScore(
        question=question,
        answer=answer,
        expected_terms=terms,
        missing_terms=_missing_terms(answer, terms),
    )


def score_fact_outputs(outputs: Iterable[Mapping[str, object]]) -> FactOutputScore:
    """Score multiple answer payloads with ``answer`` and ``expected_terms``."""

    items = []
    for output in outputs:
        items.append(
            score_fact_answer(
                str(output.get("answer", "")),
                [str(term) for term in output.get("expected_terms", ())],
                question=str(output.get("question", "")),
            )
        )
    return FactOutputScore(items=tuple(items))


def _manifest_row(
    *,
    row_id: str,
    split: str,
    fact: FactCard,
    row: Mapping[str, str],
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "fact_id": fact.fact_id,
        "split": split,
        "source_chunk_id": fact.source_chunk_id,
        "source_hash": fact.source_hash,
        "fact_kind": fact.fact_kind,
        "label": fact.label,
        "expected_terms": list(fact.expected_terms),
        "instruction": row["instruction"],
        "response": row["response"],
    }


def _train_templates(fact: FactCard) -> tuple[tuple[str, str], ...]:
    label = fact.label.lower()
    return (
        (
            f"What exact value does the note give for {label}?",
            f"The notes state that {label} is {fact.value}.",
        ),
        (
            f"Answer from the notes: identify {label}.",
            f"From the notes, {label} is {fact.value}.",
        ),
        (
            f"Create a study flashcard for {label}.",
            f"Front: What is {label}? Back: {fact.value}.",
        ),
        (
            f"Which value should be remembered for {label}?",
            f"Remember that {label} is {fact.value}.",
        ),
    )


def _extract_label_value_pairs(text: str) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        cleaned_line = raw_line.strip().lstrip("-*").strip()
        for sentence in re.split(r"(?<=\.)\s+", cleaned_line):
            candidate = sentence.strip()
            if ":" not in candidate or candidate.startswith("#"):
                continue
            label, value = candidate.split(":", 1)
            label = _clean_label(label)
            value = value.strip().rstrip(".")
            if not label or not value:
                continue
            if len(label.split()) > 6:
                continue
            facts.append((label, value))
    return facts


def _clean_label(label: str) -> str:
    cleaned = re.sub(r"^\d+[.)]\s*", "", label.strip())
    return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned


def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _normalize_question(question: str) -> str:
    lowered = question.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _missing_terms(answer: str, expected_terms: Sequence[str]) -> tuple[str, ...]:
    normalized_answer = answer.lower()
    return tuple(term for term in expected_terms if term.lower() not in normalized_answer)


def _row_ids_for_instruction(
    manifest_by_instruction: Mapping[str, Mapping[str, object]],
    instruction: str,
) -> tuple[str, ...]:
    row = manifest_by_instruction.get(instruction)
    if not row:
        return ()
    return (str(row.get("row_id", "")),)


__all__ = [
    "FactAnswerScore",
    "FactCard",
    "FactOutputScore",
    "FactQualityGateReport",
    "FactQualityIssue",
    "FactTrainEvalSplit",
    "analyze_fact_quality_gate",
    "build_fact_train_eval_split",
    "extract_fact_ledger",
    "format_fact_quality_report",
    "score_fact_answer",
    "score_fact_outputs",
]
