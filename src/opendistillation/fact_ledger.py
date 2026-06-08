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

    V0 intentionally accepts only explicit note facts. ``Label: value`` covers
    the sample notes and many beginner study notes. Safe bullet/list pairs such
    as ``- Label - value`` or ``1. Label = value`` are also accepted when the
    label is short enough to avoid turning ordinary prose into fake facts.
    """

    facts: list[FactCard] = []
    seen: set[tuple[str, str, str]] = set()
    for chunk in chunks:
        source_hash = _source_hash(chunk.text)
        for label, value, fact_kind in _extract_note_facts(chunk.text):
            fingerprint = (chunk.id, _normalize_fact_text(label), _normalize_fact_text(value))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            fact_id = f"fact-{len(facts) + 1:04d}"
            facts.append(
                FactCard(
                    fact_id=fact_id,
                    source_chunk_id=chunk.id,
                    label=label,
                    value=value,
                    expected_terms=(value,),
                    fact_kind=fact_kind,
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
            "instruction": (
                "During a closed-book check, which answer belongs with the "
                f'note field "{fact.label.lower()}"?'
            ),
            "response": f"The held-out answer for {fact.label.lower()} is {fact.value}.",
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
    token_overlap_threshold: float = 0.68,
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
            sequence_ratio = SequenceMatcher(
                None,
                normalized_eval,
                _normalize_question(train_row["instruction"]),
            ).ratio()
            token_overlap = _token_jaccard(
                _question_tokens(eval_row["instruction"]),
                _question_tokens(train_row["instruction"]),
            )
            if sequence_ratio >= near_duplicate_threshold or token_overlap >= token_overlap_threshold:
                near_leak_count += 1
                issues.append(
                    FactQualityIssue(
                        code="train_eval_near_leak",
                        severity="error",
                        message=(
                            "An eval question is too similar to a training question "
                            f"({sequence_ratio:.2f} sequence similarity, "
                            f"{token_overlap:.2f} token overlap)."
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
    if len(train_fact_ids) < len(known_fact_ids):
        missing = tuple(sorted(known_fact_ids - train_fact_ids))
        issues.append(
            FactQualityIssue(
                code="missing_train_fact_coverage",
                severity="error",
                message="Training rows do not cover fact(s): " + ", ".join(missing) + ".",
                row_ids=missing,
            )
        )
    if len(eval_fact_ids) < len(known_fact_ids):
        missing = tuple(sorted(known_fact_ids - eval_fact_ids))
        issues.append(
            FactQualityIssue(
                code="missing_eval_fact_coverage",
                severity="error",
                message="Held-out eval rows do not cover fact(s): " + ", ".join(missing) + ".",
                row_ids=missing,
            )
        )

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
        "A leakage failure means the model could pass by memorizing a copied question instead of learning the note fact.",
        "Expected terms are the exact note details that a correct answer must contain.",
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

    terms = tuple(str(term).strip() for term in expected_terms if str(term).strip())
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
        (
            f"Use the source note to name {label}.",
            f"The source note names {label} as {fact.value}.",
        ),
        (
            f"For a review quiz, give the recorded answer for {label}.",
            f"The recorded answer for {label} is {fact.value}.",
        ),
        (
            f"State the notes-only answer linked to {label}.",
            f"The notes-only answer linked to {label} is {fact.value}.",
        ),
        (
            f"Turn {label} into a short recall answer.",
            f"Recall answer: {fact.value}.",
        ),
    )


def _extract_note_facts(text: str) -> list[tuple[str, str, str]]:
    facts: list[tuple[str, str, str]] = []
    for raw_line in text.splitlines():
        if _is_markdown_heading(raw_line):
            continue
        list_body = _list_item_body(raw_line)
        if list_body:
            parsed_list_pair = _parse_list_pair(list_body)
            if parsed_list_pair:
                label, value = parsed_list_pair
                facts.append((label, value, "list_pair"))
                continue
        cleaned_line = (list_body or raw_line).strip().lstrip("-*").strip()
        for sentence in re.split(r"(?<=\.)\s+", cleaned_line):
            candidate = sentence.strip()
            if ":" not in candidate or candidate.startswith("#"):
                continue
            label, value = candidate.split(":", 1)
            if _has_list_separator(label):
                continue
            label = _clean_label(label)
            value = value.strip().rstrip(".")
            if not _safe_fact_label(label) or not _safe_fact_value(value):
                continue
            facts.append((label, value, "label_value"))
    return facts


def _clean_label(label: str) -> str:
    cleaned = re.sub(r"^\d+[.)]\s*", "", label.strip().strip("*_`"))
    return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned


def _clean_value(value: str) -> str:
    return value.strip().strip("*_`").rstrip(".")


def _is_markdown_heading(line: str) -> bool:
    return bool(re.match(r"^\s{0,3}#{1,6}\s+", line))


def _list_item_body(line: str) -> str:
    match = re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+?)\s*$", line)
    return match.group(1).strip() if match else ""


def _parse_list_pair(body: str) -> tuple[str, str] | None:
    match = re.match("(.+?)\\s+(?:--?|=|[\u2013\u2014])\\s+(.+)", body)
    if not match:
        return None
    label = _clean_label(match.group(1))
    value = _clean_value(match.group(2))
    if not _safe_fact_label(label) or not _safe_fact_value(value):
        return None
    return label, value


def _has_list_separator(text: str) -> bool:
    return bool(re.search("\\s(?:--?|=|[\u2013\u2014])\\s", text))


def _safe_fact_label(label: str) -> bool:
    if not label:
        return False
    if len(label.split()) > 6:
        return False
    if not re.search(r"[A-Za-z]", label):
        return False
    if re.search(r"[.!?]", label):
        return False
    return True


def _safe_fact_value(value: str) -> bool:
    if not value:
        return False
    if len(value.split()) > 30:
        return False
    return bool(re.search(r"[A-Za-z0-9]", value))


def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _normalize_fact_text(text: str) -> str:
    return " ".join(_text_tokens(text))


def _normalize_question(question: str) -> str:
    lowered = question.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _missing_terms(answer: str, expected_terms: Sequence[str]) -> tuple[str, ...]:
    answer_tokens = _text_tokens(answer)
    return tuple(term for term in expected_terms if not _answer_contains_expected_term(answer, answer_tokens, term))


def _answer_contains_expected_term(answer: str, answer_tokens: Sequence[str], expected_term: str) -> bool:
    term = expected_term.strip()
    if not term:
        return False
    if _requires_surface_match(term):
        return bool(_surface_term_pattern(term).search(answer))
    return _contains_token_phrase(answer_tokens, _text_tokens(term))


def _requires_surface_match(term: str) -> bool:
    return bool(re.search(r"[^\w\s]", term))


def _surface_term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.strip())
    escaped = re.sub(r"(?:\\\s)+", r"\\s+", escaped)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", flags=re.IGNORECASE)


def _text_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.lower()))


def _contains_token_phrase(tokens: Sequence[str], phrase_tokens: Sequence[str]) -> bool:
    if not phrase_tokens:
        return False
    phrase_length = len(phrase_tokens)
    return any(tuple(tokens[index : index + phrase_length]) == tuple(phrase_tokens) for index in range(len(tokens)))


def _question_tokens(question: str) -> set[str]:
    return set(_text_tokens(question))


def _token_jaccard(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


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
