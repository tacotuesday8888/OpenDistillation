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


DEFAULT_FACT_TRAIN_EXAMPLES_PER_FACT = 6
DEFAULT_NEXT_SMOKE_REQUIRED_EXACT_HITS = 3
DEFAULT_NEXT_SMOKE_MAX_INVENTED_VALUE_MISSES = 5
_ANTI_INVENTION_WARNING_TEXT = "Do not invent a new number, time, identifier, name, or color."


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
    missing_manifest_metadata_count: int
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
    fact_id: str = ""
    label: str = ""
    value: str = ""
    source_chunk_id: str = ""
    row_style: str = ""
    unscored_reason: str = ""

    @property
    def scored(self) -> bool:
        return not self.unscored_reason

    @property
    def hit(self) -> bool:
        return self.scored and not self.missing_terms


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

    @property
    def unscored_count(self) -> int:
        return sum(1 for item in self.items if not item.scored)


@dataclass(frozen=True)
class FactValueMatch:
    """Known fact value found in a wrong model answer."""

    fact_id: str
    label: str
    value: str
    source_chunk_id: str


@dataclass(frozen=True)
class FactMissDiagnostic:
    """Local explanation for one exact fact miss."""

    miss_kind: str
    question: str
    answer: str
    expected_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    fact_id: str = ""
    label: str = ""
    value: str = ""
    source_chunk_id: str = ""
    row_style: str = ""
    value_matches: tuple[FactValueMatch, ...] = ()
    invented_values: tuple[str, ...] = ()
    shape_markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactMissDiagnosticReport:
    """Summary of local failure patterns for exact fact misses."""

    answer_count: int
    hit_count: int
    miss_count: int
    unscored_count: int
    items: tuple[FactMissDiagnostic, ...]

    def count(self, miss_kind: str) -> int:
        return sum(1 for item in self.items if item.miss_kind == miss_kind)


@dataclass(frozen=True)
class FactInventedValueCandidate:
    """Plausible-looking wrong value produced by the model."""

    text: str
    value_shapes: tuple[str, ...]


@dataclass(frozen=True)
class FactMissTrainingRowSignal:
    """One internal training row tied to a missed fact."""

    row_id: str
    row_style: str
    instruction: str
    response: str
    public_row_fields: tuple[str, ...]
    exact_value_in_instruction: bool
    exact_value_in_response: bool
    known_values_only_warning_present: bool


@dataclass(frozen=True)
class FactMissContextItem:
    """Local context joining a missed answer back to its training signal."""

    fact_id: str
    label: str
    source_chunk_id: str
    question: str
    trained_answer: str
    expected_value: str
    expected_value_shapes: tuple[str, ...]
    missing_terms: tuple[str, ...]
    miss_kind: str
    invented_values: tuple[FactInventedValueCandidate, ...]
    row_styles_seen: tuple[str, ...]
    train_row_count: int
    exact_value_in_prompt_rows: int
    exact_value_in_completion_rows: int
    known_values_only_warning_present: bool
    same_chunk_known_values: tuple[str, ...]
    training_rows: tuple[FactMissTrainingRowSignal, ...]
    diagnosis: str


@dataclass(frozen=True)
class FactMissContextReport:
    """Post-run local audit for deciding whether another GPU smoke is justified."""

    answer_count: int
    expected_answer_count: int
    trained_exact_hits: int
    trained_misses: int
    invented_value_misses: int
    required_trained_exact_hits: int
    maximum_invented_value_misses: int
    items: tuple[FactMissContextItem, ...]
    unscored_count: int = 0

    @property
    def passes_next_smoke_gate(self) -> bool:
        return not self.failure_reasons

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        expected_answer_count = self.expected_answer_count or self.answer_count
        if self.expected_answer_count and self.answer_count != self.expected_answer_count:
            reasons.append(
                f"answer count {self.answer_count}/{self.expected_answer_count} "
                "does not match held-out eval rows"
            )
        if self.trained_exact_hits < self.required_trained_exact_hits:
            reasons.append(
                f"trained exact hits {self.trained_exact_hits}/{expected_answer_count} "
                f"are below required {self.required_trained_exact_hits}/{expected_answer_count}"
            )
        if self.invented_value_misses > self.maximum_invented_value_misses:
            reasons.append(
                f"invented-value misses {self.invented_value_misses}/{expected_answer_count} "
                f"exceed maximum {self.maximum_invented_value_misses}/{expected_answer_count}"
            )
        if self.unscored_count:
            reasons.append(f"unscored answers {self.unscored_count}/{expected_answer_count} need expected terms")
        return tuple(reasons)


@dataclass(frozen=True)
class FactScoreComparisonItem:
    """One base/trained exact fact outcome."""

    question: str
    expected_terms: tuple[str, ...]
    base_answer: str
    trained_answer: str
    base_hit: bool
    trained_hit: bool
    base_missing_terms: tuple[str, ...]
    trained_missing_terms: tuple[str, ...]
    outcome: str
    fact_id: str = ""
    label: str = ""
    value: str = ""
    source_chunk_id: str = ""
    row_style: str = ""


@dataclass(frozen=True)
class FactScoreComparison:
    """Per-fact outcome summary for exact fact-hit scoring."""

    items: tuple[FactScoreComparisonItem, ...]

    @property
    def learned_count(self) -> int:
        return self._count("learned")

    @property
    def missed_count(self) -> int:
        return self._count("missed")

    @property
    def unchanged_count(self) -> int:
        return self._count("unchanged")

    @property
    def worse_count(self) -> int:
        return self._count("worse")

    @property
    def unscored_count(self) -> int:
        return self._count("unscored")

    def _count(self, outcome: str) -> int:
        return sum(1 for item in self.items if item.outcome == outcome)


@dataclass(frozen=True)
class FactReadinessReport:
    """Plain-language local readiness summary before spending GPU time."""

    fact_count: int
    train_row_count: int
    eval_row_count: int
    train_examples_per_fact: int
    label_value_fact_coverage: int
    label_value_train_row_count: int
    contrastable_fact_count: int
    disambiguation_fact_coverage: int
    disambiguation_train_row_count: int
    known_values_only_fact_coverage: int
    known_values_only_train_row_count: int
    sft_preview_row_count: int
    quality_report: FactQualityGateReport

    @property
    def ready_for_gpu_smoke(self) -> bool:
        return (
            self.fact_count > 0
            and self.quality_report.passes_required_checks
            and self.train_examples_per_fact >= DEFAULT_FACT_TRAIN_EXAMPLES_PER_FACT
            and self.label_value_fact_coverage == self.fact_count
            and self.label_value_train_row_count == self.train_row_count
            and (
                self.contrastable_fact_count == 0
                or (
                    self.disambiguation_fact_coverage == self.contrastable_fact_count
                    and self.disambiguation_train_row_count
                    >= self.contrastable_fact_count * len(_DISAMBIGUATION_ROW_STYLES)
                    and self.known_values_only_fact_coverage == self.contrastable_fact_count
                    and self.known_values_only_train_row_count >= self.contrastable_fact_count
                )
            )
            and self.sft_preview_row_count > 0
        )

    @property
    def skip_reason(self) -> str:
        """Machine-readable reason when rows are not ready for a GPU smoke."""

        if self.ready_for_gpu_smoke:
            return ""
        if self.fact_count == 0:
            return "no_fact_ledger_facts"
        if not self.quality_report.passes_required_checks:
            return "fact_ledger_quality_gate_failed"
        if self.train_examples_per_fact < DEFAULT_FACT_TRAIN_EXAMPLES_PER_FACT:
            return "insufficient_fact_training_rows"
        if self.label_value_fact_coverage < self.fact_count or self.label_value_train_row_count < self.train_row_count:
            return "missing_label_value_training_signal"
        if self.contrastable_fact_count and (
            self.disambiguation_fact_coverage < self.contrastable_fact_count
            or self.disambiguation_train_row_count < self.contrastable_fact_count * len(_DISAMBIGUATION_ROW_STYLES)
        ):
            return "missing_label_value_disambiguation_signal"
        if self.contrastable_fact_count and (
            self.known_values_only_fact_coverage < self.contrastable_fact_count
            or self.known_values_only_train_row_count < self.contrastable_fact_count
        ):
            return "missing_known_values_only_signal"
        if self.sft_preview_row_count == 0:
            return "missing_sft_preview"
        return "fact_ledger_not_training_ready"


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
    train_examples_per_fact: int = DEFAULT_FACT_TRAIN_EXAMPLES_PER_FACT,
) -> FactTrainEvalSplit:
    """Build public train rows, held-out eval rows, and sidecar metadata."""

    if train_examples_per_fact < 1:
        raise ValueError("train_examples_per_fact must be at least 1")

    fact_tuple = tuple(facts)
    if not fact_tuple:
        return FactTrainEvalSplit(facts=(), train_rows=(), eval_rows=(), manifest_rows=())

    train_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, object]] = []

    facts_by_chunk: dict[str, tuple[FactCard, ...]] = {}
    for fact in fact_tuple:
        facts_by_chunk.setdefault(fact.source_chunk_id, ())
        facts_by_chunk[fact.source_chunk_id] = (*facts_by_chunk[fact.source_chunk_id], fact)

    for fact in fact_tuple:
        contrast_fact = _contrast_fact_for(fact, facts_by_chunk)
        train_templates = _train_templates(
            fact,
            contrast_fact=contrast_fact,
            same_chunk_facts=facts_by_chunk.get(fact.source_chunk_id, ()),
        )
        for index in range(train_examples_per_fact):
            row_style, instruction, response = train_templates[index % len(train_templates)]
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
                    row_style=row_style,
                    contrast_fact=contrast_fact if row_style in _CONTRAST_ROW_STYLES else None,
                )
            )

        eval_row = {
            "instruction": (
                "Closed-book check: what exact notes value should be recalled "
                f'for "{fact.label.lower()}"?'
            ),
            "response": f"Exact answer: {fact.value}.",
            "source_chunk_id": fact.source_chunk_id,
        }
        eval_rows.append(eval_row)
        manifest_rows.append(
            _manifest_row(
                row_id=f"eval-{len(eval_rows):06d}",
                split="eval",
                fact=fact,
                row=eval_row,
                row_style="held_out_direct_recall",
            )
        )

    return FactTrainEvalSplit(
        facts=fact_tuple,
        train_rows=tuple(validate_dataset(train_rows)),
        eval_rows=tuple(validate_dataset(eval_rows)),
        manifest_rows=tuple(manifest_rows),
    )


def build_fact_comparison_rows(split: FactTrainEvalSplit) -> tuple[dict[str, object], ...]:
    """Return held-out eval rows with internal fact metadata attached.

    The public JSONL schema remains ``instruction``, ``response``, and
    ``source_chunk_id``. These enriched rows are for in-memory comparison and
    scoring only, so expected terms follow the selected question even if the
    comparison helper reorders rows for source-chunk diversity.
    """

    if not split.eval_rows:
        return ()

    manifest_by_public_row = {
        _public_row_key(row): row
        for row in split.manifest_rows
        if row.get("split") == "eval"
    }
    enriched_rows: list[dict[str, object]] = []
    for row in validate_dataset(split.eval_rows):
        enriched: dict[str, object] = dict(row)
        manifest_row = manifest_by_public_row.get(_public_row_key(row), {})
        for field in ("row_id", "fact_id", "label", "value", "row_style"):
            value = manifest_row.get(field)
            if isinstance(value, str) and value:
                enriched[field] = value
        expected_terms = tuple(str(term).strip() for term in manifest_row.get("expected_terms", ()) if str(term).strip())
        if expected_terms:
            enriched["expected_terms"] = list(expected_terms)
        enriched_rows.append(enriched)
    return tuple(enriched_rows)


def analyze_fact_quality_gate(
    split: FactTrainEvalSplit,
    *,
    near_duplicate_threshold: float = 0.86,
    token_overlap_threshold: float = 0.68,
) -> FactQualityGateReport:
    """Check the fact-ledger train/eval split for leakage and coverage."""

    issues: list[FactQualityIssue] = []
    train_rows = tuple(validate_dataset(split.train_rows)) if split.train_rows else ()
    eval_rows = tuple(validate_dataset(split.eval_rows)) if split.eval_rows else ()
    known_fact_ids = {fact.fact_id for fact in split.facts}
    known_chunk_ids = {fact.source_chunk_id for fact in split.facts}
    manifest_by_instruction = {
        str(row["instruction"]): row
        for row in split.manifest_rows
        if isinstance(row.get("instruction"), str)
    }
    manifest_by_public_row = {
        _public_row_key(row): row
        for row in split.manifest_rows
        if row.get("split") in {"train", "eval"}
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

    missing_manifest_metadata_count = 0
    for split_name, public_rows in (("train", train_rows), ("eval", eval_rows)):
        for row in public_rows:
            manifest_row = manifest_by_public_row.get(_public_row_key(row))
            if _has_complete_manifest_metadata(manifest_row, split_name=split_name):
                continue
            missing_manifest_metadata_count += 1
            issues.append(
                FactQualityIssue(
                    code=f"missing_{split_name}_manifest_metadata",
                    severity="error",
                    message=(
                        f"A {split_name} row does not have matching complete fact-ledger "
                        "manifest metadata. The row may be stale or detached from its expected terms."
                    ),
                    row_ids=_row_ids_for_instruction(manifest_by_instruction, row["instruction"]),
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
        missing_manifest_metadata_count=missing_manifest_metadata_count,
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
        f"Manifest metadata checks: {report.missing_manifest_metadata_count} row(s) missing fact metadata",
        "A leakage failure means the model could pass by memorizing a copied question instead of learning the note fact.",
        "Expected terms are the exact note details that a correct answer must contain.",
    ]
    if report.passes_required_checks:
        if report.fact_count == 0:
            lines.append("No explicit facts were extracted, so fact-ledger training and held-out fact scoring are skipped.")
            lines.append("Generated teacher rows can still be previewed, but they do not prove held-out fact learning.")
            return lines

        lines.append("Fact-ledger checks passed; this split is safe enough for a bounded training smoke.")
        lines.append("This does not prove the model will learn; it only proves the local train/eval split is separated and checkable.")
        return lines

    lines.append("Fact-ledger checks failed:")
    lines.extend(f"- [{issue.severity}] {issue.message}" for issue in report.issues)
    return lines


def score_fact_answer(
    answer: str,
    expected_terms: Sequence[str],
    *,
    question: str = "",
    fact_id: str = "",
    label: str = "",
    value: str = "",
    source_chunk_id: str = "",
    row_style: str = "",
) -> FactAnswerScore:
    """Score one answer by requiring every expected term to appear."""

    terms = tuple(str(term).strip() for term in expected_terms if str(term).strip())
    unscored_reason = "" if terms else "missing_expected_terms"
    return FactAnswerScore(
        question=question,
        answer=answer,
        expected_terms=terms,
        missing_terms=_missing_terms(answer, terms) if terms else (),
        fact_id=fact_id,
        label=label,
        value=value,
        source_chunk_id=source_chunk_id,
        row_style=row_style,
        unscored_reason=unscored_reason,
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
                fact_id=str(output.get("fact_id", "")),
                label=str(output.get("label", "")),
                value=str(output.get("value", "")),
                source_chunk_id=str(output.get("source_chunk_id", "")),
                row_style=str(output.get("row_style", "")),
            )
        )
    return FactOutputScore(items=tuple(items))


def compare_fact_scores(base_score: FactOutputScore, trained_score: FactOutputScore) -> FactScoreComparison:
    """Compare base and trained exact fact hits row by row."""

    if base_score.answer_count != trained_score.answer_count:
        raise ValueError("base and trained scores must cover the same number of answers")

    items: list[FactScoreComparisonItem] = []
    for base_item, trained_item in zip(base_score.items, trained_score.items, strict=True):
        _validate_score_alignment(base_item, trained_item)
        items.append(
            FactScoreComparisonItem(
                question=trained_item.question or base_item.question,
                expected_terms=trained_item.expected_terms or base_item.expected_terms,
                base_answer=base_item.answer,
                trained_answer=trained_item.answer,
                base_hit=base_item.hit,
                trained_hit=trained_item.hit,
                base_missing_terms=base_item.missing_terms,
                trained_missing_terms=trained_item.missing_terms,
                outcome=_fact_score_outcome(base_item, trained_item),
                fact_id=trained_item.fact_id or base_item.fact_id,
                label=trained_item.label or base_item.label,
                value=trained_item.value or base_item.value,
                source_chunk_id=trained_item.source_chunk_id or base_item.source_chunk_id,
                row_style=trained_item.row_style or base_item.row_style,
            )
        )
    return FactScoreComparison(items=tuple(items))


def diagnose_fact_misses(
    score: FactOutputScore,
    facts: Iterable[FactCard],
) -> FactMissDiagnosticReport:
    """Explain exact fact misses without changing the exact-hit score."""

    fact_tuple = tuple(facts)
    items = tuple(
        diagnostic
        for item in score.items
        if (diagnostic := _diagnose_fact_answer_miss(item, facts=fact_tuple)) is not None
    )
    return FactMissDiagnosticReport(
        answer_count=score.answer_count,
        hit_count=score.hit_count,
        miss_count=score.miss_count,
        unscored_count=score.unscored_count,
        items=items,
    )


def analyze_fact_miss_contexts(
    split: FactTrainEvalSplit,
    trained_score: FactOutputScore,
    *,
    required_trained_exact_hits: int = DEFAULT_NEXT_SMOKE_REQUIRED_EXACT_HITS,
    maximum_invented_value_misses: int = DEFAULT_NEXT_SMOKE_MAX_INVENTED_VALUE_MISSES,
) -> FactMissContextReport:
    """Join trained exact misses back to the local fact-ledger training signal.

    This is a post-run audit. It does not change scoring and it does not give
    credit for wrong answers. Its job is to explain whether a missed fact had
    clear local supervision before another GPU run is justified.
    """

    if required_trained_exact_hits < 0:
        raise ValueError("required_trained_exact_hits must be non-negative")
    if maximum_invented_value_misses < 0:
        raise ValueError("maximum_invented_value_misses must be non-negative")

    facts = tuple(split.facts)
    diagnostics = diagnose_fact_misses(trained_score, facts)
    context_items = tuple(
        _fact_miss_context_item(diagnostic, split=split, facts=facts)
        for diagnostic in diagnostics.items
    )
    return FactMissContextReport(
        answer_count=trained_score.answer_count,
        expected_answer_count=len(split.eval_rows),
        trained_exact_hits=trained_score.hit_count,
        trained_misses=trained_score.miss_count,
        invented_value_misses=diagnostics.count("invented_numeric_time_identifier_value"),
        required_trained_exact_hits=required_trained_exact_hits,
        maximum_invented_value_misses=maximum_invented_value_misses,
        unscored_count=trained_score.unscored_count,
        items=context_items,
    )


def format_fact_miss_context_report(
    report: FactMissContextReport,
    *,
    max_examples: int = 6,
    max_training_rows_per_fact: int = 6,
) -> list[str]:
    """Format the post-run miss context audit in plain language."""

    if max_examples < 1:
        raise ValueError("max_examples must be at least 1")
    if max_training_rows_per_fact < 1:
        raise ValueError("max_training_rows_per_fact must be at least 1")

    verdict = "passed" if report.passes_next_smoke_gate else "failed"
    expected_answer_count = report.expected_answer_count or report.answer_count
    lines = [
        "Fact miss context report",
        (
            f"Trained exact hits: {report.trained_exact_hits}/{expected_answer_count}; "
            f"misses: {report.trained_misses}/{expected_answer_count}; "
            f"invented-value misses: {report.invented_value_misses}/{expected_answer_count}"
        ),
        (
            f"Next smoke gate: {verdict}; requires at least "
            f"{report.required_trained_exact_hits}/{expected_answer_count} exact hits "
            f"and at most {report.maximum_invented_value_misses}/{expected_answer_count} "
            "invented-value misses."
        ),
        "This audit explains failures; it does not give credit unless exact expected terms are present.",
    ]
    for reason in report.failure_reasons:
        lines.append(f"Gate failure: {reason}.")
    if not report.items:
        lines.append("No scored miss contexts to inspect.")
        return lines

    for item in report.items[:max_examples]:
        lines.append(f"Miss context: {_fact_miss_context_label(item)} | {item.miss_kind}")
        lines.append(
            "  expected value: "
            + item.expected_value
            + _format_shape_suffix(item.expected_value_shapes)
        )
        lines.append(f"  trained answer: {item.trained_answer}")
        if item.invented_values:
            invented = ", ".join(
                candidate.text + _format_shape_suffix(candidate.value_shapes)
                for candidate in item.invented_values
            )
            lines.append("  invented value candidate(s): " + invented)
        if item.same_chunk_known_values:
            lines.append("  same-chunk known value(s): " + "; ".join(item.same_chunk_known_values))
        lines.append(
            "  training signal: "
            f"{item.train_row_count} row(s); row styles: "
            + (", ".join(item.row_styles_seen) if item.row_styles_seen else "(none)")
        )
        lines.append(
            "  exact value presence: "
            f"{item.exact_value_in_prompt_rows} prompt row(s), "
            f"{item.exact_value_in_completion_rows} completion row(s)"
        )
        lines.append(
            "  known-values warning present: "
            + ("yes" if item.known_values_only_warning_present else "no")
        )
        lines.append("  diagnosis: " + item.diagnosis)
        for row in item.training_rows[:max_training_rows_per_fact]:
            lines.append(f"  train row {row.row_id} | {row.row_style}")
            lines.append("    prompt: " + row.instruction)
            lines.append("    completion: " + row.response)
    return lines


def format_fact_miss_diagnostic_report(
    report: FactMissDiagnosticReport,
    *,
    max_examples: int = 8,
) -> list[str]:
    """Format local exact-miss diagnostics in plain language."""

    if max_examples < 1:
        raise ValueError("max_examples must be at least 1")

    lines = [
        "Fact miss diagnostic report",
        (
            f"Answers checked: {report.answer_count}; exact hits: {report.hit_count}; "
            f"exact misses: {report.miss_count}; unscored: {report.unscored_count}"
        ),
        (
            "Miss patterns: "
            f"same-chunk value confusion {report.count('same_chunk_value_confusion')}, "
            f"known value confusion {report.count('known_value_confusion')}, "
            f"invented numeric/time/identifier value {report.count('invented_numeric_time_identifier_value')}, "
            f"label echo {report.count('label_echo')}, "
            f"answer shape without fact {report.count('answer_shape_without_fact')}, "
            f"generic miss {report.count('generic_miss')}"
        ),
        "Diagnostics explain wrong answers; they do not give credit unless exact expected terms are present.",
    ]
    if report.count("invented_numeric_time_identifier_value"):
        lines.append("Invented-value warning: at least one answer gave a plausible-looking value that is not in the fact ledger.")
    if report.count("same_chunk_value_confusion") or report.count("known_value_confusion"):
        lines.append("Known-value warning: at least one answer used a real note value for the wrong label.")
    if not report.items:
        if report.unscored_count:
            lines.append("No scored misses were available for diagnosis because expected terms were missing.")
        else:
            lines.append("No exact misses to diagnose.")
        return lines

    for item in report.items[:max_examples]:
        lines.append(f"Miss example: {_fact_miss_label(item)} | {item.miss_kind}")
        if item.expected_terms:
            lines.append("  expected: " + ", ".join(item.expected_terms))
        lines.append(f"  model answered: {item.answer}")
        if item.value_matches:
            matched = ", ".join(f"{match.value} ({match.label})" for match in item.value_matches)
            lines.append("  wrong known value(s): " + matched)
        if item.invented_values:
            lines.append("  invented value candidate(s): " + ", ".join(item.invented_values))
        if item.shape_markers:
            lines.append("  answer-shape marker(s): " + ", ".join(item.shape_markers))
        lines.append("  why this matters: changed wording is still failed learning when the exact note value is missing.")
    return lines


def format_fact_score_report(
    base_score: FactOutputScore,
    trained_score: FactOutputScore,
    *,
    changed_answer_count: int | None = None,
) -> list[str]:
    """Format exact fact-hit scores for beginner-readable model reporting."""

    if base_score.answer_count != trained_score.answer_count:
        raise ValueError("base and trained scores must cover the same number of answers")

    comparison = compare_fact_scores(base_score, trained_score)
    total = trained_score.answer_count
    if trained_score.hit_count > base_score.hit_count:
        judgment = "better"
    elif trained_score.hit_count < base_score.hit_count:
        judgment = "worse"
    else:
        judgment = "unchanged"

    lines = [
        "Exact fact-hit quality report",
        f"Base exact fact hits: {base_score.hit_count}/{total}",
        f"Trained exact fact hits: {trained_score.hit_count}/{total}",
    ]
    if changed_answer_count is not None:
        lines.append(f"Changed answers: {changed_answer_count}/{total}")
    lines.append(f"Judgment: {judgment}")
    lines.append(
        "Outcome counts: "
        f"learned {comparison.learned_count}, "
        f"missed {comparison.missed_count}, "
        f"unchanged {comparison.unchanged_count}, "
        f"worse {comparison.worse_count}"
    )
    if comparison.unscored_count:
        lines.append(f"Unscored answers: {comparison.unscored_count}/{total} missing expected terms")
    lines.append("Exact fact hits require the expected note value to appear in the answer.")
    if changed_answer_count and trained_score.hit_count <= base_score.hit_count:
        lines.append("Changed answers with wrong facts are still a failure, not learned note memory.")
    if trained_score.hit_count == 0:
        lines.append("The trained adapter did not hit any checked facts.")
    for outcome in ("learned", "missed", "worse"):
        for item in comparison.items:
            if item.outcome != outcome:
                continue
            title = {"learned": "Learned", "missed": "Missed", "worse": "Worse"}[outcome]
            lines.append(f"{title} fact: {_fact_score_label(item)}")
            if item.expected_terms:
                lines.append("  expected term(s): " + ", ".join(item.expected_terms))
            if item.trained_missing_terms:
                lines.append("  trained missing term(s): " + ", ".join(item.trained_missing_terms))
            lines.append(f"  base answer: {item.base_answer}")
            lines.append(f"  trained answer: {item.trained_answer}")
    return lines


def analyze_fact_readiness(
    split: FactTrainEvalSplit,
    *,
    sft_preview_row_count: int = 0,
) -> FactReadinessReport:
    """Summarize whether local fact rows are ready for one bounded GPU smoke."""

    if sft_preview_row_count < 0:
        raise ValueError("sft_preview_row_count must be non-negative")

    quality_report = analyze_fact_quality_gate(split)
    fact_count = len(split.facts)
    train_row_count = len(validate_dataset(split.train_rows)) if split.train_rows else 0
    train_examples_per_fact = train_row_count // fact_count if fact_count and train_row_count % fact_count == 0 else 0
    label_value_rows_by_fact: dict[str, int] = {}
    facts_by_id = {fact.fact_id: fact for fact in split.facts}
    contrastable_fact_ids = _contrastable_fact_ids(split.facts)
    disambiguation_styles_by_fact: dict[str, set[str]] = {}
    disambiguation_train_row_count = 0
    known_values_only_fact_ids: set[str] = set()
    known_values_only_train_row_count = 0
    for manifest_row in split.manifest_rows:
        if manifest_row.get("split") != "train":
            continue
        fact_id = str(manifest_row.get("fact_id", ""))
        binding = _canonical_label_value_binding(
            str(manifest_row.get("label", "")),
            str(manifest_row.get("value", "")),
        )
        if binding and binding in str(manifest_row.get("response", "")):
            label_value_rows_by_fact[fact_id] = label_value_rows_by_fact.get(fact_id, 0) + 1
        if _has_complete_disambiguation_signal(manifest_row, facts_by_id=facts_by_id):
            row_style = str(manifest_row.get("row_style", ""))
            disambiguation_styles_by_fact.setdefault(fact_id, set()).add(row_style)
            disambiguation_train_row_count += 1
        if _has_complete_known_values_only_signal(manifest_row, facts_by_id=facts_by_id):
            known_values_only_fact_ids.add(fact_id)
            known_values_only_train_row_count += 1

    return FactReadinessReport(
        fact_count=fact_count,
        train_row_count=train_row_count,
        eval_row_count=len(validate_dataset(split.eval_rows)) if split.eval_rows else 0,
        train_examples_per_fact=train_examples_per_fact,
        label_value_fact_coverage=sum(1 for fact in split.facts if label_value_rows_by_fact.get(fact.fact_id, 0) > 0),
        label_value_train_row_count=sum(label_value_rows_by_fact.values()),
        contrastable_fact_count=len(contrastable_fact_ids),
        disambiguation_fact_coverage=sum(
            1
            for fact_id in contrastable_fact_ids
            if _DISAMBIGUATION_ROW_STYLES.issubset(disambiguation_styles_by_fact.get(fact_id, set()))
        ),
        disambiguation_train_row_count=disambiguation_train_row_count,
        known_values_only_fact_coverage=sum(1 for fact_id in contrastable_fact_ids if fact_id in known_values_only_fact_ids),
        known_values_only_train_row_count=known_values_only_train_row_count,
        sft_preview_row_count=sft_preview_row_count,
        quality_report=quality_report,
    )


def format_fact_readiness_report(report: FactReadinessReport) -> list[str]:
    """Format a plain-language ready/not-ready report for the next GPU run."""

    lines = [
        "Fact-ledger label/value readiness report",
        f"Facts: {report.fact_count}",
        f"Train rows: {report.train_row_count}, {report.train_examples_per_fact} per fact",
        f"Held-out eval rows: {report.eval_row_count}",
        (
            "Canonical Label: value bindings: "
            f"{report.label_value_fact_coverage}/{report.fact_count} facts covered, "
            f"{report.label_value_train_row_count} total rows"
        ),
        (
            "Label/value disambiguation rows: "
            f"{report.disambiguation_fact_coverage}/{report.contrastable_fact_count} contrastable facts, "
            f"{report.disambiguation_train_row_count} total rows"
        ),
        (
            "Anti-invention known-values rows: "
            f"{report.known_values_only_fact_coverage}/{report.contrastable_fact_count} contrastable facts, "
            f"{report.known_values_only_train_row_count} total rows"
        ),
        (
            "Train/eval leakage: "
            f"{report.quality_report.exact_leak_count} exact, "
            f"{report.quality_report.near_leak_count} near-duplicate"
        ),
        f"Expected-term checks: {report.quality_report.missing_expected_term_count} missing",
        f"Manifest metadata checks: {report.quality_report.missing_manifest_metadata_count} missing",
        f"SFT preview: {report.sft_preview_row_count} exact prompt/completion row(s)",
    ]
    if report.ready_for_gpu_smoke:
        lines.append("Verdict: local data split is structurally ready; model quality is still unproven")
        lines.append(
            "Reason: local data checks pass, the training rows repeat each Label: value fact, "
            "contrast rows distinguish nearby labels, and known-values rows tell the model to choose "
            "recorded note values instead of inventing substitutes; this does not claim model quality "
            "or by itself justify another GPU run."
        )
        return lines

    lines.append("Verdict: not ready for another GPU training smoke")
    skip_reason = report.skip_reason
    if skip_reason == "fact_ledger_quality_gate_failed":
        lines.append("Reason: fact-ledger quality checks failed.")
    elif skip_reason == "no_fact_ledger_facts":
        lines.append("Reason: no explicit facts were extracted from the TXT/MD notes.")
    elif skip_reason == "insufficient_fact_training_rows":
        lines.append("Reason: each fact needs the stronger default six training rows before another GPU run.")
    elif skip_reason == "missing_label_value_training_signal":
        lines.append("Reason: at least one training row is missing canonical Label: value training signal.")
    elif skip_reason == "missing_label_value_disambiguation_signal":
        lines.append("Reason: contrastable facts need rows that distinguish nearby labels and values.")
    elif skip_reason == "missing_known_values_only_signal":
        lines.append("Reason: contrastable facts need known-values-only rows that discourage invented substitutes.")
    elif skip_reason == "missing_sft_preview":
        lines.append("Reason: inspectable SFT prompt/completion preview has not been produced.")
    else:
        lines.append("Reason: local readiness checks found a problem that should be inspected before training.")
    return lines


_KNOWN_VALUES_ONLY_ROW_STYLE = "known_values_only_label_value"
_DISAMBIGUATION_ROW_STYLES = frozenset({"same_chunk_label_disambiguation"})
_CONTRAST_ROW_STYLES = frozenset({"same_chunk_label_disambiguation", _KNOWN_VALUES_ONLY_ROW_STYLE})
_MAX_KNOWN_VALUE_OPTIONS = 4


def _manifest_row(
    *,
    row_id: str,
    split: str,
    fact: FactCard,
    row: Mapping[str, str],
    row_style: str,
    contrast_fact: FactCard | None = None,
) -> dict[str, object]:
    manifest = {
        "row_id": row_id,
        "fact_id": fact.fact_id,
        "split": split,
        "source_chunk_id": fact.source_chunk_id,
        "source_hash": fact.source_hash,
        "fact_kind": fact.fact_kind,
        "row_style": row_style,
        "label": fact.label,
        "value": fact.value,
        "expected_terms": list(fact.expected_terms),
        "instruction": row["instruction"],
        "response": row["response"],
    }
    if contrast_fact is not None:
        manifest["contrast_label"] = contrast_fact.label
        manifest["contrast_value"] = contrast_fact.value
        manifest["contrast_fact_id"] = contrast_fact.fact_id
    return manifest


def _public_row_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("instruction", "")).strip(),
        str(row.get("response", "")).strip(),
        str(row.get("source_chunk_id", "")).strip(),
    )


def _train_templates(
    fact: FactCard,
    *,
    contrast_fact: FactCard | None = None,
    same_chunk_facts: Sequence[FactCard] = (),
) -> tuple[tuple[str, str, str], ...]:
    label = fact.label.lower()
    binding = _canonical_label_value_binding(fact.label, fact.value)
    contrast_templates: tuple[tuple[str, str, str], ...] = ()
    if contrast_fact is not None:
        contrast_label = contrast_fact.label.lower()
        contrast_binding = _canonical_label_value_binding(contrast_fact.label, contrast_fact.value)
        known_value_options = _known_values_only_options(fact, same_chunk_facts=same_chunk_facts)
        contrast_templates = (
            (
                "same_chunk_label_disambiguation",
                (
                    f"Choose the exact value for {label}, not the value for "
                    f"{contrast_label}."
                ),
                (
                    f"Exact answer: {fact.value}. {binding}. "
                    f"{contrast_fact.value} belongs to {contrast_fact.label}, not {fact.label}. "
                    f"{contrast_binding}."
                ),
            ),
            (
                _KNOWN_VALUES_ONLY_ROW_STYLE,
                (
                    f'Choose the exact value for "{label}" from these recorded same-note values only: '
                    f"{known_value_options}. Do not invent a new number, time, identifier, name, or color."
                ),
                (
                    f"Exact answer: {fact.value}. {binding}."
                ),
            ),
        )
    return (
        (
            "canonical_label_value_statement",
            f"Learn this exact note fact as a label/value pair: {label}.",
            f"Exact answer: {fact.value}. {binding}.",
        ),
        (
            "exact_value_from_label",
            f"Answer with the exact value stored for {label}.",
            f"Exact answer: {fact.value}. {binding}.",
        ),
        (
            "label_value_flashcard",
            f"Create the back of a flashcard for the note label {label}.",
            f"Exact answer: {fact.value}. {binding}.",
        ),
        (
            "closed_book_label_value",
            f"Closed-book drill: recall the stored label/value pair for {label}.",
            f"Exact answer: {fact.value}. {binding}.",
        ),
        *contrast_templates,
        (
            "source_note_label_value",
            f"Use the source note to give the value for {label}.",
            f"Exact answer: {fact.value}. {binding}.",
        ),
        (
            "review_quiz_label_value",
            f"For a review quiz, give the recorded answer for {label}.",
            f"Exact answer: {fact.value}. {binding}.",
        ),
        (
            "notes_only_label_value",
            f"State the notes-only answer linked to {label}.",
            f"Exact answer: {fact.value}. {binding}.",
        ),
        (
            "short_label_value_recall",
            f"Turn the label {label} into a short exact recall answer.",
            f"Exact answer: {fact.value}. {binding}.",
        ),
    )


def _contrast_fact_for(fact: FactCard, facts_by_chunk: Mapping[str, Sequence[FactCard]]) -> FactCard | None:
    same_chunk_facts = facts_by_chunk.get(fact.source_chunk_id, ())
    for candidate in same_chunk_facts:
        if candidate.fact_id != fact.fact_id:
            return candidate
    return None


def _contrastable_fact_ids(facts: Sequence[FactCard]) -> set[str]:
    facts_by_chunk: dict[str, list[FactCard]] = {}
    for fact in facts:
        facts_by_chunk.setdefault(fact.source_chunk_id, []).append(fact)
    return {
        fact.fact_id
        for same_chunk_facts in facts_by_chunk.values()
        if len(same_chunk_facts) > 1
        for fact in same_chunk_facts
    }


def _known_values_only_options(fact: FactCard, *, same_chunk_facts: Sequence[FactCard]) -> str:
    candidates = [candidate for candidate in same_chunk_facts if candidate.source_chunk_id == fact.source_chunk_id]
    if not candidates:
        candidates = [fact]

    selected = candidates[:_MAX_KNOWN_VALUE_OPTIONS]
    if fact not in selected:
        selected = [*selected[: _MAX_KNOWN_VALUE_OPTIONS - 1], fact]
        selected.sort(key=lambda candidate: candidates.index(candidate))

    return "; ".join(_canonical_label_value_binding(candidate.label, candidate.value) for candidate in selected)


def _has_complete_disambiguation_signal(
    row: Mapping[str, object],
    *,
    facts_by_id: Mapping[str, FactCard],
) -> bool:
    row_style = str(row.get("row_style", ""))
    if row_style != "same_chunk_label_disambiguation":
        return False
    fact_id = str(row.get("fact_id", "")).strip()
    label = str(row.get("label", "")).strip()
    value = str(row.get("value", "")).strip()
    contrast_label = str(row.get("contrast_label", "")).strip()
    contrast_value = str(row.get("contrast_value", "")).strip()
    contrast_fact_id = str(row.get("contrast_fact_id", "")).strip()
    response = str(row.get("response", ""))
    if not all((fact_id, label, value, contrast_label, contrast_value, contrast_fact_id)):
        return False
    fact = facts_by_id.get(fact_id)
    contrast_fact = facts_by_id.get(contrast_fact_id)
    if fact is None or contrast_fact is None:
        return False
    if contrast_fact.fact_id == fact.fact_id:
        return False
    if fact.source_chunk_id != contrast_fact.source_chunk_id:
        return False
    if str(row.get("source_chunk_id", "")).strip() != fact.source_chunk_id:
        return False
    if label != fact.label or value != fact.value:
        return False
    if contrast_label != contrast_fact.label or contrast_value != contrast_fact.value:
        return False
    own_binding = _canonical_label_value_binding(label, value)
    return own_binding in response and f"{contrast_value} belongs to {contrast_label}" in response


def _has_complete_known_values_only_signal(
    row: Mapping[str, object],
    *,
    facts_by_id: Mapping[str, FactCard],
) -> bool:
    if str(row.get("row_style", "")) != _KNOWN_VALUES_ONLY_ROW_STYLE:
        return False
    fact_id = str(row.get("fact_id", "")).strip()
    label = str(row.get("label", "")).strip()
    value = str(row.get("value", "")).strip()
    contrast_label = str(row.get("contrast_label", "")).strip()
    contrast_value = str(row.get("contrast_value", "")).strip()
    contrast_fact_id = str(row.get("contrast_fact_id", "")).strip()
    instruction = str(row.get("instruction", ""))
    response = str(row.get("response", ""))
    if not all((fact_id, label, value, contrast_label, contrast_value, contrast_fact_id)):
        return False
    fact = facts_by_id.get(fact_id)
    contrast_fact = facts_by_id.get(contrast_fact_id)
    if fact is None or contrast_fact is None:
        return False
    if contrast_fact.fact_id == fact.fact_id:
        return False
    if fact.source_chunk_id != contrast_fact.source_chunk_id:
        return False
    if str(row.get("source_chunk_id", "")).strip() != fact.source_chunk_id:
        return False
    if label != fact.label or value != fact.value:
        return False
    if contrast_label != contrast_fact.label or contrast_value != contrast_fact.value:
        return False
    own_binding = _canonical_label_value_binding(label, value)
    contrast_binding = _canonical_label_value_binding(contrast_label, contrast_value)
    return (
        own_binding in instruction
        and contrast_binding in instruction
        and "Do not invent a new number, time, identifier, name, or color." in instruction
        and f"Exact answer: {value}." in response
        and own_binding in response
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


def _has_complete_manifest_metadata(row: Mapping[str, object] | None, *, split_name: str) -> bool:
    if row is None:
        return False
    if row.get("split") != split_name:
        return False
    for field in ("row_id", "fact_id", "source_chunk_id", "row_style", "label", "value"):
        if not str(row.get(field, "")).strip():
            return False
    return bool(tuple(str(term).strip() for term in row.get("expected_terms", ()) if str(term).strip()))


def _canonical_label_value_binding(label: str, value: str) -> str:
    clean_label = label.strip()
    clean_value = value.strip()
    if not clean_label or not clean_value:
        return ""
    return f"{clean_label}: {clean_value}"


def _validate_score_alignment(base_item: FactAnswerScore, trained_item: FactAnswerScore) -> None:
    if base_item.expected_terms != trained_item.expected_terms:
        raise ValueError("base and trained scores must use matching expected terms")
    if base_item.fact_id and trained_item.fact_id and base_item.fact_id != trained_item.fact_id:
        raise ValueError("base and trained scores must use matching fact_id values")
    if not base_item.fact_id and not trained_item.fact_id and base_item.question != trained_item.question:
        raise ValueError("base and trained scores must use matching questions when fact_id is absent")


def _fact_score_outcome(base_item: FactAnswerScore, trained_item: FactAnswerScore) -> str:
    if not base_item.scored or not trained_item.scored:
        return "unscored"
    if base_item.hit and trained_item.hit:
        return "unchanged"
    if base_item.hit and not trained_item.hit:
        return "worse"
    if not base_item.hit and trained_item.hit:
        return "learned"
    return "missed"


def _fact_score_label(item: FactScoreComparisonItem) -> str:
    parts = [part for part in (item.fact_id, item.label or item.question, item.source_chunk_id) if part]
    return " | ".join(parts) if parts else item.question


def _diagnose_fact_answer_miss(
    item: FactAnswerScore,
    *,
    facts: Sequence[FactCard],
) -> FactMissDiagnostic | None:
    if item.hit:
        return None
    if not item.scored:
        return None

    target_fact = _target_fact_for_score(item, facts)
    value_matches = _known_value_matches(item.answer, facts=facts, target_fact=target_fact)
    same_chunk_matches = tuple(
        match
        for match in value_matches
        if target_fact is not None and match.source_chunk_id == target_fact.source_chunk_id
    )
    label = item.label or (target_fact.label if target_fact else "")
    invented_values = tuple(
        candidate
        for candidate in _invented_value_candidates(item.answer, facts=facts)
        if not _candidate_echoes_label(candidate, label)
    )
    shape_markers = _answer_shape_markers(item.answer)
    label_echo = _answer_echoes_label(item.answer, label)

    if same_chunk_matches:
        miss_kind = "same_chunk_value_confusion"
        selected_value_matches = same_chunk_matches
    elif value_matches:
        miss_kind = "known_value_confusion"
        selected_value_matches = value_matches
    elif invented_values:
        miss_kind = "invented_numeric_time_identifier_value"
        selected_value_matches = ()
    elif label_echo:
        miss_kind = "label_echo"
        selected_value_matches = ()
    elif shape_markers:
        miss_kind = "answer_shape_without_fact"
        selected_value_matches = ()
    else:
        miss_kind = "generic_miss"
        selected_value_matches = ()

    return FactMissDiagnostic(
        miss_kind=miss_kind,
        question=item.question,
        answer=item.answer,
        expected_terms=item.expected_terms,
        missing_terms=item.missing_terms,
        fact_id=item.fact_id or (target_fact.fact_id if target_fact else ""),
        label=item.label or (target_fact.label if target_fact else ""),
        value=item.value or (target_fact.value if target_fact else ""),
        source_chunk_id=item.source_chunk_id or (target_fact.source_chunk_id if target_fact else ""),
        row_style=item.row_style,
        value_matches=selected_value_matches,
        invented_values=invented_values if miss_kind == "invented_numeric_time_identifier_value" else (),
        shape_markers=shape_markers,
    )


def _target_fact_for_score(item: FactAnswerScore, facts: Sequence[FactCard]) -> FactCard | None:
    if item.fact_id:
        for fact in facts:
            if fact.fact_id == item.fact_id:
                return fact
    expected_terms = set(item.expected_terms)
    label = _normalize_fact_text(item.label)
    value = item.value.strip()
    for fact in facts:
        if value and fact.value == value:
            return fact
        if label and _normalize_fact_text(fact.label) == label and any(term in fact.expected_terms for term in expected_terms):
            return fact
    return None


def _fact_miss_context_item(
    diagnostic: FactMissDiagnostic,
    *,
    split: FactTrainEvalSplit,
    facts: Sequence[FactCard],
) -> FactMissContextItem:
    target_fact = _target_fact_for_diagnostic(diagnostic, facts)
    expected_value = diagnostic.value or (target_fact.value if target_fact else "")
    training_rows = _fact_training_row_signals(
        diagnostic.fact_id or (target_fact.fact_id if target_fact else ""),
        expected_value=expected_value,
        manifest_rows=split.manifest_rows,
    )
    row_styles_seen = _unique_in_order(row.row_style for row in training_rows if row.row_style)
    same_chunk_known_values = _same_chunk_known_values(target_fact, facts)
    exact_value_in_prompt_rows = sum(1 for row in training_rows if row.exact_value_in_instruction)
    exact_value_in_completion_rows = sum(1 for row in training_rows if row.exact_value_in_response)
    known_values_only_warning_present = any(row.known_values_only_warning_present for row in training_rows)
    invented_values = tuple(
        FactInventedValueCandidate(text=value, value_shapes=_value_shapes(value))
        for value in diagnostic.invented_values
    )
    return FactMissContextItem(
        fact_id=diagnostic.fact_id or (target_fact.fact_id if target_fact else ""),
        label=diagnostic.label or (target_fact.label if target_fact else ""),
        source_chunk_id=diagnostic.source_chunk_id or (target_fact.source_chunk_id if target_fact else ""),
        question=diagnostic.question,
        trained_answer=diagnostic.answer,
        expected_value=expected_value,
        expected_value_shapes=_value_shapes(expected_value),
        missing_terms=diagnostic.missing_terms,
        miss_kind=diagnostic.miss_kind,
        invented_values=invented_values,
        row_styles_seen=row_styles_seen,
        train_row_count=len(training_rows),
        exact_value_in_prompt_rows=exact_value_in_prompt_rows,
        exact_value_in_completion_rows=exact_value_in_completion_rows,
        known_values_only_warning_present=known_values_only_warning_present,
        same_chunk_known_values=same_chunk_known_values,
        training_rows=training_rows,
        diagnosis=_fact_miss_context_diagnosis(
            diagnostic,
            expected_value_shapes=_value_shapes(expected_value),
            invented_values=invented_values,
            training_rows=training_rows,
            exact_value_in_completion_rows=exact_value_in_completion_rows,
        ),
    )


def _target_fact_for_diagnostic(
    diagnostic: FactMissDiagnostic,
    facts: Sequence[FactCard],
) -> FactCard | None:
    if diagnostic.fact_id:
        for fact in facts:
            if fact.fact_id == diagnostic.fact_id:
                return fact
    label = _normalize_fact_text(diagnostic.label)
    for fact in facts:
        if label and _normalize_fact_text(fact.label) == label:
            return fact
    return None


def _fact_training_row_signals(
    fact_id: str,
    *,
    expected_value: str,
    manifest_rows: Sequence[Mapping[str, object]],
) -> tuple[FactMissTrainingRowSignal, ...]:
    rows: list[FactMissTrainingRowSignal] = []
    for row in manifest_rows:
        if row.get("split") != "train" or str(row.get("fact_id", "")) != fact_id:
            continue
        instruction = str(row.get("instruction", ""))
        response = str(row.get("response", ""))
        rows.append(
            FactMissTrainingRowSignal(
                row_id=str(row.get("row_id", "")),
                row_style=str(row.get("row_style", "")),
                instruction=instruction,
                response=response,
                public_row_fields=("instruction", "response", "source_chunk_id"),
                exact_value_in_instruction=_contains_expected_value(instruction, expected_value),
                exact_value_in_response=_contains_expected_value(response, expected_value),
                known_values_only_warning_present=_ANTI_INVENTION_WARNING_TEXT in instruction,
            )
        )
    return tuple(rows)


def _same_chunk_known_values(target_fact: FactCard | None, facts: Sequence[FactCard]) -> tuple[str, ...]:
    if target_fact is None:
        return ()
    return tuple(
        fact.value
        for fact in facts
        if fact.source_chunk_id == target_fact.source_chunk_id
    )


def _fact_miss_context_diagnosis(
    diagnostic: FactMissDiagnostic,
    *,
    expected_value_shapes: tuple[str, ...],
    invented_values: tuple[FactInventedValueCandidate, ...],
    training_rows: tuple[FactMissTrainingRowSignal, ...],
    exact_value_in_completion_rows: int,
) -> str:
    if not training_rows:
        return "The miss could not be linked to fact-ledger training rows; inspect stale or missing metadata."
    if diagnostic.miss_kind in {"same_chunk_value_confusion", "known_value_confusion"}:
        return "The model selected a real note value for the wrong label; strengthen label/value separation."
    if diagnostic.miss_kind == "label_echo":
        return "The model repeated the label but did not recall the exact note value."
    if diagnostic.miss_kind == "invented_numeric_time_identifier_value":
        invented_shapes = {shape for candidate in invented_values for shape in candidate.value_shapes}
        if expected_value_shapes and invented_shapes.intersection(expected_value_shapes):
            return "The model learned answer shape/value type, but not the exact note value."
        return "The model invented a plausible-looking value instead of using the recorded note value."
    if exact_value_in_completion_rows == len(training_rows):
        return "The exact value is present in every completion row, so the miss is not a missing-value row issue."
    return "The model missed the exact note value; inspect the prompt/completion rows for weak value emphasis."


def _contains_expected_value(text: str, expected_value: str) -> bool:
    if not expected_value:
        return False
    return _answer_contains_expected_term(text, _text_tokens(text), expected_value)


def _unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _known_value_matches(
    answer: str,
    *,
    facts: Sequence[FactCard],
    target_fact: FactCard | None,
) -> tuple[FactValueMatch, ...]:
    answer_tokens = _text_tokens(answer)
    matches: list[FactValueMatch] = []
    seen_values: set[str] = set()
    for fact in facts:
        if target_fact is not None and fact.fact_id == target_fact.fact_id:
            continue
        normalized_value = _normalize_fact_text(fact.value)
        if normalized_value in seen_values:
            continue
        if not _answer_contains_expected_term(answer, answer_tokens, fact.value):
            continue
        seen_values.add(normalized_value)
        matches.append(
            FactValueMatch(
                fact_id=fact.fact_id,
                label=fact.label,
                value=fact.value,
                source_chunk_id=fact.source_chunk_id,
            )
        )
    return tuple(matches)


def _invented_value_candidates(answer: str, *, facts: Sequence[FactCard]) -> tuple[str, ...]:
    candidates: list[str] = []
    for candidate in (*_exact_answer_candidates(answer), *_numeric_time_identifier_candidates(answer)):
        clean_candidate = _clean_invented_candidate(candidate)
        if not clean_candidate or clean_candidate in candidates:
            continue
        if _candidate_is_known_fact_value(clean_candidate, facts):
            continue
        if _candidate_looks_like_plausible_value(clean_candidate):
            candidates.append(clean_candidate)
    return tuple(candidates)


def _value_shapes(value: str) -> tuple[str, ...]:
    clean_value = _clean_invented_candidate(value)
    if not clean_value:
        return ()

    shapes: list[str] = []
    if _looks_like_time(clean_value):
        shapes.append("time")
    if _looks_like_identifier(clean_value):
        shapes.append("identifier")
    # Identifier-shaped values are more useful than also tagging their embedded digits as numbers.
    elif _looks_like_number(clean_value):
        shapes.append("number")
    if _looks_like_color(clean_value):
        shapes.append("color")
    if _looks_like_name(clean_value):
        shapes.append("name")
    if not shapes:
        shapes.append("phrase")
    return tuple(shapes)


def _exact_answer_candidates(answer: str) -> tuple[str, ...]:
    candidates = []
    for match in re.finditer(r"exact answer:\s*([^.\n]+)", answer, flags=re.IGNORECASE):
        candidates.append(match.group(1))
    return tuple(candidates)


def _numeric_time_identifier_candidates(answer: str) -> tuple[str, ...]:
    patterns = (
        r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b",
        r"\b[A-Za-z]+(?:-[A-Za-z0-9]+)+\b",
        r"\b[A-Za-z]*\d+[A-Za-z0-9-]*\b",
        r"\bversion\s+\d+\b",
    )
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(match.group(0) for match in re.finditer(pattern, answer))
    return tuple(candidates)


def _clean_invented_candidate(candidate: str) -> str:
    return candidate.strip().strip("\"'`*_ ").rstrip(".,;:!?")


def _candidate_is_known_fact_value(candidate: str, facts: Sequence[FactCard]) -> bool:
    candidate_tokens = _text_tokens(candidate)
    return any(_answer_contains_expected_term(candidate, candidate_tokens, fact.value) for fact in facts)


def _candidate_echoes_label(candidate: str, label: str) -> bool:
    normalized_candidate = _normalize_fact_text(candidate)
    normalized_label = _normalize_fact_text(label)
    return bool(normalized_candidate and normalized_candidate == normalized_label)


def _candidate_looks_like_plausible_value(candidate: str) -> bool:
    if len(candidate) < 2:
        return False
    return any(shape in {"time", "identifier", "number", "name", "color"} for shape in _value_shapes(candidate))


def _looks_like_time(value: str) -> bool:
    return bool(re.search(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b", value))


def _looks_like_identifier(value: str) -> bool:
    if re.fullmatch(r"version\s+\d+", value, flags=re.IGNORECASE):
        return True
    return bool(
        "-" in value
        and re.search(r"[A-Za-z0-9]", value)
    )


def _looks_like_number(value: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:[.,]\d+)?", value))


def _looks_like_name(value: str) -> bool:
    words = value.split()
    return (
        1 < len(words) <= 4
        and all(re.fullmatch(r"[A-Z][A-Za-z0-9-]*", word) for word in words)
    )


def _looks_like_color(value: str) -> bool:
    return _normalize_fact_text(value) in {
        "black",
        "blue",
        "brown",
        "cyan",
        "gold",
        "green",
        "grey",
        "gray",
        "indigo",
        "magenta",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "ultramarine",
        "violet",
        "white",
        "yellow",
    }


def _answer_shape_markers(answer: str) -> tuple[str, ...]:
    markers: list[str] = []
    checks = (
        ("Exact answer:", r"\bexact answer\s*:"),
        ("Label: value", r"\b[A-Za-z][A-Za-z ]{1,40}\s*:\s*[^.\n]+"),
        ("belongs to", r"\bbelongs to\b"),
        ("not <label>", r"\bnot\b"),
    )
    for label, pattern in checks:
        if re.search(pattern, answer, flags=re.IGNORECASE):
            markers.append(label)
    return tuple(markers)


def _answer_echoes_label(answer: str, label: str) -> bool:
    normalized_label = _normalize_fact_text(label)
    if not normalized_label:
        return False
    return _contains_token_phrase(_text_tokens(answer), _text_tokens(normalized_label))


def _fact_miss_label(item: FactMissDiagnostic) -> str:
    parts = [part for part in (item.fact_id, item.label or item.question, item.source_chunk_id) if part]
    return " | ".join(parts) if parts else item.question


def _fact_miss_context_label(item: FactMissContextItem) -> str:
    parts = [part for part in (item.fact_id, item.label or item.question, item.source_chunk_id) if part]
    return " | ".join(parts) if parts else item.question


def _format_shape_suffix(shapes: tuple[str, ...]) -> str:
    return " (shape: " + ", ".join(shapes) + ")" if shapes else ""


__all__ = [
    "DEFAULT_FACT_TRAIN_EXAMPLES_PER_FACT",
    "DEFAULT_NEXT_SMOKE_MAX_INVENTED_VALUE_MISSES",
    "DEFAULT_NEXT_SMOKE_REQUIRED_EXACT_HITS",
    "FactAnswerScore",
    "FactCard",
    "FactInventedValueCandidate",
    "FactMissContextItem",
    "FactMissContextReport",
    "FactMissTrainingRowSignal",
    "FactMissDiagnostic",
    "FactMissDiagnosticReport",
    "FactOutputScore",
    "FactQualityGateReport",
    "FactQualityIssue",
    "FactReadinessReport",
    "FactScoreComparison",
    "FactScoreComparisonItem",
    "FactTrainEvalSplit",
    "FactValueMatch",
    "analyze_fact_miss_contexts",
    "analyze_fact_quality_gate",
    "analyze_fact_readiness",
    "build_fact_train_eval_split",
    "compare_fact_scores",
    "diagnose_fact_misses",
    "extract_fact_ledger",
    "format_fact_miss_context_report",
    "format_fact_miss_diagnostic_report",
    "format_fact_quality_report",
    "format_fact_readiness_report",
    "format_fact_score_report",
    "score_fact_answer",
    "score_fact_outputs",
]
