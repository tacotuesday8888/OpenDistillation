"""Internal experiment manifests for bounded notes-model quality smokes.

This module does not start training. It prepares the exact local data contract
that a later Colab GPU run should satisfy before spending GPU time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import subprocess
from typing import Any

from .comparison import build_comparison_request
from .engines import TrainingResult
from .fact_ledger import (
    analyze_fact_readiness,
    build_fact_comparison_rows,
    build_fact_train_eval_split,
    extract_fact_ledger,
)
from .text import chunk_text, load_text_document
from .training import (
    SFTLoRAConfig,
    SFTLoRATrainingEngine,
    build_lora_config_kwargs,
    build_sft_config_kwargs,
    build_sft_preview_rows,
    build_training_request,
)


ANTI_INVENTION_SMOKE_SCHEMA_VERSION = "anti-invention-smoke-manifest/v1"
ANTI_INVENTION_SMOKE_NAME = "anti-invention-known-values-t4-smoke"
ANTI_INVENTION_CHUNK_MAX_CHARS = 300
ANTI_INVENTION_PREVIEW_ROWS = 6
ANTI_INVENTION_MAX_STEPS = 30
ANTI_INVENTION_MAX_COMPARISON_EXAMPLES = 8
ANTI_INVENTION_PREVIOUS_BEST_EXACT_HITS = 1
ANTI_INVENTION_REQUIRED_EXACT_HITS = 2


@dataclass(frozen=True)
class AntiInventionSmokeContract:
    """Expected local shape for the next sample-notes T4 smoke."""

    fact_count: int = 8
    train_row_count: int = 48
    eval_row_count: int = 8
    contrastable_fact_count: int = 8
    disambiguation_train_row_count: int = 8
    known_values_only_train_row_count: int = 8
    sft_preview_row_count: int = 6
    comparison_question_count: int = 8
    previous_best_exact_hits: int = ANTI_INVENTION_PREVIOUS_BEST_EXACT_HITS
    required_trained_exact_hits: int = ANTI_INVENTION_REQUIRED_EXACT_HITS


@dataclass(frozen=True)
class SmokeManifestValidationReport:
    """Result of checking whether a smoke manifest is ready to run."""

    ready: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def build_anti_invention_smoke_manifest(
    note_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    training_output_dir: str | Path = "outputs/notes-lora",
    contract: AntiInventionSmokeContract = AntiInventionSmokeContract(),
    git_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a deterministic preflight manifest for the next T4 smoke.

    The default contract intentionally targets the committed ``sample-notes.md``
    run that exercises the anti-invention known-values rows. Custom notes can be
    inspected with the same builder, but they will not satisfy this exact sample
    smoke contract unless their fact-ledger shape matches it.
    """

    repo_path = Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    note_file = Path(note_path)
    if not note_file.is_absolute():
        note_file = repo_path / note_file
    note_file = note_file.resolve()
    note_bytes = note_file.read_bytes()
    document = load_text_document(note_file.name, note_bytes)
    chunks = chunk_text(document.text, max_chars=ANTI_INVENTION_CHUNK_MAX_CHARS)
    facts = extract_fact_ledger(chunks)
    split = build_fact_train_eval_split(facts)

    preview_limit = min(ANTI_INVENTION_PREVIEW_ROWS, len(split.train_rows))
    preview_rows = (
        build_sft_preview_rows(
            split.train_rows,
            manifest_rows=split.manifest_rows,
            max_rows=preview_limit,
        )
        if preview_limit
        else ()
    )
    readiness = analyze_fact_readiness(split, sft_preview_row_count=len(preview_rows))
    training_config = SFTLoRAConfig(max_steps=ANTI_INVENTION_MAX_STEPS)
    training_request = build_training_request(
        split.train_rows,
        Path(training_output_dir),
        config=training_config,
    )
    training_engine = SFTLoRATrainingEngine(training_config)
    training_plan = training_engine.describe(training_request)
    fake_training_result = TrainingResult(
        engine_name=str(training_plan["engine"]),
        output_path=Path(str(training_plan["artifact"])),
        created_model_artifact=True,
    )
    comparison_rows = build_fact_comparison_rows(split)
    comparison_request = build_comparison_request(
        comparison_rows,
        fake_training_result,
        config=training_config,
        max_examples=ANTI_INVENTION_MAX_COMPARISON_EXAMPLES,
    )
    anti_invention_signal = _anti_invention_signal(split.manifest_rows)
    sft_preview_payload = [
        {
            "row_index": row.row_index,
            "row_id": row.row_id,
            "fact_id": row.fact_id,
            "label": row.label,
            "row_style": row.row_style,
            "source_chunk_id": row.source_chunk_id,
            "expected_terms": list(row.expected_terms),
            "prompt": row.prompt,
            "completion": row.completion,
        }
        for row in preview_rows
    ]
    comparison_question_payload = [
        {
            "row_id": example.row_id,
            "fact_id": example.fact_id,
            "label": example.label,
            "expected_terms": list(example.expected_terms),
            "source_chunk_id": example.source_chunk_id,
            "question": example.question,
            "reference_response": example.reference_response,
        }
        for example in comparison_request.examples
    ]

    manifest: dict[str, object] = {
        "schema_version": ANTI_INVENTION_SMOKE_SCHEMA_VERSION,
        "experiment_name": ANTI_INVENTION_SMOKE_NAME,
        "purpose": (
            "Verify whether anti-invention known-values rows help the small notes "
            "model beat the previous best held-out exact fact score."
        ),
        "repo": dict(git_state) if git_state is not None else _collect_git_state(repo_path),
        "source": {
            "path": str(note_file),
            "filename": document.filename,
            "extension": document.extension,
            "sha256": hashlib.sha256(note_bytes).hexdigest(),
            "char_count": document.char_count,
            "word_count": document.word_count,
            "chunk_max_chars": ANTI_INVENTION_CHUNK_MAX_CHARS,
            "chunk_count": len(chunks),
        },
        "contract": _contract_dict(contract),
        "data": {
            "fact_count": len(split.facts),
            "train_row_count": len(split.train_rows),
            "eval_row_count": len(split.eval_rows),
            "manifest_row_count": len(split.manifest_rows),
            "train_row_styles": _count_by(split.manifest_rows, split_name="train", field="row_style"),
            "eval_row_styles": _count_by(split.manifest_rows, split_name="eval", field="row_style"),
            "public_schema": {
                "train_fields": ["instruction", "response", "source_chunk_id"],
                "eval_fields": ["instruction", "response", "source_chunk_id"],
                "train_schema_valid": all(set(row) == {"instruction", "response", "source_chunk_id"} for row in split.train_rows),
                "eval_schema_valid": all(set(row) == {"instruction", "response", "source_chunk_id"} for row in split.eval_rows),
            },
            "hashes": {
                "train_rows_sha256": _stable_json_hash(split.train_rows),
                "eval_rows_sha256": _stable_json_hash(split.eval_rows),
                "manifest_rows_sha256": _stable_json_hash(split.manifest_rows),
            },
            "facts": [
                {
                    "fact_id": fact.fact_id,
                    "source_chunk_id": fact.source_chunk_id,
                    "label": fact.label,
                    "value": fact.value,
                    "expected_terms": list(fact.expected_terms),
                    "source_hash": fact.source_hash,
                }
                for fact in split.facts
            ],
        },
        "quality_gate": _quality_report_dict(readiness.quality_report),
        "readiness": _readiness_report_dict(readiness),
        "anti_invention_signal": anti_invention_signal,
        "sft_preview": sft_preview_payload,
        "sft_preview_sha256": _stable_json_hash(sft_preview_payload),
        "training": {
            "row_source": "fact-ledger train rows",
            "engine": training_plan["engine"],
            "student_model": training_request.student_model,
            "max_steps": training_request.max_steps,
            "output_dir": str(training_request.output_dir),
            "adapter_path": str(training_plan["artifact"]),
            "requires_gpu": training_plan["requires_gpu"],
            "dependencies": list(training_plan["dependencies"]),
            "sft_config": build_sft_config_kwargs(training_request, training_config),
            "lora_config": build_lora_config_kwargs(training_config),
        },
        "comparison": {
            "engine": "transformers-peft-before-after",
            "max_examples": ANTI_INVENTION_MAX_COMPARISON_EXAMPLES,
            "question_count": len(comparison_request.examples),
            "max_new_tokens": comparison_request.max_new_tokens,
            "adapter_path": str(comparison_request.adapter_path),
            "questions_sha256": _stable_json_hash(comparison_question_payload),
            "questions": comparison_question_payload,
        },
        "quality_rule": {
            "previous_best_trained_exact_hits": contract.previous_best_exact_hits,
            "required_trained_exact_hits": contract.required_trained_exact_hits,
            "required_total_questions": contract.comparison_question_count,
            "failure_rule": (
                "Changed answers are a failure when trained exact fact hits do not "
                "beat the previous best held-out score."
            ),
        },
    }
    validation = validate_anti_invention_smoke_manifest(manifest, contract=contract)
    manifest["validation"] = {
        "ready": validation.ready,
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
    }
    manifest["manifest_sha256"] = _stable_json_hash({**manifest, "manifest_sha256": ""})
    return manifest


def validate_anti_invention_smoke_manifest(
    manifest: Mapping[str, object],
    *,
    contract: AntiInventionSmokeContract = AntiInventionSmokeContract(),
) -> SmokeManifestValidationReport:
    """Validate the manifest as the exact next anti-invention GPU contract."""

    errors: list[str] = []
    warnings: list[str] = []
    if manifest.get("schema_version") != ANTI_INVENTION_SMOKE_SCHEMA_VERSION:
        errors.append("manifest schema_version is not anti-invention-smoke-manifest/v1")
    if manifest.get("experiment_name") != ANTI_INVENTION_SMOKE_NAME:
        errors.append("manifest experiment_name does not match the anti-invention smoke")

    source = _mapping(manifest.get("source"))
    if source.get("extension") not in {".txt", ".md"}:
        errors.append("source note must be a .txt or .md file")

    data = _mapping(manifest.get("data"))
    _expect_equal(errors, "fact_count", data.get("fact_count"), contract.fact_count)
    _expect_equal(errors, "train_row_count", data.get("train_row_count"), contract.train_row_count)
    _expect_equal(errors, "eval_row_count", data.get("eval_row_count"), contract.eval_row_count)
    public_schema = _mapping(data.get("public_schema"))
    if public_schema.get("train_schema_valid") is not True:
        errors.append("public training rows do not match the v0 JSONL schema")
    if public_schema.get("eval_schema_valid") is not True:
        errors.append("public eval rows do not match the v0 JSONL schema")

    readiness = _mapping(manifest.get("readiness"))
    if readiness.get("ready_for_gpu_smoke") is not True:
        errors.append("readiness report is not ready for GPU smoke")
    _expect_equal(errors, "contrastable_fact_count", readiness.get("contrastable_fact_count"), contract.contrastable_fact_count)
    _expect_equal(
        errors,
        "disambiguation_train_row_count",
        readiness.get("disambiguation_train_row_count"),
        contract.disambiguation_train_row_count,
    )
    _expect_equal(
        errors,
        "known_values_only_train_row_count",
        readiness.get("known_values_only_train_row_count"),
        contract.known_values_only_train_row_count,
    )

    quality_gate = _mapping(manifest.get("quality_gate"))
    if quality_gate.get("passes_required_checks") is not True:
        errors.append("fact-ledger quality gate did not pass")
    _expect_equal(errors, "exact_leak_count", quality_gate.get("exact_leak_count"), 0)
    _expect_equal(errors, "near_leak_count", quality_gate.get("near_leak_count"), 0)
    _expect_equal(errors, "missing_expected_term_count", quality_gate.get("missing_expected_term_count"), 0)
    _expect_equal(errors, "missing_manifest_metadata_count", quality_gate.get("missing_manifest_metadata_count"), 0)

    anti_invention_signal = _mapping(manifest.get("anti_invention_signal"))
    _expect_equal(
        errors,
        "anti_invention known_values_only_row_count",
        anti_invention_signal.get("known_values_only_row_count"),
        contract.known_values_only_train_row_count,
    )
    _expect_equal(
        errors,
        "anti_invention warning_text_row_count",
        anti_invention_signal.get("warning_text_row_count"),
        contract.known_values_only_train_row_count,
    )
    if anti_invention_signal.get("all_known_values_only_rows_have_warning") is not True:
        errors.append("known-values-only rows do not all include the anti-invention warning")
    if anti_invention_signal.get("all_known_values_only_rows_use_same_chunk_contrast") is not True:
        errors.append("known-values-only rows do not all use same-chunk contrast values")

    preview = manifest.get("sft_preview")
    preview_count = len(preview) if isinstance(preview, list) else 0
    _expect_equal(errors, "sft_preview_row_count", preview_count, contract.sft_preview_row_count)
    if isinstance(preview, list):
        for index, row in enumerate(preview, start=1):
            row_map = _mapping(row)
            if not row_map.get("prompt") or not row_map.get("completion"):
                errors.append(f"sft_preview row {index} is missing prompt or completion text")
            if not row_map.get("expected_terms"):
                errors.append(f"sft_preview row {index} is missing expected terms")

    comparison = _mapping(manifest.get("comparison"))
    _expect_equal(errors, "comparison question_count", comparison.get("question_count"), contract.comparison_question_count)
    questions = comparison.get("questions")
    if isinstance(questions, list):
        for index, question in enumerate(questions, start=1):
            question_map = _mapping(question)
            if not question_map.get("fact_id") or not question_map.get("expected_terms"):
                errors.append(f"comparison question {index} is missing fact identity or expected terms")
    else:
        errors.append("comparison questions are missing")

    quality_rule = _mapping(manifest.get("quality_rule"))
    if quality_rule.get("required_trained_exact_hits") != contract.required_trained_exact_hits:
        errors.append("quality rule does not require beating the previous best exact-hit score")
    if contract.required_trained_exact_hits <= contract.previous_best_exact_hits:
        errors.append("quality contract must require a trained score above the previous best")

    repo = _mapping(manifest.get("repo"))
    if repo.get("dirty") is True:
        warnings.append("repo is dirty; commit the runner before using the manifest as GPU evidence")
    if not repo.get("commit"):
        warnings.append("git commit could not be recorded")

    return SmokeManifestValidationReport(
        ready=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def format_anti_invention_smoke_report(manifest: Mapping[str, object]) -> list[str]:
    """Format a plain-language preflight report for local and Colab logs."""

    validation = _mapping(manifest.get("validation"))
    source = _mapping(manifest.get("source"))
    data = _mapping(manifest.get("data"))
    readiness = _mapping(manifest.get("readiness"))
    quality_gate = _mapping(manifest.get("quality_gate"))
    training = _mapping(manifest.get("training"))
    comparison = _mapping(manifest.get("comparison"))
    quality_rule = _mapping(manifest.get("quality_rule"))

    lines = [
        "Anti-invention T4 smoke preflight",
        f"Ready for GPU smoke: {'yes' if validation.get('ready') else 'no'}",
        f"Source: {source.get('filename', '')} ({source.get('extension', '')}), {source.get('chunk_count', 0)} chunks",
        f"Facts: {data.get('fact_count', 0)}",
        f"Train rows: {data.get('train_row_count', 0)}",
        f"Held-out eval rows: {data.get('eval_row_count', 0)}",
        (
            "Disambiguation rows: "
            f"{readiness.get('disambiguation_train_row_count', 0)}; "
            "known-values anti-invention rows: "
            f"{readiness.get('known_values_only_train_row_count', 0)}"
        ),
        (
            "Leakage: "
            f"{quality_gate.get('exact_leak_count', 0)} exact, "
            f"{quality_gate.get('near_leak_count', 0)} near-duplicate"
        ),
        f"SFT preview rows: {len(manifest.get('sft_preview', ())) if isinstance(manifest.get('sft_preview'), list) else 0}",
        (
            "Training plan: "
            f"{training.get('student_model', '')}, {training.get('max_steps', 0)} steps, "
            f"adapter {training.get('adapter_path', '')}"
        ),
        f"Held-out comparison questions: {comparison.get('question_count', 0)}",
        (
            "Pass condition: trained exact fact hits must be at least "
            f"{quality_rule.get('required_trained_exact_hits', 0)}/"
            f"{quality_rule.get('required_total_questions', 0)} "
            f"and beat the previous best {quality_rule.get('previous_best_trained_exact_hits', 0)}/"
            f"{quality_rule.get('required_total_questions', 0)}."
        ),
        "Changed answers with wrong facts are failure evidence, not progress.",
    ]
    errors = tuple(str(error) for error in validation.get("errors", ()) if str(error))
    warnings = tuple(str(warning) for warning in validation.get("warnings", ()) if str(warning))
    if errors:
        lines.append("Validation errors:")
        lines.extend(f"- {error}" for error in errors)
    if warnings:
        lines.append("Validation warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    return lines


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _contract_dict(contract: AntiInventionSmokeContract) -> dict[str, int]:
    return {
        "fact_count": contract.fact_count,
        "train_row_count": contract.train_row_count,
        "eval_row_count": contract.eval_row_count,
        "contrastable_fact_count": contract.contrastable_fact_count,
        "disambiguation_train_row_count": contract.disambiguation_train_row_count,
        "known_values_only_train_row_count": contract.known_values_only_train_row_count,
        "sft_preview_row_count": contract.sft_preview_row_count,
        "comparison_question_count": contract.comparison_question_count,
        "previous_best_exact_hits": contract.previous_best_exact_hits,
        "required_trained_exact_hits": contract.required_trained_exact_hits,
    }


def _quality_report_dict(report: Any) -> dict[str, object]:
    return {
        "passes_required_checks": report.passes_required_checks,
        "fact_count": report.fact_count,
        "train_row_count": report.train_row_count,
        "eval_row_count": report.eval_row_count,
        "train_fact_coverage": report.train_fact_coverage,
        "eval_fact_coverage": report.eval_fact_coverage,
        "exact_leak_count": report.exact_leak_count,
        "near_leak_count": report.near_leak_count,
        "missing_expected_term_count": report.missing_expected_term_count,
        "unknown_source_chunk_count": report.unknown_source_chunk_count,
        "missing_manifest_metadata_count": report.missing_manifest_metadata_count,
        "issues": [
            {
                "code": issue.code,
                "severity": issue.severity,
                "message": issue.message,
                "row_ids": list(issue.row_ids),
            }
            for issue in report.issues
        ],
    }


def _readiness_report_dict(report: Any) -> dict[str, object]:
    return {
        "ready_for_gpu_smoke": report.ready_for_gpu_smoke,
        "skip_reason": report.skip_reason,
        "fact_count": report.fact_count,
        "train_row_count": report.train_row_count,
        "eval_row_count": report.eval_row_count,
        "train_examples_per_fact": report.train_examples_per_fact,
        "label_value_fact_coverage": report.label_value_fact_coverage,
        "label_value_train_row_count": report.label_value_train_row_count,
        "contrastable_fact_count": report.contrastable_fact_count,
        "disambiguation_fact_coverage": report.disambiguation_fact_coverage,
        "disambiguation_train_row_count": report.disambiguation_train_row_count,
        "known_values_only_fact_coverage": report.known_values_only_fact_coverage,
        "known_values_only_train_row_count": report.known_values_only_train_row_count,
        "sft_preview_row_count": report.sft_preview_row_count,
    }


def _anti_invention_signal(rows: tuple[Mapping[str, object], ...]) -> dict[str, object]:
    rows_by_fact_id = {
        str(row.get("fact_id", "")): row
        for row in rows
        if row.get("split") == "train" and str(row.get("fact_id", ""))
    }
    signal_rows = [
        row
        for row in rows
        if row.get("split") == "train" and row.get("row_style") == "known_values_only_label_value"
    ]
    warning_text = "Do not invent a new number, time, identifier, name, or color."
    warning_count = sum(1 for row in signal_rows if warning_text in str(row.get("instruction", "")))
    same_chunk_count = 0
    compact_rows: list[dict[str, str]] = []
    for row in signal_rows:
        contrast_row = rows_by_fact_id.get(str(row.get("contrast_fact_id", "")), {})
        if str(contrast_row.get("source_chunk_id", "")) == str(row.get("source_chunk_id", "")):
            same_chunk_count += 1
        compact_rows.append(
            {
                "row_id": str(row.get("row_id", "")),
                "fact_id": str(row.get("fact_id", "")),
                "label": str(row.get("label", "")),
                "value": str(row.get("value", "")),
                "source_chunk_id": str(row.get("source_chunk_id", "")),
                "contrast_fact_id": str(row.get("contrast_fact_id", "")),
                "contrast_label": str(row.get("contrast_label", "")),
                "contrast_value": str(row.get("contrast_value", "")),
            }
        )
    return {
        "known_values_only_row_count": len(signal_rows),
        "warning_text": warning_text,
        "warning_text_row_count": warning_count,
        "same_chunk_contrast_row_count": same_chunk_count,
        "all_known_values_only_rows_have_warning": warning_count == len(signal_rows),
        "all_known_values_only_rows_use_same_chunk_contrast": same_chunk_count == len(signal_rows),
        "rows": compact_rows,
    }


def _count_by(
    rows: tuple[Mapping[str, object], ...],
    *,
    split_name: str,
    field: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("split") != split_name:
            continue
        value = str(row.get(field, ""))
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _collect_git_state(repo_root: Path) -> dict[str, object]:
    return {
        "root": str(repo_root),
        "commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "branch": _git_output(repo_root, "branch", "--show-current"),
        "dirty": bool(_git_output(repo_root, "status", "--porcelain")),
    }


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _expect_equal(errors: list[str], label: str, actual: object, expected: object) -> None:
    if actual != expected:
        errors.append(f"{label} expected {expected}, got {actual}")


__all__ = [
    "ANTI_INVENTION_CHUNK_MAX_CHARS",
    "ANTI_INVENTION_MAX_COMPARISON_EXAMPLES",
    "ANTI_INVENTION_MAX_STEPS",
    "ANTI_INVENTION_PREVIEW_ROWS",
    "ANTI_INVENTION_REQUIRED_EXACT_HITS",
    "ANTI_INVENTION_SMOKE_NAME",
    "ANTI_INVENTION_SMOKE_SCHEMA_VERSION",
    "AntiInventionSmokeContract",
    "SmokeManifestValidationReport",
    "build_anti_invention_smoke_manifest",
    "format_anti_invention_smoke_report",
    "validate_anti_invention_smoke_manifest",
]
