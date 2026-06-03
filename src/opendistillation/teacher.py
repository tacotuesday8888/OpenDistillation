"""Teacher engines for v0 notes dataset generation."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Callable
from dataclasses import dataclass
import json
import re
from typing import Any, Protocol

from .dataset import DatasetValidationError, validate_dataset
from .runtime import build_pip_install_command
from .text import TextChunk


DEFAULT_REAL_TEACHER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
QUESTION_STYLE_GUIDE = (
    "factual recall",
    "explanation",
    "flashcard",
    "misconception-check",
)


@dataclass(frozen=True)
class TeacherRequest:
    """Input for a teacher engine.

    Mock and real teacher engines use the same request shape so the notebook
    flow does not change when a user opts into real generation.
    """

    chunks: list[TextChunk]
    examples_per_chunk: int = 4


@dataclass(frozen=True)
class RealTeacherConfig:
    """Configuration for the optional local Hugging Face teacher path."""

    model_name: str = DEFAULT_REAL_TEACHER_MODEL
    max_new_tokens: int = 512


class RealTeacherError(RuntimeError):
    """Base error for the optional real teacher path."""


class RealTeacherDependencyError(RealTeacherError):
    """Raised when optional Hugging Face teacher dependencies are unavailable."""


class RealTeacherModelLoadError(RealTeacherError):
    """Raised when the teacher model cannot be downloaded or loaded."""


class RealTeacherGenerationError(RealTeacherError):
    """Raised when the teacher model fails during generation."""


class RealTeacherOutputError(RealTeacherError):
    """Raised when generated teacher rows fail the v0 JSONL schema."""


class TeacherEngine(Protocol):
    """Interface for mock and real teacher-generation engines."""

    name: str
    sends_data_remote: bool

    def generate(self, request: TeacherRequest) -> list[dict[str, str]]:
        """Generate validated dataset rows from text chunks."""


class MockTeacherEngine:
    """Safe deterministic teacher used by the notebook default path."""

    name = "mock-local-teacher"
    sends_data_remote = False

    def generate(self, request: TeacherRequest) -> list[dict[str, str]]:
        return _generate_rows(request.chunks, examples_per_chunk=request.examples_per_chunk)


class HuggingFaceLocalTeacherEngine:
    """Optional local open-source teacher backed by a Transformers chat model."""

    name = "huggingface-local-teacher"
    sends_data_remote = False

    def __init__(
        self,
        config: RealTeacherConfig | None = None,
        *,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or RealTeacherConfig()
        self.model_name = self.config.model_name
        self._pipeline_factory = pipeline_factory or _build_text_generation_pipeline

    def generate(self, request: TeacherRequest) -> list[dict[str, str]]:
        if request.examples_per_chunk < 1:
            raise ValueError("examples_per_chunk must be at least 1")

        pipeline = self._load_pipeline()
        rows: list[dict[str, str]] = []
        for chunk in request.chunks:
            generated_text = self._generate_for_chunk(pipeline, chunk, request.examples_per_chunk)
            rows.extend(parse_teacher_jsonl_output(generated_text, expected_chunk_id=chunk.id))

        try:
            return validate_dataset(rows)
        except DatasetValidationError as exc:
            raise RealTeacherOutputError(
                "Real teacher output did not match the v0 JSONL schema: " + str(exc)
            ) from exc

    def _load_pipeline(self) -> Any:
        try:
            return self._pipeline_factory(
                task="text-generation",
                model=self.model_name,
                dtype="auto",
                device_map="auto",
            )
        except RealTeacherDependencyError:
            raise
        except ModuleNotFoundError as exc:
            raise RealTeacherDependencyError(
                f"Optional Hugging Face teacher dependency is missing: {exc.name or exc}"
            ) from exc
        except OSError as exc:
            raise RealTeacherModelLoadError(
                f"Could not download or load teacher model {self.model_name}: {exc}"
            ) from exc
        except Exception as exc:
            raise RealTeacherModelLoadError(
                f"Could not initialize teacher model {self.model_name}: {exc}"
            ) from exc

    def _generate_for_chunk(self, pipeline: Any, chunk: TextChunk, examples_per_chunk: int) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You create concise question-answer training rows from study notes. "
                    "Return only JSONL. Each line must be one JSON object with exactly "
                    "instruction, response, and source_chunk_id."
                ),
            },
            {"role": "user", "content": build_teacher_prompt(chunk, examples_per_chunk=examples_per_chunk)},
        ]
        try:
            response = pipeline(
                messages,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                return_full_text=False,
            )
        except Exception as exc:
            raise RealTeacherGenerationError(
                f"Teacher model generation failed for {chunk.id}: {exc}"
            ) from exc
        return _extract_generated_text(response)


def build_teacher_prompt(chunk: TextChunk, *, examples_per_chunk: int = 4) -> str:
    """Build the prompt shape used by the real teacher path."""

    if examples_per_chunk < 1:
        raise ValueError("examples_per_chunk must be at least 1")

    return (
        f"Create {examples_per_chunk} question-answer pairs for {chunk.id}.\n"
        "Return exactly one JSON object per line as JSONL.\n"
        "Return JSONL rows with exactly these fields: instruction, response, source_chunk_id.\n"
        "Use varied study-question styles in this order when possible: "
        + ", ".join(QUESTION_STYLE_GUIDE)
        + ".\n"
        "Each instruction should name the task naturally, and each response must be grounded in the source chunk.\n"
        "Do not invent facts outside the source chunk. Do not add any fields beyond the schema.\n\n"
        f"Source chunk:\n{chunk.text}"
    )


def parse_teacher_jsonl_output(
    generated_text: str,
    *,
    expected_chunk_id: str,
) -> list[dict[str, str]]:
    """Parse and validate JSONL rows from a real teacher model response."""

    for candidate in _candidate_jsonl_blocks(generated_text):
        try:
            rows = _parse_json_rows(candidate)
            validated = validate_dataset(rows)
            wrong_chunk_ids = sorted(
                {row["source_chunk_id"] for row in validated if row["source_chunk_id"] != expected_chunk_id}
            )
            if wrong_chunk_ids:
                raise RealTeacherOutputError(
                    "Real teacher output used unexpected source_chunk_id values: "
                    + ", ".join(wrong_chunk_ids)
                )
            return validated
        except RealTeacherOutputError:
            raise
        except Exception:
            continue

    raise RealTeacherOutputError("Real teacher output did not contain valid v0 JSONL rows.")


def explain_teacher_failure(exc: BaseException) -> list[str]:
    """Return beginner-readable next steps for optional real teacher failures."""

    message = str(exc)
    lowered = message.lower()
    lines = ["Real teacher generation failed."]

    if isinstance(exc, RealTeacherDependencyError) or "modulenotfounderror" in lowered:
        lines.append("A required optional Hugging Face package is missing or not importable.")
        lines.append("Run: " + build_pip_install_command())

    if isinstance(exc, RealTeacherModelLoadError):
        lines.append("The real teacher model could not be downloaded or loaded.")
        lines.append("Check that the Colab runtime has internet access, then rerun the teacher cell.")

    if isinstance(exc, RealTeacherGenerationError):
        if "out of memory" in lowered or ("cuda" in lowered and "memory" in lowered):
            lines.append("The GPU ran out of memory while running the real teacher.")
            lines.append("Restart the runtime, keep RUN_REAL_TEACHER = True, and rerun from setup.")
        elif "cuda" in lowered:
            lines.append("The real teacher needs a working CUDA GPU runtime in Colab.")
        else:
            lines.append("The real teacher model failed while generating rows.")

    if isinstance(exc, RealTeacherOutputError):
        lines.append("The real teacher output did not match the v0 JSONL schema.")
        lines.append("Keep MockTeacherEngine as the fallback and inspect the generated text before retrying.")

    if len(lines) == 1:
        lines.append("Read the error above, then rerun the teacher cell after checking setup.")

    return lines


def generate_mock_qa_pairs(
    chunks: Iterable[TextChunk],
    *,
    examples_per_chunk: int = 4,
) -> list[dict[str, str]]:
    """Generate safe deterministic QA rows without model calls or network access."""

    return MockTeacherEngine().generate(
        TeacherRequest(chunks=list(chunks), examples_per_chunk=examples_per_chunk)
    )


def _build_text_generation_pipeline(**kwargs: Any) -> Any:
    try:
        from transformers import pipeline
    except ModuleNotFoundError as exc:
        raise RealTeacherDependencyError(
            f"Optional Hugging Face teacher dependency is missing: {exc.name or exc}"
        ) from exc
    return pipeline(**kwargs)


def _extract_generated_text(response: Any) -> str:
    payload = response[0] if isinstance(response, list) and response else response
    if isinstance(payload, dict):
        generated = payload.get("generated_text", payload.get("text", ""))
    else:
        generated = payload

    if isinstance(generated, list) and generated:
        final_message = generated[-1]
        if isinstance(final_message, dict):
            return str(final_message.get("content", final_message.get("text", ""))).strip()
        return str(final_message).strip()

    if isinstance(generated, dict):
        return str(generated.get("content", generated.get("text", ""))).strip()

    return str(generated).strip()


def _candidate_jsonl_blocks(text: str) -> list[str]:
    stripped = text.strip()
    fenced = [
        match.group(1).strip()
        for match in re.finditer(r"```(?:jsonl|json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    ]
    return fenced + [stripped]


def _parse_json_rows(text: str) -> list[dict[str, object]]:
    stripped = text.strip()
    if not stripped:
        raise RealTeacherOutputError("Real teacher output was empty.")

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return [dict(parsed)]
    if isinstance(parsed, list):
        return [dict(item) for item in parsed if isinstance(item, dict)]

    rows: list[dict[str, object]] = []
    for line in stripped.splitlines():
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            parsed_line = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise RealTeacherOutputError("Real teacher JSONL contained an invalid JSON line.") from exc
        if not isinstance(parsed_line, dict):
            raise RealTeacherOutputError("Real teacher JSONL lines must be JSON objects.")
        rows.append(dict(parsed_line))

    if not rows:
        raise RealTeacherOutputError("Real teacher output did not contain JSON objects.")
    return rows


def _generate_rows(
    chunks: Iterable[TextChunk],
    *,
    examples_per_chunk: int,
) -> list[dict[str, str]]:
    if examples_per_chunk < 1:
        raise ValueError("examples_per_chunk must be at least 1")

    rows: list[dict[str, str]] = []
    for chunk in chunks:
        excerpt = _excerpt(chunk.text)
        fact_pairs = _extract_key_value_facts(chunk.text)
        templates = _fact_templates(chunk, fact_pairs) if fact_pairs else _excerpt_templates(chunk, excerpt)

        for example_index in range(examples_per_chunk):
            question, answer = templates[example_index % len(templates)]
            if example_index >= len(templates):
                if fact_pairs:
                    label, value = fact_pairs[example_index % len(fact_pairs)]
                    question = f"What exact value is attached to {label.lower()} in {chunk.id}?"
                    answer = f"The exact value attached to {label.lower()} is {value}."
                else:
                    question = f"What supporting fact {example_index + 1} appears in {chunk.id}?"
                    answer = f"A supporting fact from the source text is: {excerpt}"
            rows.append(
                {
                    "instruction": question,
                    "response": answer,
                    "source_chunk_id": chunk.id,
                }
            )

    return validate_dataset(rows)


def _excerpt_templates(chunk: TextChunk, excerpt: str) -> list[tuple[str, str]]:
    return [
        (
            f"Factual recall: what detail from {chunk.id} should be remembered?",
            f"The key factual detail is: {excerpt}",
        ),
        (
            f"Explain the main idea of {chunk.id} in plain language.",
            f"In plain language, the source says: {excerpt}",
        ),
        (
            f"Flashcard: what should the front and back say for {chunk.id}?",
            f"Front: What does this note say? Back: {excerpt}",
        ),
        (
            f"What misconception should be avoided after reading {chunk.id}?",
            f"Do not misread the note as saying something unsupported. The grounded point is: {excerpt}",
        ),
    ]


def _fact_templates(chunk: TextChunk, fact_pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    style_builders = (
        lambda label, value: (
            f"What exact value does {chunk.id} give for {label}?",
            f"The {label} is {value}.",
        ),
        lambda label, value: (
            f"Answer plainly: identify {label} from the note.",
            f"The note identifies {label} as {value}.",
        ),
        lambda label, value: (
            f"Create a front/back flashcard for {label} from {chunk.id}.",
            f"Front: What is {label}? Back: {value}.",
        ),
        lambda label, value: (
            f"Which made-up value should be rejected for {label} in {chunk.id}?",
            f"Reject invented values; the source says {label} is {value}.",
        ),
        lambda label, value: (
            f"Use only the source note: what should someone remember about {label}?",
            f"Remember that {label} is {value}.",
        ),
        lambda label, value: (
            f"In a review quiz for {chunk.id}, how should {label} be answered?",
            f"Answer with {value} for {label}.",
        ),
    )

    templates: list[tuple[str, str]] = []
    for index in range(len(fact_pairs) * len(style_builders)):
        fact_index = index % len(fact_pairs)
        style_index = index % len(style_builders)
        label, value = fact_pairs[fact_index]
        normalized_label = label.lower()
        templates.append(style_builders[style_index](normalized_label, value))
    return templates


def _extract_key_value_facts(text: str) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        cleaned_line = raw_line.strip().lstrip("-*").strip()
        for sentence in re.split(r"(?<=\.)\s+", cleaned_line):
            candidate = sentence.strip()
            if ":" not in candidate or candidate.startswith("#"):
                continue
            label, value = candidate.split(":", 1)
            label = label.strip()
            value = value.strip().rstrip(".")
            if not label or not value:
                continue
            if len(label.split()) > 6:
                continue
            facts.append((label, value))
    return facts


def _excerpt(text: str, *, max_chars: int = 220) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 1].rstrip() + "..."
