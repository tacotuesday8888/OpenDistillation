"""Deterministic local mock teacher for the v0 notebook skeleton."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re
from typing import Protocol

from .dataset import validate_dataset
from .text import TextChunk


@dataclass(frozen=True)
class TeacherRequest:
    """Input for a teacher engine.

    Later real teacher engines can use the same request shape while deciding
    whether prompts run locally or through a remote open-source endpoint.
    """

    chunks: list[TextChunk]
    examples_per_chunk: int = 2


class TeacherEngine(Protocol):
    """Interface for mock and future real teacher-generation engines."""

    name: str
    sends_data_remote: bool

    def generate(self, request: TeacherRequest) -> list[dict[str, str]]:
        """Generate validated dataset rows from text chunks."""


class MockTeacherEngine:
    """Safe deterministic teacher used by the notebook skeleton."""

    name = "mock-local-teacher"
    sends_data_remote = False

    def generate(self, request: TeacherRequest) -> list[dict[str, str]]:
        return _generate_rows(request.chunks, examples_per_chunk=request.examples_per_chunk)


def build_teacher_prompt(chunk: TextChunk, *, examples_per_chunk: int = 2) -> str:
    """Build the prompt shape a real teacher path can use later."""

    if examples_per_chunk < 1:
        raise ValueError("examples_per_chunk must be at least 1")

    return (
        f"Create {examples_per_chunk} question-answer pairs for {chunk.id}.\n"
        "Return JSONL rows with exactly these fields: instruction, response, source_chunk_id.\n\n"
        f"Source chunk:\n{chunk.text}"
    )


def generate_mock_qa_pairs(
    chunks: Iterable[TextChunk],
    *,
    examples_per_chunk: int = 2,
) -> list[dict[str, str]]:
    """Generate safe deterministic QA rows without model calls or network access."""

    return MockTeacherEngine().generate(
        TeacherRequest(chunks=list(chunks), examples_per_chunk=examples_per_chunk)
    )


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
        templates = [
            (
                f"What is the main point of {chunk.id}?",
                f"The main point is: {excerpt}",
            ),
            (
                f"Which detail from {chunk.id} should be remembered?",
                f"Remember this detail from the source text: {excerpt}",
            ),
        ]

        for example_index in range(examples_per_chunk):
            question, answer = templates[example_index % len(templates)]
            if example_index >= len(templates):
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


def _excerpt(text: str, *, max_chars: int = 220) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 1].rstrip() + "..."
